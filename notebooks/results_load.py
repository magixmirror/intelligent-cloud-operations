# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pred-ops-os
#     language: python
#     name: pred-ops-os
# ---

# %%
import json
from datetime import timedelta
from itertools import zip_longest

import holoviews as hv
import pandas as pd
from constants import (
    DATA_ROOT,
    DATETIME_FORMAT,
    FAULT_RUNS,
    STRESS_RUNS,
)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

hv.extension("bokeh")
pd.options.plotting.backend = "holoviews"

# %%
stress_runs_lim = (7, None)
# stress_runs_lim = (20, 20)
stress_runs_exclude = set([6])
fault_runs_lim = (1, None)
# fault_runs_lim = (2, 2)
fault_runs_exclude = set([15])

# run_time_limit = None
run_time_limit = 66

# %% tags=[]
# load .json export files into DFs
df_list = list()

for real_file, pred_file in zip_longest(
    sorted(DATA_ROOT.glob("*-real.json")), sorted(DATA_ROOT.glob("*-pred.json"))
):
    if real_file:
        real_file = real_file.resolve()
    if pred_file:
        pred_file = pred_file.resolve()
    i = int(real_file.name.split("-")[-2])

    if "-stress-" in real_file.name:
        if i in stress_runs_exclude:
            continue
        if stress_runs_lim[0] is not None and i < stress_runs_lim[0]:
            continue
        if stress_runs_lim[1] is not None and i > stress_runs_lim[1]:
            continue
        if stress_runs_lim[0] is None and stress_runs_lim[1] is None:
            continue
    elif "-fault-" in real_file.name:
        if i in fault_runs_exclude:
            continue
        if fault_runs_lim[0] is not None and i < fault_runs_lim[0]:
            continue
        if fault_runs_lim[1] is not None and i > fault_runs_lim[1]:
            continue
        if fault_runs_lim[0] is None and fault_runs_lim[1] is None:
            continue

    print(f"reading from {real_file} and {pred_file} ...")

    with open(real_file, "r+") as fp:
        real_json_body = json.load(fp)

    real_metric = real_json_body[0]["name"]

    real_df = pd.DataFrame(
        columns=["timestamp", "resource_id", "hostname", real_metric]
    )
    for item in real_json_body:
        resource_id = item["dimensions"]["resource_id"]
        hostname = item["dimensions"]["hostname"]
        measurement_list = item["measurements"]

        real_df = pd.concat(
            [
                real_df,
                pd.DataFrame(
                    [
                        pd.Series(
                            [m[0], resource_id, hostname, m[1]], index=real_df.columns
                        )
                        for m in measurement_list
                    ]
                ),
            ]
        )
    real_df = real_df.astype(
        {
            "resource_id": "string",
            "hostname": "string",
            real_metric: "float64",
        }
    )

    # cast index to DateTimeIndex
    real_df.set_index(["timestamp"], inplace=True)
    real_df.index = pd.to_datetime(real_df.index, format=DATETIME_FORMAT)

    label = real_file.name.split("-real.json")[0]
    df_list.append((label, real_df))

# %% tags=[]
fig_list = []
label_list = []
color_cycle = hv.Cycle(
    [
        "#30a2da",
        "#e5ae38",
        "#6d904f",
        "#8b8b8b",
        "#17becf",
        "#9467bd",
        "#e377c2",
        "#8c564b",
        "#bcbd22",
        "#1f77b4",
    ]
)
opts = [hv.opts.Curve(tools=["hover"])]
opts_scatter = hv.opts.Scatter(size=5, marker="o", tools=["hover"])
opts_scatter_cross = hv.opts.Scatter(size=10, marker="x", tools=["hover"])
opts_scatter_diam = hv.opts.Scatter(size=12, marker="d", tools=["hover"])

for label, real_df in df_list:
    traces = []
    index = int(label[-2:])

    if "-stress-" in label:
        mapping = STRESS_RUNS
    elif "-fault-" in label:
        mapping = FAULT_RUNS

    ### data manipulation ###
    table = pd.pivot_table(
        real_df,
        values="cpu.utilization_perc",
        index=["timestamp"],
        columns=["hostname"],
    )
    # table.index = pd.to_datetime(table.index, format=DATETIME_FORMAT)
    table = table.resample("1min").mean()

    # reorder columns by VMs start time
    table = table[
        table.apply(pd.Series.first_valid_index).sort_values().index.to_list()
    ]

    orig_cols = table.columns.copy()
    orig_cols_num = len(orig_cols)

    # compute spatial statistics
    table["count"] = (~table.isnull()).iloc[:, 0:orig_cols_num].sum(axis=1)
    table["sum"] = table.iloc[:, 0:orig_cols_num].sum(axis=1)
    table["mean"] = table["sum"] / table["count"]
    table["std"] = (
        ((table.iloc[:, 0:orig_cols_num].subtract(table["mean"], axis=0)) ** 2).sum(
            axis=1
        )
        / table["count"]
    ) ** 0.5

    table.reset_index(inplace=True)

    # insert distwalk trace data to align timestamps
    load_file_basename = mapping[index]["load_profile"]
    load_file = (DATA_ROOT / load_file_basename).resolve()
    print(f"reading from {load_file} ...")
    load_df = pd.read_csv(load_file, header=None, names=["distwalk"])
    table = table.join(load_df / 10, how="outer")

    # insert stress-ng schedule to align timestamps
    stress_file_basename = mapping[index].get("stress")
    if stress_file_basename is not None:
        stress_file = (DATA_ROOT / stress_file_basename).resolve()
        print(f"reading from {stress_file} ...")
        stress_df = pd.read_csv(stress_file, header=None, names=["stress"])
        table = table.join(stress_df, how="outer")

    # insert fault schedule to align timestamps
    fault_file_basename = mapping[index].get("fault")
    if fault_file_basename is not None:
        fault_file = (DATA_ROOT / fault_file_basename).resolve()
        print(f"reading from {fault_file} ...")
        fault_df = pd.read_csv(fault_file, header=None, names=["fault"])
        table = table.join(fault_df, how="outer")

    # truncate data & remove NaN-only cols
    if run_time_limit:
        table = table.iloc[:run_time_limit, :].dropna(axis=1, how="all")

    # save to .csv
    csv_dump_file = DATA_ROOT / f"{label}.csv"
    if csv_dump_file.exists():
        print(f"{csv_dump_file} exists, skipping...")
    else:
        print(f"Saving to {csv_dump_file} ...")
        table.to_csv(csv_dump_file, index=False)

    ### plot customization ###
    # plot distwalk trace
    distwalk_trace_label = "distwalk thread trace"
    traces.append(
        hv.Scatter(
            (table.index, table["distwalk"].values),
            label=distwalk_trace_label,
        )
        .opts(color=color_cycle)
        .opts(opts_scatter)
    )
    traces.append(
        hv.Curve(
            (table.index, table["distwalk"].values),
            label=distwalk_trace_label,
        ).opts(color=color_cycle)
    )

    # plot metrics observed by VMs
    instance_idx = 0
    for group_label in orig_cols:
        if group_label in table.columns:
            load_trace_label = f"instance {instance_idx}"
            traces.append(
                hv.Scatter(
                    (table.index, table[group_label].values),
                    label=load_trace_label,
                    kdims=[],
                )
                .opts(color=color_cycle)
                .opts(opts_scatter)
            )
            traces.append(
                hv.Curve(
                    (table.index, table[group_label].values),
                    label=load_trace_label,
                ).opts(color=color_cycle)
            )
            instance_idx += 1

    # plot stress schedule
    if stress_file_basename is not None:
        stress_trace_label = "stress-ng schedule"
        traces.append(
            hv.Scatter(
                (table.index, table["stress"].values),
                label=stress_trace_label,
            )
            .opts(color="black")
            .opts(opts_scatter_cross)
        )

    # plot fault schedule
    if fault_file_basename is not None:
        fault_trace_label = "fault schedule"
        traces.append(
            hv.Scatter(
                (table.index, table["fault"].values),
                label=fault_trace_label,
            )
            .opts(color="black")
            .opts(opts_scatter_cross)
        )

    title = f"{label} - load: {load_file_basename}"
    if stress_file_basename is not None:
        title += f" | stress: {stress_file_basename}"
    if fault_file_basename is not None:
        title += f" | fault: {fault_file_basename}"

    fig = (
        hv.Overlay(traces)
        .opts(
            width=950,
            height=550,
            show_grid=True,
            title=title,
            xlabel="time [min]",
            ylabel="CPU usage [%]",
            legend_position="top_left",
            # legend_cols=2, # still buggy: https://github.com/holoviz/holoviews/issues/3780
            legend_opts={"background_fill_alpha": 0.5, "padding": 20, "spacing": 1},
            fontsize={
                "title": 13,
                "legend": 12,
                "labels": 15,
                "xticks": 13,
                "yticks": 13,
            },
            # logy=True,
            padding=0.05,
        )
        .opts(opts)
    )
    fig_list.append(fig)
    label_list.append(label)

layout = hv.Layout(fig_list).cols(1).opts(shared_axes=False)
layout

# %%
