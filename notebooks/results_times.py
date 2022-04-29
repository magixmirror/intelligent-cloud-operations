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

import holoviews as hv
import pandas as pd
from constants import (
    DATA_ROOT,
    FAULT_RUNS,
    STRESS_RUNS,
)
from holoviews.operation.datashader import datashade

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

# %% [markdown]
# ## Client-side response times plots

# %%
descr_stats_table = pd.DataFrame()
times_fig_list = []
times_label_list = []
for times_file in sorted(DATA_ROOT.glob("*-times.csv")):
    ## filter-out excluded result files
    i = int(times_file.name.split("-")[-2])
    if "-stress-" in times_file.name:
        if i in stress_runs_exclude:
            continue
        if stress_runs_lim[0] is not None and i < stress_runs_lim[0]:
            continue
        if stress_runs_lim[1] is not None and i > stress_runs_lim[1]:
            continue
        if stress_runs_lim[0] is None and stress_runs_lim[1] is None:
            continue
        label = "Stress"
        mapping = STRESS_RUNS
    elif "-fault-" in times_file.name:
        if i in fault_runs_exclude:
            continue
        if fault_runs_lim[0] is not None and i < fault_runs_lim[0]:
            continue
        if fault_runs_lim[1] is not None and i > fault_runs_lim[1]:
            continue
        if fault_runs_lim[0] is None and fault_runs_lim[1] is None:
            continue
        label = "Fault"
        mapping = FAULT_RUNS

    traces = []

    print(times_file)
    df = pd.read_csv(times_file, header=None, names=["timestamp", "delay"])

    # drop rows containing 0 beacuse:
    # - timestamp == 0 means the request was never sent
    # - delay == 0 means the response was never received
    len_before = len(df)
    df.drop(df[(df["timestamp"] == 0) | (df["delay"] == 0)].index, inplace=True)
    len_after = len(df)
    print(f"dropped {len_before - len_after}/{len_before} rows.")

    # convert microsec to millisec
    df = df / 1000

    # convert timestamp to min
    df["timestamp"] = df["timestamp"] / 1000 / 60

    if run_time_limit:
        df = df[df["timestamp"] < run_time_limit]

    # estract run parameters
    stats_col_name = f"{label}"
    input_size = mapping[i].get("input_size")
    vm_delay_min = mapping[i].get("vm_delay_min")
    if input_size:
        stats_col_name += f" ({input_size:02})"

    ## compute percentiles
    # overall stats
    descr_stats_table[stats_col_name] = pd.Series(
        {
            "avg (ms)": df["delay"].mean(),
            "p90 (ms)": df["delay"].quantile(0.9),
            "p95 (ms)": df["delay"].quantile(0.95),
            "p99 (ms)": df["delay"].quantile(0.99),
            "p99.5 (ms)": df["delay"].quantile(0.995),
            "p99.9 (ms)": df["delay"].quantile(0.999),
        }
    )

    ## filter out outliers before plotting
    df = df[df["delay"] > 0]
    df = df[df["delay"] <= df["delay"].quantile(0.999)]

    ## build scatter plot
    times_fig = hv.Overlay(
        [
            hv.Scatter(
                (df["timestamp"].values, df["delay"].values),
                label=stats_col_name,
            )
        ]
    ).opts(
        width=950,
        height=550,
        show_grid=True,
        title=f"{times_file.name}",
        xlabel="time [min]",
        ylabel="delay [ms]",
        legend_position="top_left",
        fontsize={
            "title": 15,
            "legend": 15,
            "labels": 15,
            "xticks": 13,
            "yticks": 13,
        },
        logy=True,
    )
    times_fig_list.append(times_fig)
    times_label_list.append(times_file.stem)

times_layout = datashade(hv.Layout(times_fig_list).cols(1).opts(shared_axes=False))
times_layout

# %% [markdown] tags=[]
# ## Client-side response times tables

# %%
# render client-side delays stats table
ordered_groups = ["Stress", "Fault"]
ordered_cols = []
for group in ordered_groups:
    ordered_cols += (
        descr_stats_table.filter(regex=f"^{group}", axis=1)
        .columns.sort_values()
        .to_list()
    )

printable_table = descr_stats_table[ordered_cols].T
col_fmt = "r" + "r" * printable_table.columns.size
print(printable_table.round(2).style.to_latex(column_format=col_fmt, hrules=True))
printable_table

# %%
