# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -tags,-jupyter,-jp-MarkdownHeadingCollapsed
#     comment_magics: true
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version,-jupytext.text_representation.format_version
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: pred-ops-os
#     language: python
#     name: pred-ops-os
# ---

# %%
import json
from pathlib import Path

import holoviews as hv
import pandas as pd
from bokeh.io import export_png
from constants import DEFAULT_DATA_ROOT, DEFAULT_IMG_DEST, NOTEBOOKS_CONFIG_FILE
from intops_utils import make_spatial_aggregates
from monasca_utils import json_to_df

hv.extension("bokeh")
pd.options.plotting.backend = "holoviews"

# %% [markdown]
# ## Read config files

# %%
default_data_root = Path(DEFAULT_DATA_ROOT).resolve()
default_img_dest = Path(DEFAULT_IMG_DEST).resolve()

results_config_dict = dict()
results_config_file = Path(NOTEBOOKS_CONFIG_FILE)
if results_config_file.exists():
    with open(results_config_file, "r") as f:
        results_config_dict = json.load(f)
else:
    raise ValueError(f"config file '{results_config_file}' not found")

data_root = results_config_dict.get("data_root", None)
if data_root is not None:
    data_root = Path(data_root)
else:
    raise ValueError(f"'data_root' property not found")

run_time_limit = results_config_dict.get("run_time_limit", None)
metric_names = results_config_dict.get("metric_names", list())

# %% [markdown]
# ## Read run results

# %%
metric_names_norm = [x.replace(".", "-") for x in metric_names]

# read run metadata
metadata_file = list(data_root.glob("*-metadata.json"))[
    0
]  # NOTE: assuming only one matching file in dir
with open(metadata_file, "r+") as f:
    metadata = json.load(f)

run_basename = metadata_file.name.split("-metadata.json")[0]

# load .json export files into DFs
df_list = list()
for metric_norm in metric_names_norm:
    metric_file = (data_root / f"{run_basename}_{metric_norm}.json").resolve()

    print(f"reading from '{metric_file}'...")

    df_list.append(json_to_df(metric_file))

# %% [markdown]
# ## Build plots

# %%
fig_list = []
title_list = []

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
        # '#fc4f30',  # red-ish
        # '#d62728',  # red-ish
    ]
)

opts = [hv.opts.Curve(tools=["hover"])]
opts_scatter = hv.opts.Scatter(size=5, marker="o", tools=["hover"])
opts_scatter_cross = hv.opts.Scatter(size=15, marker="x", tools=["hover"])
opts_scatter_diam = hv.opts.Scatter(size=12, marker="d", tools=["hover"])
opts_overlay = {
    "width": 950,
    "height": 550,
    "show_grid": True,
    "xlabel": "time [min]",
    "legend_position": "top_left",
    # "legend_cols": 2, # still buggy: https://github.com/holoviz/holoviews/issues/3780
    "legend_opts": {"background_fill_alpha": 0.5},  # 'padding': 20, 'spacing': 1},
    "fontsize": {
        "title": 13,
        "legend": 16,
        "labels": 20,
        "xticks": 20,
        "yticks": 20,
    },
    "padding": 0.05,
}

for metric, metric_norm, metric_df in zip(metric_names, metric_names_norm, df_list):
    traces = []

    ### data manipulation ###
    table = pd.pivot_table(
        metric_df,
        values=metric,
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

    table.reset_index(inplace=True)

    # truncate data & remove NaN-only cols
    if run_time_limit:
        table = table.iloc[:run_time_limit, :].dropna(axis=1, how="all")

    # save to .csv
    csv_dump_file = data_root / f"{run_basename}_{metric_norm}.csv"
    print(f"Saving to {csv_dump_file} ...")
    table.to_csv(csv_dump_file, index=False)

    # insert distwalk trace data to align timestamps
    # load_file_basename = metadata["load_profile"]
    # load_file = (default_data_root / load_file_basename).resolve()
    # print(f"reading from {load_file} ...")
    # load_df = pd.read_csv(load_file, header=None, names=["distwalk"])
    # table = table.join(load_df, how="left")

    # insert anomaly schedule to align timestamps
    anomaly_type = metadata.get("anomaly_type")
    if anomaly_type is not None:
        anomaly_file_basename = metadata.get("anomaly")
        if anomaly_file_basename is not None:
            anomaly_file = (default_data_root / anomaly_file_basename).resolve()
        else:
            anomaly_file = (data_root / f"{run_basename}-{anomaly_type}-sched.dat").resolve()

        print(f"reading from {anomaly_file} ...")
        anomaly_df = pd.read_csv(anomaly_file, header=None, names=["injection"])
        table = table.join(anomaly_df, how="left")

    ### plot customization ###
    # plot scale-out threshold
    # traces.append(hv.HLine(80).opts(color="black", line_dash="dashed"))

    # plot distwalk trace
    # distwalk_trace_label = "distwalk"
    # traces.append(
    #     hv.Scatter(
    #         (table.index, table["distwalk"].values),
    #         label=distwalk_trace_label,
    #     )
    #     .opts(color=color_cycle)
    #     .opts(opts_scatter)
    # )
    # traces.append(
    #     hv.Curve(
    #         (table.index, table["distwalk"].values),
    #         label=distwalk_trace_label,
    #     ).opts(color=color_cycle)
    # )

    # plot metrics observed by VMs
    instance_idx = 0
    for group_label in orig_cols:
        if group_label in table.columns:
            load_trace_label = f"VM {instance_idx}"
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
    if anomaly_type is not None:
        traces.append(
            hv.Scatter(
                table[["injection"]].replace(0, None).replace(1, 0),
                label="injection",
            )
            .opts(color="black")
            .opts(opts_scatter_cross)
        )

    title = f"{data_root.name}-{run_basename}_{metric_norm}"
    fig = (
        hv.Overlay(traces)
        .opts(opts)
        .opts(**opts_overlay)
        .opts(
            title=title,
            ylabel=metric,
        )
    )
    fig_list.append(fig)
    title_list.append(title)

layout = hv.Layout(fig_list).cols(1).opts(shared_axes=False)
layout

# %% [markdown]
# ## Consolidate load traces

# %%
res_df = None
for metric, metric_norm in zip(metric_names, metric_names_norm):
    metric_csv_file = data_root / f"{run_basename}_{metric_norm}.csv"
    print(f"Reading '{metric_csv_file}'...")

    metric_df = pd.read_csv(metric_csv_file, index_col=["timestamp"])
    metric_df.rename(lambda x: f"{x}.{metric}", axis=1, inplace=True)

    if res_df is None:
        res_df = metric_df
    else:
        res_df = res_df.join(metric_df, how="outer")

res_csv_file = data_root / f"{run_basename}-load.csv"
print(f"Saving consolidated traces to '{res_csv_file}'...")
res_df.to_csv(res_csv_file)

# %%
# res_df.head()

# %% [markdown]
# ## Spatial aggregations

# %%
agg_df = make_spatial_aggregates(res_df).reset_index(drop=True)
agg_title = f"{data_root.name}-{run_basename}-load-agg"
agg_fig = (
    agg_df.plot()
    .opts(opts)
    .opts(**opts_overlay)
    .opts(
        title=agg_title,
        legend_position="top_right",
        legend_opts={"background_fill_alpha": 0.5, "title": ""},
        # logy=True
    )
)
agg_fig

# %% [markdown]
# ## Save plots

# %%
# NOTE: automatically save individual plots to disk as they are rendered in the above layout
# (e.g., possibly with a shared axis). Save function is only available if the additional
# system dependencies are installed.

plot = hv.renderer("bokeh").get_plot(layout)
subplots = plot.state.children[1].children

for _subplot, _title in zip(subplots, title_list):
    png_file = default_img_dest / f"{_title}.png"
    print(f"Saving plot to '{png_file}'...")
    export_png(_subplot[0], filename=png_file)

# %%
# aggregates plot
agg_plot = hv.renderer("bokeh").get_plot(agg_fig.options(toolbar=None))
png_file = default_img_dest / f"{agg_title}.png"
print(f"Saving plot to '{png_file}'...")
export_png(agg_plot.state, filename=png_file)

# %%
