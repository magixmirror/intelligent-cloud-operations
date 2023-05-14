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
import re
from itertools import cycle
from pathlib import Path

import holoviews as hv
import pandas as pd
from bokeh.io import export_png, export_svg
from constants import DEFAULT_IMG_DEST, NOTEBOOKS_CONFIG_FILE

hv.extension("bokeh")
pd.options.plotting.backend = "holoviews"

# %% [markdown]
# ## Read config files

# %%
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
time_stats_freq = results_config_dict.get("time_stats_freq", "1min")
time_stats_legend_pos = results_config_dict.get("time_stats_legend_pos", "top_right")

# %% [markdown]
# ## Client-side response times plots

# %%
# NOTE: not using hv.Cycle because multiple hv.Curve with the same color are required
color_cycle = cycle(
    [
        "#d62728",
        "#e5ae38",
        "#6d904f",
        "#8b8b8b",
        "#17becf",
        "#9467bd",
        "#e377c2",
        "#8c564b",
        "#bcbd22",
        "#1f77b4",
        # "#30a2da", # blue
    ]
)

opts = [hv.opts.Curve(tools=["hover"])]
opts_scatter = hv.opts.Scatter(size=5, marker="o", tools=["hover"])

times_file = list(data_root.glob("*-times.csv"))[0] # NOTE: assuming only one matching file in dir
print(f"Reading from '{times_file}'...")

run_basename = times_file.name.split("-times.csv")[0]
df = pd.read_csv(times_file, header=None, names=["timestamp", "delay"])

# drop rows containing 0 beacuse:
# - timestamp == 0 means the request was never sent
# - delay == 0 means the response was never received
len_before = len(df)
df.drop(df[(df["timestamp"] == 0) | (df["delay"] == 0)].index, inplace=True)
len_after = len(df)
print(f"dropped {len_before - len_after}/{len_before} rows.")

# set datetimeindex for easy resampling
df["index"] = pd.to_datetime(df["timestamp"], unit="us")
df.set_index("index", inplace=True)

# convert microsec to millisec
df = df / 1000

# convert timestamp to min
df["timestamp"] = df["timestamp"] / 1000 / 60

if run_time_limit:
    df = df[df["timestamp"] < run_time_limit]

# read run metadata
metadata_file = (data_root / f"{run_basename}-metadata.json").resolve()
with open(metadata_file, "r+") as f:
    metadata = json.load(f)

## filter out outliers before plotting
df = df[df["delay"] > 0]
df = df[df["delay"] <= df["delay"].quantile(0.999)]

# rolling stats
df_quant = pd.DataFrame()
traces = []
quantiles = [
    0.5,
    0.9,
    # 0.95,
    0.99,
    # 0.995,
    # 0.999,
]
time_amount, time_unit = (
    re.compile("^(\d+)([a-zA-Z]+)$").search(time_stats_freq).groups()
)
time_amount = int(time_amount)
for q in quantiles:
    df_res = (
        df["delay"]
        .resample(time_stats_freq)#, closed="right", label="right")
        .quantile(q)
    )
    # df_res.index = [x * time_amount for x in range(1, df_res.index.size + 1)]
    df_res.index = [x * time_amount for x in range(0, df_res.index.size)]
    q_label = f"p{q*100:g}"
    df_quant[q_label] = df_res

    trace_label = f"{time_stats_freq} {q_label}"
    curr_color = next(color_cycle)
    traces.append(
        hv.Scatter(
            (df_res.index, df_res.values),
            label=trace_label,
        )
        .opts(color=curr_color)
        .opts(opts_scatter)
    )
    traces.append(
        hv.Curve(
            (df_res.index, df_res.values),
            label=trace_label,
        ).opts(color=curr_color)
    )

    # add signal avg
    avg = df_res.mean()
    traces.append(
        hv.Curve(
            [
                [df_res.index[0], avg],
                [df_res.index[-1], avg]
            ],
            label=f"{trace_label} (avg)",
        ).opts(
            color=curr_color,
            line_dash="dashed"
        )
    )

title = f"{data_root.name}-{run_basename}-times-{time_stats_freq}"
times_fig = (
    hv.Overlay(traces)
    .opts(opts)
    .opts(
        width=950,
        height=550,
        show_grid=True,
        title=title,
        xlabel=f"time [{time_unit}]",
        ylabel="delay [ms]",
        legend_position=time_stats_legend_pos,
        legend_opts={"background_fill_alpha": 0.5},
        fontsize={
            "title": 13,
            "legend": 16,
            "labels": 20,
            "xticks": 20,
            "yticks": 20,
        },
        logy=True,
    )
)
times_fig

# %% [markdown]
# ## Save stats

# %%
origin_time = (
    pd.Timestamp(metadata["start_real"])
    .tz_convert(None)
    .floor(time_unit) # NOTE: to align timestamps to the start of the period
)

# %%
if time_unit == "min":
    # NOTE: convert to seconds because pd.to_datetime() doesn't support 'min' frequency
    df_quant.index = pd.to_datetime(df_quant.index * 60, unit="s", origin=origin_time)
else:
    df_quant.index = pd.to_datetime(df_quant.index, unit=time_unit, origin=origin_time)

# df_quant

# %%
df_quant.index.name = "timestamp"
df_quant.to_csv(data_root / f"{run_basename}-times-{time_stats_freq}.csv")

# %% [markdown]
# ## Save plots

# %%
# NOTE: save function is only available if the additional system dependencies are installed.

plot = hv.renderer("bokeh").get_plot(times_fig.options(toolbar=None))
# plot = hv.plotting.mpl.MPLRenderer.instance().get_plot(times_fig.options(toolbar=None))

# plot.output_backend = "svg"
# export_svg(plot.state, filename=f"{title}.svg")

png_file = default_img_dest / f"{title}.png"
print(f"Saving plot to '{png_file}'...")
export_png(plot.state, filename=png_file)

# %%
