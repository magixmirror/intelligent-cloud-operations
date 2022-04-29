# ---
# jupyter:
#   jupytext:
#     comment_magics: true
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
from datetime import datetime

import holoviews as hv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from constants import DATA_ROOT
from madi.detectors.neg_sample_random_forest import NegativeSamplingRandomForestAd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, binarize

hv.extension("bokeh")
pd.options.plotting.backend = "holoviews"

# %%
# plotting
opts_global = [
    # hv.opts.Scatter(size=5, marker="o"),
    hv.opts.Curve(tools=["hover"]),
]
opts_overlay = {
    "width": 950,
    "height": 550,
    "show_grid": True,
    "xlabel": "time [min]",
    "ylabel": "CPU usage [%]",
    "legend_position": "top_left",
    "fontsize": {
        "title": 13,
        "legend": 12,
        "labels": 15,
        "xticks": 13,
        "yticks": 13,
    },
    # logy=True,
    # padding=0.05,
}

# noise
rng = np.random.default_rng(None)
rng_repr = np.random.default_rng(42)

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Training data

# %%
vms_num = 3
days = 7
total_minutes = 1440 * days
# total_minutes = 300

# train_end = 200
# train_end = int(total_minutes * 0.7)
train_end = total_minutes

# limit = 2880
# limit = None


# generate ideal load
x = np.arange(1, total_minutes + 1, 1)

# cpu_sum = (((1 - np.cos(np.pi * x / 50)) * 0.6 * (1 + np.sin(np.pi * x / 11))) * 100) + 60
period_days = 28
period_hours = 1
period_minutes = 60
cpu_sum = (
    0.03
    + 2.9
    * (1 - np.sin(np.pi * x / period_hours / period_minutes))
    / 2
    * (2 - np.cos(np.pi * x / period_days / period_hours / period_minutes))
    / 4
    + 1
    * ((1 - np.cos(np.pi * x / period_days / period_hours / period_minutes)) / 2) ** 4
) * 100

cpu_vm_ideal = cpu_sum / vms_num

# %%
(
    (
        hv.HLine(100).opts(color="black", line_dash="dashed")
        # * hv.VLine(train_end).opts(color="blue", line_dash="dashed")
        * (
            hv.Curve((x, cpu_vm_ideal), label="ideal")
            * hv.Curve((x, cpu_sum), label="ideal_sum")
        ).opts([hv.opts.Curve(line_dash="dashed")])
    )
    .opts(opts_global)
    .opts(**opts_overlay)
)

# %%
# define noise scale for individual vm traces
noise_scale = np.std(cpu_vm_ideal) * 0.01
# print(noise_scale)

# generate noisy vm traces
vm_traces = np.vstack([cpu_vm_ideal.copy()] * vms_num)
vm_traces_noised = np.clip(
    vm_traces + rng.normal(scale=noise_scale, size=vm_traces.shape), 0, 100
)
vm_traces_df = pd.DataFrame(
    data=vm_traces_noised.T,
    columns=[f"vm{i+1}" for i in range(vm_traces_noised.shape[0])],
    index=pd.Index(x - 1, name="timestamp"),
)

# %%
(
    (
        hv.HLine(100).opts(color="black", line_dash="dashed")
        # * hv.VLine(train_end).opts(color="blue", line_dash="dashed")
        * (
            hv.Curve((vm_traces_df.index, cpu_vm_ideal), label="ideal")
            # * hv.Curve((vm_traces_df.index, cpu_sum), label="ideal_sum")
            * hv.Curve((vm_traces_df.index, vm_traces_df.mean(axis=1)), label="avg")
            * hv.Curve((vm_traces_df.index, vm_traces_df.std(axis=1)), label="std")
            # * hv.Curve((vm_traces_df.index, vm_trace_df.quantile(.9, axis=1)), label="p90")
            # * hv.Scatter(
            #     (vm_traces_df.index, faults_presence * 50), label="labelling"
            # ).opts(marker="o")
        ).opts([hv.opts.Curve(line_dash="dashed")])
        * vm_traces_df.plot()
    )
    .opts(opts_global)
    .opts(**opts_overlay)
)

# %% [markdown] tags=[]
# ## Anomaly Detection

# %% [markdown] tags=[]
# ### Preprocessing

# %%
# TODO: handle traces containing missing obs (e.g., VMs started later wrt others)
# extract aggregated features
vm_traces_agg_df = vm_traces_df.agg(
    [
        "mean",
        "std",
        # "count",  # might be relevant when dealing with elastic groups
    ],
    axis=1,
)

# %%
# apply scaling
scaler = StandardScaler()
vm_traces_agg_train = scaler.fit_transform(vm_traces_agg_df.iloc[:train_end, :].values)
# vm_traces_agg_test = scaler.transform(vm_traces_agg_df.iloc[train_end:, :].values)

# %%
(
    hv.VLine(train_end).opts(color="blue", line_dash="dashed")
    * pd.DataFrame(
        # data=np.vstack((vm_traces_agg_train, vm_traces_agg_test)),
        data=vm_traces_agg_train,
        columns=vm_traces_agg_df.columns.copy(),
        index=vm_traces_agg_df.index.copy(),
    ).plot()
).opts(opts_global).opts(**opts_overlay)

# %% [markdown]
# ### Build training/test set

# %%
input_len = 5
# input_len = 3

x_train = np.array(
    [
        vm_traces_agg_train[i - input_len : i]
        for i in range(input_len, vm_traces_agg_train.shape[0])
    ]
)

print("# Training")
print(f"inputs shape: {x_train.shape}")

# %% [markdown]
# ### Negative Sample RF

# %%
# NOTE: training set must be reshaped such that it is 2D for NS-RF
x_train_nsrf = pd.DataFrame(
    x_train.reshape(-1, x_train.shape[1] * x_train.shape[2]),
    columns=[f"c{i}" for i in range(x_train.shape[1] * x_train.shape[2])],
)
# x_train_nsrf

# %%
nsrf_params = {}
nsrf_params["sample_ratio"] = 10.0
nsrf_params["sample_delta"] = 0.05
nsrf_params["num_estimators"] = 150
nsrf_params["criterion"] = "gini"
nsrf_params["max_depth"] = 50
nsrf_params["min_samples_split"] = 12
nsrf_params["min_samples_leaf"] = 5
nsrf_params["min_weight_fraction_leaf"] = 0.06
nsrf_params["max_features"] = 0.3
nsrf_params["sample_ratio"] = 2.0
nsrf_params["sample_delta"] = 0.05

nsrf = NegativeSamplingRandomForestAd(
    n_estimators=nsrf_params["num_estimators"],
    criterion=nsrf_params["criterion"],
    max_depth=nsrf_params["max_depth"],
    min_samples_split=nsrf_params["min_samples_split"],
    min_samples_leaf=nsrf_params["min_samples_leaf"],
    min_weight_fraction_leaf=nsrf_params["min_weight_fraction_leaf"],
    max_features=nsrf_params["max_features"],
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    # min_impurity_split=None, # NOTE: this param does not exist in scikit-learn 1.0.2
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    sample_delta=nsrf_params["sample_delta"],
    sample_ratio=nsrf_params["sample_ratio"],
)
nsrf.train_model(x_train=x_train_nsrf)

# %%
# dump trained model to disk
current_date = datetime.today().strftime("%Y-%m-%d")
dump_filename = f"ad_nsrf_{current_date}.joblib"
print(dump_filename)
joblib.dump(nsrf, dump_filename)


# %% [markdown] tags=[]
# ## Validation & Testing
#
# - positive samples -> class_label = 1
# - negative (anomalous) samples -> class_label = 0
#
# **NOTE:** it could be useful to hold out a portion of the test set to be used as validation set. In this way, we could use val to plot ROC-AUC and tune the clf threshold, and test to evaluate precision.

# %%
def is_positive(input_sample, std_thresh=1.5, obs_perc=0.66):
    obs_num = input_sample.shape[0]

    # an obs is positive iff std is below tresh
    is_positive_arr = (input_sample[:, 1] <= std_thresh).astype(int)

    # sample is positive iff at least {obs_perc}% obs are positive
    return int(is_positive_arr.sum() >= np.floor(obs_num * obs_perc))


# %%
# load labelled anomalous data from stress/fault runs
stress_runs_lim = (7, 26)
stress_runs_exclude = set([6])
stress_test = set([13, 15, 20, 23, 26])
fault_runs_lim = (1, 17)
fault_runs_exclude = set([15])
fault_test = set([3, 4, 8, 13, 16])

x_val = None
y_val = None
x_test = None
y_test = None

run_trace_file_list = sorted(DATA_ROOT.glob("distwalk-fault-[0-9][0-9].csv")) + sorted(
    DATA_ROOT.glob("distwalk-stress-[0-9][0-9].csv")
)
for run_trace_file in run_trace_file_list:
    run_trace_file = run_trace_file.resolve()
    i = int(run_trace_file.stem.split("-")[-1])

    if "-stress-" in run_trace_file.name:
        if i in stress_runs_exclude:
            continue
        if stress_runs_lim[0] is not None and i < stress_runs_lim[0]:
            continue
        if stress_runs_lim[1] is not None and i > stress_runs_lim[1]:
            continue
        if stress_runs_lim[0] is None and stress_runs_lim[1] is None:
            continue
        label = "stress"
    elif "-fault-" in run_trace_file.name:
        if i in fault_runs_exclude:
            continue
        if fault_runs_lim[0] is not None and i < fault_runs_lim[0]:
            continue
        if fault_runs_lim[1] is not None and i > fault_runs_lim[1]:
            continue
        if fault_runs_lim[0] is None and fault_runs_lim[1] is None:
            continue
        label = "fault"

    # print(f"reading from '{run_trace_file}'...")

    df = pd.read_csv(run_trace_file, index_col=["timestamp"])
    df = df.loc[~df["distwalk"].isna() & ~df[label].isna(), ["mean", "std"]]

    x_chunk = np.array([df[i - input_len : i] for i in range(input_len, df.shape[0])])
    y_chunk = np.array([is_positive(xi) for xi in x_chunk])

    if i in stress_test or i in fault_test:
        if x_test is None:
            x_test = x_chunk
            y_test = y_chunk
        else:
            x_test = np.vstack((x_test, x_chunk))
            y_test = np.hstack((y_test, y_chunk))
    else:
        if x_val is None:
            x_val = x_chunk
            y_val = y_chunk
        else:
            x_val = np.vstack((x_val, x_chunk))
            y_val = np.hstack((y_val, y_chunk))

print("# Validation")
print(f"inputs shape: {x_val.shape}")
print(f"labels shape: {y_val.shape}")
print(f"positive samples ratio: {y_val.sum() / len(y_val)}")

print("# Test")
print(f"inputs shape: {x_test.shape}")
print(f"labels shape: {y_test.shape}")
print(f"positive samples ratio: {y_test.sum() / len(y_test)}")

# %%
# scale and reshape data for NSRF
x_val_scaled = np.array([scaler.transform(e) for e in x_val])
x_val_nsrf = pd.DataFrame(
    x_val_scaled.reshape(-1, x_val_scaled.shape[1] * x_val_scaled.shape[2]),
    columns=[f"c{i}" for i in range(x_val_scaled.shape[1] * x_val_scaled.shape[2])],
)

x_test_scaled = np.array([scaler.transform(e) for e in x_test])
x_test_nsrf = pd.DataFrame(
    x_test_scaled.reshape(-1, x_test_scaled.shape[1] * x_test_scaled.shape[2]),
    columns=[f"c{i}" for i in range(x_test_scaled.shape[1] * x_test_scaled.shape[2])],
)

# %%
y_val_pred = nsrf.predict(x_val_nsrf)["class_prob"]

# %%
plt.rcParams["font.size"] = 16
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
roc_plot = RocCurveDisplay.from_predictions(
    y_val, y_val_pred, pos_label=1, ax=ax, marker="o", name="MADI - NSRF"
)
ax.set_ylabel("True Positive Rate")
ax.set_xlabel("False Positive Rate")
ax.plot([[0, 0], [1, 1]], linestyle="dashed", color="red")
fig.savefig("ad_roc-auc.eps", bbox_inches="tight")

# %%
# tune clf threshold on validation results
fpr, tpr, thresh = roc_curve(y_val, y_val_pred, pos_label=1)
roc_curve_df = pd.DataFrame(
    np.vstack((fpr, tpr, thresh)).T, columns=["fpr", "tpr", "thresh"]
)

roc_curve_df.loc[
    (roc_curve_df["fpr"] < 0.1)  # keep false positive rate below 10%
    & (roc_curve_df["tpr"] > 0.8)
].tail(
    1
)  # take last entry (max fpr & max tpr)

# %%
# use tuned clf threshold on test set
y_test_pred = nsrf.predict(x_test_nsrf)["class_prob"]
y_test_pred_bin = binarize(y_test_pred.values.reshape(-1, 1), threshold=0.0665).reshape(
    -1
)

test_acc = (y_test == y_test_pred_bin).astype(int).sum() / y_test.shape[0]
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# %%
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred_bin, normalize="true", cmap="Blues", ax=ax
)
fig.savefig("ad_conf-mat.eps", bbox_inches="tight")

# %%
