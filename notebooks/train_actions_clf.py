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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from constants import DATA_ROOT
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve
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

# randomness
seed = None
rng = np.random.default_rng(seed)


# %% [markdown] tags=[]
# ## Corrective Action Classification
#
# **Other design choices to be made:**
# - In case multiple candidates are from the same elastic group, should we process them together? -> May be hard, given that we don't know the number of VMs apriori
#
# **Other ideas to be implemented:**
#
# 1. compute statistics escluding the VM the current input sample refer to
# 2. use more robust statistics like median (or inter-quartile range-like, as a robust alternative to std)

# %% [markdown]
# ### Training & Test data

# %%
def is_faulty(input_sample, std_col_idx=-1, std_thresh=1.5, obs_perc=0.66):
    obs_num = input_sample.shape[0]

    # print(f"D: input_sample:\n{input_sample}")

    # an obs is faulty iff std is above tresh
    is_faulty_arr = (input_sample[:, std_col_idx] > std_thresh).astype(int)

    # sample is faulty iff at least {obs_perc}% obs are faulty
    return is_faulty_arr.sum() >= np.floor(obs_num * obs_perc)


def make_input_pairs(data, anomaly_type, agg_cols_num=2):
    # print(f"D: data.shape: {data.shape}")
    vms_num = data.shape[1] - agg_cols_num

    # last 2 columns are to be replicated for all samples
    samples = np.array(
        [
            np.hstack((data[:, i].reshape(-1, 1), data[:, -agg_cols_num:]))
            for i in range(vms_num)
        ]
    )
    # print(f"D: samples.shape: {samples.shape}")

    # NOTE: inferring (possible) faulty VM from low overall cpu utilization
    culprit_idx = data[:, :-2].sum(axis=0).argmin()
    # print(f"D: culprit_idx: {culprit_idx}")

    labels = np.zeros(vms_num)
    if is_faulty(samples[culprit_idx]):
        labels[culprit_idx] = anomaly_type

    # print(f"D: labels.shape: {labels.shape}")
    # print(f"D: labels: {labels}")

    return samples, labels.astype(int)


# %%
stress_runs_lim = (7, 26)
stress_runs_exclude = set([6])
stress_test = set([13, 15, 20, 23, 26])
fault_runs_lim = (1, 17)
fault_runs_exclude = set([15])
fault_test = set([3, 4, 8, 13, 16])

data_classes = {
    "stress": 1,
    "fault": 2,
}

input_len = 5
x_train = None
y_train = None
x_test = None
y_test = None

# load labelled anomalous data from stress/fault runs
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
    df = df.loc[~df["distwalk"].isna() & ~df[label].isna()].filter(
        regex="node-.*|mean|std"
    )

    # TODO: add feature scaling
    x_chunk = np.array([df[i - input_len : i] for i in range(input_len, df.shape[0])])

    # print(f"D: x_chunk.shape: {x_chunk.shape}")

    for data in x_chunk:
        samples, labels = make_input_pairs(data, data_classes[label])
        if i in stress_test or i in fault_test:
            if x_test is None:
                x_test = samples
                y_test = labels
            else:
                x_test = np.vstack((x_test, samples))
                y_test = np.hstack((y_test, labels))
        else:
            if x_train is None:
                x_train = samples
                y_train = labels
            else:
                x_train = np.vstack((x_train, samples))
                y_train = np.hstack((y_train, labels))

print("# Training")
print(f"inputs shape: {x_train.shape}")
print(f"labels shape: {y_train.shape}")
print("classes distribution:")
train_idx_arr, train_count_arr = np.unique(y_train, return_counts=True)
for idx, count in zip(train_idx_arr.astype(int), train_count_arr):
    print(f"  {idx} -> {count} ({count / len(y_train) * 100:.2f}%)")

print("# Test")
print(f"inputs shape: {x_test.shape}")
print(f"labels shape: {y_test.shape}")
print("classes distribution:")
test_idx_arr, test_count_arr = np.unique(y_test, return_counts=True)
for idx, count in zip(test_idx_arr.astype(int), test_count_arr):
    print(f"  {idx} -> {count} ({count / len(y_test) * 100:.2f}%)")

# %% [markdown] tags=[]
# ### XGBoost

# %% tags=[]
# NOTE: training set must be reshaped such that it is 2D for XGBoost.
# Also, provided that the data are highly unmbalanced, we use a weight vector.
class_weights = train_count_arr.min() / train_count_arr
weight_arr = np.array([class_weights[int(yi)] for yi in y_train])
dtrain = xgb.DMatrix(
    x_train.reshape(-1, x_train.shape[1] * x_train.shape[2]),
    label=y_train,
    weight=weight_arr,
)

# %% tags=[]
# see https://xgboost.readthedocs.io/en/stable/parameter.html
param = {
    "max_depth": 10,
    "eta": 1,
    "objective": "multi:softmax",
    # "objective": "multi:softprob",
    "num_class": 3,
    "verbosity": 1,
}
bst = xgb.train(param, dtrain, num_boost_round=10)

# %%
# dump trained model to disk
current_date = datetime.today().strftime("%Y-%m-%d")
dump_filename = f"clf_xgboost_{current_date}.joblib"
print(dump_filename)
bst.save_model(dump_filename)

# %%
# bst.trees_to_dataframe()
# xgb.plot_tree(bst)
# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(200, 100)
# fig.savefig("reboot_scenario_xgboost.pdf")

# %% [markdown] tags=[]
# ### Testing

# %%
dtest = xgb.DMatrix(x_test.reshape(-1, x_test.shape[1] * x_test.shape[2]), label=y_test)
y_pred = bst.predict(dtest)

test_acc = (y_test == y_pred).astype(int).sum() / y_test.shape[0]
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# %%
plt.rcParams["font.size"] = 16
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, normalize="true", cmap="Blues", ax=ax
)
fig.savefig("clf_conf-mat.eps", bbox_inches="tight")

# %%
# plot ROC curves 'one-vs-rest'
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
RocCurveDisplay.from_predictions(
    y_test, y_pred, pos_label=1, ax=ax, marker="o", name="1 - Stress"
)
RocCurveDisplay.from_predictions(
    y_test, y_pred, pos_label=2, ax=ax, marker="o", name="2 - Fault"
)
ax.plot([[0, 0], [1, 1]], linestyle="dashed", color="red")
ax.set_ylabel("True Positive Rate")
ax.set_xlabel("False Positive Rate")
fig.savefig("clf_roc-auc.eps", bbox_inches="tight")

# %% [markdown]
# ### Interpretation

# %%
fig, ax = plt.subplots(2, 2)
xgb.plot_importance(
    bst, importance_type="weight", title="Feature importance - weight", ax=ax[0, 0]
)
xgb.plot_importance(
    bst, importance_type="gain", title="Feature importance - gain", ax=ax[0, 1]
)
xgb.plot_importance(
    bst, importance_type="cover", title="Feature importance - cover", ax=ax[1, 0]
)
fig.set_size_inches(30, 10)

# %%
