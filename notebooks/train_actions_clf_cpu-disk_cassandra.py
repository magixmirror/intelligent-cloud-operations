# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -tags,-jupyter,-jp-MarkdownHeadingCollapsed
#     comment_magics: true
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
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import holoviews as hv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from constants import DEFAULT_DATA_ROOT, DEFAULT_IMG_DEST
from intops_utils import load_cassandra_data, make_clf_input_pairs, make_data_chunks
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder

hv.extension("bokeh")
pd.options.plotting.backend = "holoviews"
plt.rcParams["font.size"] = 20

# %%
default_data_root = Path(DEFAULT_DATA_ROOT).resolve()
default_img_dest = Path(DEFAULT_IMG_DEST).resolve()

# randomness
seed = None
rng = np.random.default_rng(seed)


# %% [markdown]
# # Corrective Action Classification

# %% [markdown]
# ## Training & Test data


# %%
input_len = 5
x_train = None
y_train = None
x_test = None
y_test = None

train_data_list = sorted(
    list(default_data_root.glob("FINAL-rl3-cl1-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl1-theoden-kill-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl1-theoden-kill-try2-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl1-theoden-kill-try4-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl1-theoden-nox-hog-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl1-theoden-nox-try2-hog-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl1-theoden-nox-try4-hog-ops3000000-l1000-t1000"))
    #
    + list(default_data_root.glob("FINAL-rl3-cl2-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl2-theoden-kill-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl2-theoden-kill-try2-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl2-theoden-kill-try4-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl2-theoden-nox-hog-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl2-theoden-nox-try2-hog-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl2-theoden-nox-try4-hog-ops3000000-l1000-t1000"))
)
test_data_list = sorted(
    list(default_data_root.glob("FINAL-rl3-cl1-try2-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl1-theoden-kill-try3-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl1-theoden-nox-try3-hog-ops3000000-l1000-t1000"))
    #
    + list(default_data_root.glob("FINAL-rl3-cl2-theoden-kill-try3-ops3000000-l1000-t1000"))
    + list(default_data_root.glob("FINAL-rl3-cl2-theoden-nox-try3-hog-ops3000000-l1000-t1000"))
)

train_len = len(train_data_list)
run_data_list = train_data_list + test_data_list
for data_idx, run_data in enumerate(run_data_list):
    load_df, times_df, injection_df, anomaly_type = load_cassandra_data(run_data)

    # Input features
    load_chunks = make_data_chunks(load_df, input_len)

    # Input labels
    times_chunks = make_data_chunks(times_df, input_len)
    injection_chunks = make_data_chunks(injection_df, input_len)

    for data, time_data, injection_data in zip_longest(
        load_chunks, times_chunks, injection_chunks
    ):
        samples, labels = make_clf_input_pairs(
            anomaly_type, data, time_data, injection_data
        )

        if data_idx < train_len:
            if x_train is None:
                x_train = samples
                y_train = labels
            else:
                x_train = np.vstack((x_train, samples))
                y_train = np.hstack((y_train, labels))
        else:
            if x_test is None:
                x_test = samples
                y_test = labels
            else:
                x_test = np.vstack((x_test, samples))
                y_test = np.hstack((y_test, labels))

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(
    x_train.shape
)
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

# %%
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

# %% [markdown]
# ## XGBoost
#
# The training set has to be reshaped such that it is 2D. An original input sample has shape $1 \times 5 \times 6$, and consists in a 5-mins VM measurements window with the following features (in this order):
#
# 1. VM CPU utilization
# 2. VM I/O write ops/sec
# 3. Average CPU utilization of the other VMs
# 4. Average I/O write ops/sec of the other VMs
# 5. Std deviation of the CPU utilization of the other VMs
# 6. Std deviation of the I/O write ops/sec of the other VMs
#
# A transformed one will have shape $1 \times 30$, such that the original rows are stacked horizontally.

# %%
# NOTE: since data classes are highly imbalanced, we use a weight vector
class_weights = dict(zip(train_idx_arr, train_count_arr.min() / train_count_arr))
weight_arr = np.array([class_weights[int(yi)] for yi in y_train])

# NOTE: training set must be reshaped such that it is 2D for XGBoost.
dtrain = xgb.DMatrix(
    x_train.reshape(-1, x_train.shape[1] * x_train.shape[2]),
    label=y_train,
    weight=weight_arr,
)

# %%
# see https://xgboost.readthedocs.io/en/stable/parameter.html
param = {
    "max_depth": 10,
    "eta": 1,
    # "objective": "multi:softmax",
    "objective": "multi:softprob",
    "num_class": 4,
    "verbosity": 1,
}

train_start = datetime.now()
bst = xgb.train(param, dtrain, num_boost_round=10)
train_end = datetime.now()

# %%
print(f"training time: {train_end - train_start}")

# %%
# dump trained model and scaler to disk
current_date = datetime.today().strftime("%Y-%m-%d")

model_dump_filename = f"cassandra_clf_xgboost_{current_date}.joblib"
bst.save_model(model_dump_filename)
print(model_dump_filename)

scaler_dump_filename = f"cassandra_clf_scaler_{current_date}.joblib"
joblib.dump(scaler, scaler_dump_filename)
print(scaler_dump_filename)

# %%
# bst.trees_to_dataframe()

# %%
# xgb.plot_tree(bst)
# fig = plt.gcf()
# fig.set_size_inches(200, 100)
# fig.savefig("reboot_scenario_xgboost.pdf")

# %% [markdown]
# ## Testing

# %%
# load trained model and scaler from disk
# NOTE: decomment only when reproducing previous results

# model trained on Cassandra runs
bst = xgb.Booster(model_file="cassandra_clf_xgboost_2023-03-09.joblib")
scaler = joblib.load("cassandra_clf_scaler_2023-03-09.joblib")

# model trained on distwalk runs
# bst = xgb.Booster(model_file="clf_xgboost_2023-02-23.joblib")
# scaler = joblib.load("clf_scaler_2023-02-23.joblib")

# %%
dtest = xgb.DMatrix(x_test.reshape(-1, x_test.shape[1] * x_test.shape[2]), label=y_test)
y_pred_prob = bst.predict(dtest)
y_pred = y_pred_prob.argmax(axis=1)

# %%
# plot ROC curves 'one-vs-rest'
roc_labels = [
    "0",  # normal
    "1",  # stress
    "2",  # fault
    "3"   # saturation
]

onehot_enc = OneHotEncoder(sparse=False).fit(y_test.reshape(-1, 1))
print(f"categories: {onehot_enc.categories_}")

y_test_onehot = onehot_enc.transform(y_test.reshape(-1, 1))

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)

for idx in range(y_test_onehot.shape[1]):
    RocCurveDisplay.from_predictions(
        y_test_onehot[:, idx],
        y_pred_prob[:, idx],
        pos_label=1,
        ax=ax,
        # marker="o",
        name=roc_labels[idx],
    )

# RocCurveDisplay.from_predictions(
#     y_test_onehot.ravel(),
#     y_pred_prob.ravel(),
#     pos_label=1,
#     ax=ax,
#     name="micro-average",
# )

ax.plot([[0, 0], [1, 1]], linestyle="dashed", color="black")
ax.set_ylabel("True Positive Rate")
ax.set_xlabel("False Positive Rate")
ax.grid()
fig.savefig(default_img_dest / "cassandra_clf_roc-auc.eps", bbox_inches="tight")

# %%
test_acc = (y_test == y_pred).astype(int).sum() / y_test.shape[0]
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# %%
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, normalize="true", cmap="Blues", ax=ax, colorbar=False
)
fig.savefig(default_img_dest / "cassandra_clf_conf-mat.eps", bbox_inches="tight")

# %%
# compute performance metrics 'one-vs-rest'
# labels = [0, 1, 2, 3]
labels = [0, 1, 2]

# per-class accuracy
mcm = multilabel_confusion_matrix(y_test, y_pred, labels=labels)
acc = (mcm * np.stack([np.eye(2)] * len(labels))).sum(axis=1).sum(axis=1) / mcm[0].sum()

# per-class precision/recall/f1 score
prec, rec, fscore, _ = precision_recall_fscore_support(
    y_test, y_pred, labels=labels, pos_label=1
)

perf_df = pd.DataFrame(
    np.vstack((acc, prec, rec, fscore)).T,
    index=labels,
    columns=["Accuracy", "Precision", "Recall", "F1 score"],
)

# print(perf_df.round(3).to_markdown())
col_fmt = "r" + "r" * perf_df.columns.size
print(
    perf_df.style.format(precision=3)
    .format_index(escape="latex", axis=1)
    .format_index(escape="latex", axis=0)
    .hide(axis=0, names=True)
    .to_latex(column_format=col_fmt, hrules=True, sparse_index=False)
)

perf_df

# %%
