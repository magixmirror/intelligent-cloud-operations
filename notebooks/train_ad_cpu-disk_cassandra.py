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
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import holoviews as hv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants import DEFAULT_DATA_ROOT, DEFAULT_IMG_DEST
from intops_utils import (
    is_anomaly,
    load_cassandra_data,
    make_data_chunks,
    make_spatial_aggregates,
)
from madi.detectors.neg_sample_random_forest import NegativeSamplingRandomForestAd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, binarize

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
# # Anomaly Detection

# %% [markdown]
# ## Build training and test data

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
run_data_list = (
    train_data_list
    + test_data_list
)
for data_idx, run_data in enumerate(run_data_list):
    load_df, times_df, injection_df, anomaly_type = load_cassandra_data(run_data)
    agg_load_df = make_spatial_aggregates(load_df)

    # Input features
    load_chunks = make_data_chunks(agg_load_df, input_len)

    # Input labels
    times_chunks = make_data_chunks(times_df, input_len)
    injection_chunks = make_data_chunks(injection_df, input_len)

    for data, time_data, injection_data in zip_longest(
        load_chunks, times_chunks, injection_chunks
    ):
        # negative samples -> class_label = 0 (anomalies)
        # positive samples -> class_label = 1
        label = 0 if is_anomaly(time_data, injection_data) else 1

        if data_idx < train_len:
            if x_train is None:
                x_train = np.array([data])
                y_train = label
            else:
                x_train = np.vstack((x_train, np.array([data])))
                y_train = np.hstack((y_train, label))
        else:
            if x_test is None:
                x_test = np.array([data])
                y_test = label
            else:
                x_test = np.vstack((x_test, np.array([data])))
                y_test = np.hstack((y_test, label))

# %%
# load_df.plot() + times_df.plot() + injection_df.plot()

# %%
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
print(f"positive samples: {y_train.sum()} ({y_train.sum() / len(y_train) * 100:.2f}%)")

print("# Test")
print(f"inputs shape: {x_test.shape}")
print(f"labels shape: {y_test.shape}")
print(f"positive samples: {y_test.sum()} ({y_test.sum() / len(y_test) * 100:.2f}%)")

# %% [markdown]
# ## Negative Sample RF

# %%
# NOTE: NSRF should observe positive examples only
mask = y_train.reshape(-1, 1).any(axis=1)
x_train_pos = x_train[mask]
x_train_pos.shape

# %%
# NOTE: training set must be reshaped such that it is 2D for NSRF
x_train_nsrf = pd.DataFrame(
    x_train_pos.reshape(-1, x_train_pos.shape[1] * x_train_pos.shape[2]),
    columns=[f"c{i}" for i in range(x_train_pos.shape[1] * x_train_pos.shape[2])],
)

# %%
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

train_start = datetime.now()
nsrf.train_model(x_train=x_train_nsrf)
train_end = datetime.now()

# %%
print(f"training time: {train_end - train_start}")

# %%
# dump trained model and scaler to disk
current_date = datetime.today().strftime("%Y-%m-%d")

model_dump_filename = f"cassandra_ad_nsrf_{current_date}.joblib"
joblib.dump(nsrf, model_dump_filename)
print(model_dump_filename)

scaler_dump_filename = f"cassandra_ad_scaler_{current_date}.joblib"
joblib.dump(scaler, scaler_dump_filename)
print(scaler_dump_filename)

# %% [markdown]
# ## Testing
#
# - negative samples -> class_label = 0 (anomalies)
# - positive samples -> class_label = 1

# %%
# load trained model and scaler from disk
# NOTE: decomment only when reproducing previous results

# model trained on Cassandra runs
nsrf = joblib.load("cassandra_ad_nsrf_2023-03-09.joblib")
scaler = joblib.load("cassandra_ad_scaler_2023-03-09.joblib")

# model trained on distwalk runs
# nsrf = joblib.load("ad_nsrf_2023-02-23.joblib")
# scaler = joblib.load("ad_scaler_2023-02-23.joblib")

# %%
# reshape data for NSRF
x_test_nsrf = pd.DataFrame(
    x_test.reshape(-1, x_test.shape[1] * x_test.shape[2]),
    columns=[f"c{i}" for i in range(x_test.shape[1] * x_test.shape[2])],
)

# %%
y_test_pred = nsrf.predict(x_test_nsrf)["class_prob"]

# %%
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)

roc_plot = RocCurveDisplay.from_predictions(
    # y_val,
    # y_val_pred,
    y_test,
    y_test_pred,
    pos_label=1,
    ax=ax,
    # marker="o",
    name="NSRF",
)

ax.set_ylabel("True Positive Rate")
ax.set_xlabel("False Positive Rate")
ax.grid()
ax.plot([[0, 0], [1, 1]], linestyle="dashed", color="black")
fig.savefig(default_img_dest / "cassandra_ad_roc-auc.eps", bbox_inches="tight")

# %%
# tune clf threshold on validation results
fpr, tpr, thresh = roc_curve(
    y_test,
    y_test_pred,
    pos_label=1,
)
roc_curve_df = pd.DataFrame(
    np.vstack((fpr, tpr, thresh)).T, columns=["fpr", "tpr", "thresh"]
)

thresh_df = roc_curve_df.loc[
    (roc_curve_df["fpr"] <= 0.2) & (roc_curve_df["tpr"] >= 0.8)
    # (roc_curve_df["fpr"] <= 0.3) & (roc_curve_df["tpr"] >= 0.8)  # distwalk
].tail(
    1
)  # take last entry (max fpr & max tpr)

thresh_df

# %%
# use tuned clf threshold on test set
y_test_pred_bin = binarize(
    y_test_pred.values.reshape(-1, 1), threshold=thresh_df["thresh"].values[0]
).reshape(-1)

test_acc = (y_test == y_test_pred_bin).astype(int).sum() / y_test.shape[0]
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# %%
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred_bin, normalize="true", cmap="Blues", ax=ax, colorbar=False
)
fig.savefig(default_img_dest / "cassandra_ad_conf-mat.eps", bbox_inches="tight")

# %%
# compute performance metrics
# NOTE: since it is binary classification, per-class accuracy is already
# included in the confusion matrix.

# per-class precision/recall/f1 score
prec, rec, fscore, _ = precision_recall_fscore_support(
    y_test, y_test_pred_bin, pos_label=1
)

perf_df = pd.DataFrame(
    np.vstack((prec, rec, fscore)).T,
    columns=["Precision", "Recall", "F1 score"],
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
