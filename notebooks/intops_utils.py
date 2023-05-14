import json
import numpy as np
import pandas as pd
from constants import NOTEBOOKS_CONFIG_FILE
from monasca_utils import json_to_df
from pathlib import Path


def load_run_data(run_data):
    # Read run metadata (NOTE: assuming only one matching file in dir)
    metadata_file = list(run_data.glob("*-metadata.json"))[0]
    with open(metadata_file, "r+") as f:
        metadata = json.load(f)

    basename = metadata_file.name.split("-metadata.json")[0]

    # Read run load data
    load_file = run_data / f"{basename}-load.csv"
    print(f"reading from '{load_file}'...")
    load_df = pd.read_csv(load_file, index_col=["timestamp"])

    # Read run times data (NOTE: assuming 1min granularity)
    times_file = run_data / f"{basename}-times-1min.csv"
    print(f"reading from '{times_file}'...")
    times_df = pd.read_csv(
        times_file, usecols=["timestamp", "p90"], index_col=["timestamp"]
    )

    # Read anomaly-injection schedule
    anomaly_type = metadata["anomaly_type"]
    if anomaly_type is None:
        anomaly_type = "saturation"
        injection_df = None
    else:
        # NOTE: anomaly-injection schedule is generally longer than the actual run duration,
        # so we truncate according to the longest possible run time-span.
        union_idx = load_df.index.union(times_df.index)

        anomaly_sched_file = run_data / f"{basename}-{anomaly_type}-sched.dat"
        print(f"reading from '{anomaly_sched_file}'...")
        injection_df = pd.read_csv(anomaly_sched_file, header=None, names=["injection"])

        injection_df = injection_df[: union_idx.shape[0]]
        injection_df.index = union_idx

    # Index intersection (NOTE: assuming NaN only at the beginning or end of the run)
    common_idx = load_df.dropna(axis=0, how="any").index.intersection(
        times_df.dropna(axis=0, how="any").index
    )

    return (
        load_df.loc[common_idx],
        times_df.loc[common_idx],
        injection_df.loc[common_idx] if injection_df is not None else None,
        anomaly_type,
    )


def load_cassandra_data(run_data):
    results_config_file = Path(NOTEBOOKS_CONFIG_FILE).resolve()
    if results_config_file.exists():
        with open(results_config_file, "r") as f:
            results_config_dict = json.load(f)
    else:
        raise ValueError(f"config file '{results_config_file}' not found")

    # Read run load data
    load_df = None
    metric_names = results_config_dict.get("metric_names", list())
    metric_names_norm = [x.replace(".", "-") for x in metric_names]
    for metric, metric_norm in zip(metric_names, metric_names_norm):
        metric_file = (run_data / f"{metric_norm}.json").resolve()
        print(f"reading from '{metric_file}'...")

        metric_df = json_to_df(metric_file)
        table = pd.pivot_table(
            metric_df,
            values=metric,
            index=["timestamp"],
            columns=["hostname"],
        )
        table = table.resample("1min").mean()

        if metric == "cpu.utilization_perc":
            # NOTE: assuming that Cassandra instances were provided 2 vCPUs
            table = table / 2

        # reorder columns by VMs start time
        table = table[
            table.apply(pd.Series.first_valid_index).sort_values().index.to_list()
        ]

        table.rename(lambda x: f"{x}.{metric}", axis=1, inplace=True)
        if load_df is None:
            load_df = table
        else:
            load_df = load_df.join(table, how="outer")

    # Read run times data
    times_file = (run_data / "run.csv").resolve()
    print(f"reading from '{times_file}'...")
    times_df = (
        pd.read_csv(
            times_file,
            usecols=[
                "Abs Time (datetime)",
                "P90 Latency (us)",
            ],
            index_col=["Abs Time (datetime)"],
            parse_dates=True,
        )
        .resample("1min")
        .mean()
    )
    times_df.index.name = "timestamp"
    times_df.index = times_df.index.tz_convert(None)  # convert to UTC and drop TZ info
    times_df.columns = ["p90"]
    times_df["p90"] = times_df["p90"] / 1000  # convert to ms

    # Index intersection (NOTE: assuming NaN only at the beginning or end of the run)
    common_idx = load_df.dropna(axis=0, how="any").index.intersection(
        times_df.dropna(axis=0, how="any").index
    )

    # Read anomaly-injection schedule (NOTE: assuming only one matching file in dir)
    injection_df = None
    anomaly_type = "saturation"
    anomaly_sched_file = list(run_data.glob("run_*_sched.csv"))
    if anomaly_sched_file:
        print(f"reading from '{anomaly_sched_file[0]}'...")
        injection_df = (
            pd.read_csv(
                anomaly_sched_file[0],
                index_col=["Abs Time (datetime)"],
                parse_dates=True,
            )
            .resample("1min")
            .last()
        )
        injection_df.index.name = "timestamp"
        injection_df.columns = ["injection"]

        anomaly_type = anomaly_sched_file[0].name.split("_")[1]
        if anomaly_type == "kill":
            anomaly_type = "fault"

        common_idx = common_idx.intersection(
            injection_df.dropna(axis=0, how="any").index
        )

    return (
        load_df.loc[common_idx],
        times_df.loc[common_idx],
        injection_df.loc[common_idx] if injection_df is not None else None,
        anomaly_type,
    )


def make_data_chunks(data, input_len):
    data_chunks = []
    if data is not None:
        data_chunks = np.array(
            [data[i - input_len : i] for i in range(input_len, data.shape[0])]
        )
        # print(f"D: data_chunks.shape: {data_chunks.shape}")
    return data_chunks


def make_spatial_aggregates(load_df):
    # TODO: handle traces containing missing obs (e.g., VMs started later wrt
    # others) extract aggregated features
    grouped = load_df.groupby(
        load_df.columns.str.extract("\w+\.(\w+\.\w+)", expand=False), axis=1
    )
    grouped_mean = grouped.mean(numeric_only=True).rename(lambda x: f"mean.{x}", axis=1)
    grouped_std = grouped.std(numeric_only=True).rename(lambda x: f"std.{x}", axis=1)

    return pd.concat([grouped_mean, grouped_std], axis=1)


def make_clf_input_pairs(anomaly_type, data, time_data, injection_data):
    # NOTE: making the following assumptions on input data:
    # - input containing data for k unique VMs
    # - input containing data for 2 unique metrics: CPU and disk usage
    # - colums related to the same metric being adjacent
    # - each column group ordered in the same way (VMs-wise)
    classes = {"stress": 1, "fault": 2, "saturation": 3}

    # print(f"D: data.shape: {data.shape}")

    vms_num = data.shape[1] // 2
    vms_idx_set = set(range(vms_num))

    # NOTE: make input samples containing specific VM data and spatial aggregates
    samples = []
    for i in range(vms_num):
        cpu_data = data[:, i].reshape(-1, 1)
        disk_data = data[:, i + vms_num].reshape(-1, 1)

        # NOTE: computing spatial aggregates leaving one out each time
        other_vms = np.array(list(vms_idx_set - {i}))

        # print(f"D: other_vms: {other_vms}")

        cpu_data_avg = np.mean(data[:, other_vms], axis=1).reshape(-1, 1)
        disk_data_avg = np.mean(data[:, other_vms + vms_num], axis=1).reshape(-1, 1)

        # print(f"D: cpu_data_agg.shape: {cpu_data_avg.shape}")
        # print(f"D: disk_data_agg.shape: {disk_data_avg.shape}")

        cpu_data_std = np.std(data[:, other_vms], axis=1).reshape(-1, 1)
        disk_data_std = np.std(data[:, other_vms + vms_num], axis=1).reshape(-1, 1)

        # print(f"D: cpu_data_std.shape: {cpu_data_std.shape}")
        # print(f"D: disk_data_std.shape: {disk_data_std.shape}")

        samples.append(
            np.hstack(
                (
                    cpu_data,
                    disk_data,
                    cpu_data_avg,
                    disk_data_avg,
                    cpu_data_std,
                    disk_data_std,
                )
            )
        )

    samples = np.array(samples)
    # print(f"D: samples.shape: {samples.shape}")

    labels = np.zeros(vms_num)
    if anomaly_type == "saturation":
        if is_saturation(time_data):
            labels = np.array([classes[anomaly_type]] * vms_num)
    else:
        # NOTE: inferring (possible) faulty VM from low total disk utilization.
        # Assuming the related columns to be the last group 2nd and that there
        # is only a single VM that was injected with a fault.
        if is_injected(injection_data):
            # print(f"D: raw data: {data}")
            # print(f"D: sum data: {data[:, -vms_num:].sum(axis=0)}")

            culprit_idx = data[:, -vms_num:].sum(axis=0).argmin()

            # print(f"D: culprit_idx: {culprit_idx}")

            labels[culprit_idx] = classes[anomaly_type]

    # print(f"D: labels.shape: {labels.shape}")
    # print(f"D: labels: {labels}")

    return samples, labels.astype(int)


def is_anomaly(time_data, injection_data):
    return is_saturation(time_data) or is_injected(injection_data)


def is_injected(injection_data, obs_perc=0.66):
    # NOTE: assuming that there is only a single VM that was injected with a fault.
    #
    # An input sample is anomalous iff an anomaly was injected for at least the
    # {obs_perc*100}% of the observations.
    if injection_data is None:
        return False

    obs_num = injection_data.shape[0]
    return injection_data.sum() >= np.floor(obs_num * obs_perc)


def is_saturation(time_data, delay_ms_thresh=35, obs_perc=0.66):
    # An input sample is related to a saturation scenario iff, for the {obs_perc}%
    # of the observations, the p90 clien-side delay is greater than {delay_ms_thresh} ms
    obs_num = time_data.shape[0]

    # print(f"D: time_data:\n{time_data}")

    is_saturation_arr = time_data > delay_ms_thresh

    # print(f"D: is_saturation_arr:\n{is_saturation_arr}")

    return is_saturation_arr.sum() >= np.floor(obs_num * obs_perc)
