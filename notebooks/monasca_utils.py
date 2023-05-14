import json

import pandas as pd
from constants import DATETIME_FORMAT


def json_to_df(json_file):
    with open(json_file, "r+") as f:
        json_body = json.load(f)

    metric = json_body[0]["name"]
    df = pd.DataFrame(columns=["timestamp", "resource_id", "hostname", metric])

    for item in json_body:
        resource_id = item["dimensions"].get("resource_id")  # optional
        hostname = item["dimensions"]["hostname"]
        measurement_list = item["measurements"]

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        pd.Series([m[0], resource_id, hostname, m[1]], index=df.columns)
                        for m in measurement_list
                    ]
                ),
            ]
        )

    df = df.astype(
        {
            "resource_id": "string",
            "hostname": "string",
            metric: "float64",
        }
    )
    df.set_index(["timestamp"], inplace=True)
    df.index = pd.to_datetime(df.index, format=DATETIME_FORMAT)

    return df
