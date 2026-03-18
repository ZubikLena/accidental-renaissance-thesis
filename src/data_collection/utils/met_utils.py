import pandas as pd
import requests


def apply_filters(df, config):
    f = config["filters"]

    df["Classification"] = df["Classification"].fillna("").str.lower()
    df["Medium"] = df["Medium"].fillna("").str.lower()
    df["Department"] = df["Department"].fillna("").str.lower()

    df = df[
        df["Classification"].str.contains(f["classification"], na=False) &
        df["Medium"].str.contains("|".join(f["medium_include"]), na=False) &
        ~df["Medium"].str.contains("|".join(f["medium_exclude"]), na=False) &
        df["Department"].str.contains(f["department"], na=False) &
        df["Object Begin Date"].between(f["date_range"][0], f["date_range"][1])
    ]

    return df


def get_met_items(config):
    df = pd.read_csv(config["csv_path"])
    df = apply_filters(df, config)
    return df.to_dict("records")


def extract_met_image(item, config):
    object_id = item["Object ID"]
    url = config["api_url"].format(object_id=object_id)

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data.get(config["image_field"])
    except:
        return None