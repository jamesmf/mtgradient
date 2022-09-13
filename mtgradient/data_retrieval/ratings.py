import requests
import datetime
import time
import json
import os
import typing as T

import pandas as pd


def get_card_ratings_df(
    expansion: str,
    start_date="2020-01-01",
    end_date: T.Optional[str] = None,
    format_priority: T.List[str] = [
        "PremierDraft",
        "QuickDraft",
        "TradDraft",
        "CompDraft",
    ],
):
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    col_map = {
        "avg_seen": "alsa",
        "avg_pick": "ata",
        "opening_hand_win_rate": "oh wr",
        "ever_drawn_win_rate": "gd wr",
        "drawn_improvement_win_rate": "iwd",
    }
    card_df = None
    for format_str in format_priority:
        format_url = f"https://www.17lands.com/card_ratings/data?expansion={expansion.upper()}&format={format_str}&start_date={start_date}&end_date={end_date}"
        resp = requests.get(format_url)
        try:
            json_data = resp.json()
            card_df = pd.DataFrame(json_data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"failed fetching data for {format_url}")
            card_df = pd.DataFrame([])
        if len(card_df) == 0:
            print(
                f"failed to find data for {format_str} for {expansion} between {start_date} and {end_date}, trying next format."
            )
            time.sleep(1)  # try to be polite
        else:
            print(f"fetched {card_df.shape[0]} rows for {expansion} {format_str}")
            break
    return card_df.rename(columns=col_map)  # type: ignore


def read_data(
    data_path: str,
    supported_expansions: T.List[str] = [
        "ZNR",
        "KHM",
        "STX",
        "AFR",
        "MID",
        "VOW",
        "NEO",
        "SNC",
        "HBG",
        "DMU",
    ],
    card_ratings_kwargs: T.Dict[str, T.Any] = {},
    refresh=True,
) -> pd.DataFrame:
    """
    Programmatically pull the 17Lands card ratings and stack them. Also
    load the card_id mapping and join it.

    See get_card_ratings_df() for options  that can be passed to card_ratings_kwargs
    """
    if not refresh:
        return pd.read_csv(f"{data_path}/card_ratings_data.csv", index_col="id")
    card_ids_df = pd.read_csv(
        os.path.join(data_path, "card_list.csv"),
        usecols=[
            "id",
            "expansion",
            "name",
            "rarity",
            "color_identity",
            "mana_value",
            "types",
        ],
    )
    card_ids_df = card_ids_df[card_ids_df.expansion.isin(supported_expansions)]
    card_ids_df = card_ids_df[["id", "expansion", "name", "types"]]

    ratings_df = None
    for set_name in supported_expansions:
        single_df = get_card_ratings_df(set_name, **card_ratings_kwargs)

        single_df["expansion"] = set_name.upper()
        if ratings_df is None:
            ratings_df = single_df
        else:
            ratings_df = pd.concat([ratings_df, single_df])
        time.sleep(1)  # try to be polite
    ratings_df = ratings_df.rename(columns={k: k.lower() for k in ratings_df.columns})  # type: ignore

    joined = pd.merge(
        ratings_df,
        card_ids_df,
        how="left",
        left_on=["name", "expansion"],
        right_on=["name", "expansion"],
        suffixes=["", "_y"],
    )
    joined.to_csv(f"{data_path}/card_ratings_data.csv", index=False)
    return joined.set_index("id")