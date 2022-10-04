from collections import defaultdict
import json
import os
import pickle
import csv
import re
import datetime
import typing as T

import numpy as np
import pandas as pd
from tqdm import tqdm

from .constants import (
    TOTAL_PICKS,
    N_CARDS_IN_PACK,
    N_PICKS_PER_PACK,
    MAX_WHEEL_ROUND,
    PlayerRank,
)
from .expansion_info import expansion_dict


class WholeDraft(T.TypedDict, total=False):
    expansion: str
    rank_id: int
    rank: str
    draft_id: str
    draft_time: pd.Timestamp
    pack_data: T.List[T.List[int]]
    pick_data: T.List[int]
    maindeck_rates: T.List[int]
    wheels: T.List[T.List[int]]
    user_game_win_rate_bucket: float
    event_match_wins: int
    event_match_losses: int
    set_id_str: str
    set_id_int: int
    format_id_int: int


DictDataset = T.Dict[str, WholeDraft]


DRAFT_LEVEL_COLS = {
    "expansion": str,
    "event_type": str,
    "draft_id": str,
    "draft_time": pd.to_datetime,
    # "draft_time": str,
    "rank": str,
    "event_match_wins": int,
    "event_match_losses": int,
    "user_n_games_bucket": int,
    "user_game_win_rate_bucket": lambda x: float(x) if x else 0.5,
}


def pack_pick_to_index(
    pack: int, pick: int, n_cards_in_pack: int = N_PICKS_PER_PACK
) -> int:
    """Convert from a pack and pick to the index
    into the flattened pick list

    Args:
        pack (int): pack number
        pick (int): pick number

    Returns:
        int: index in the flattened pick data
    """
    return pack * n_cards_in_pack + pick


def round_to_pack_and_pick(
    round_num: int, n_cards_in_pack: int = N_PICKS_PER_PACK
) -> T.Tuple[int, int]:
    """convert a round

    Args:
        round_num (int): _description_

    Returns:
        T.Tuple[int, int]: _description_
    """
    pack_num = round_num // n_cards_in_pack
    pick_num = round_num % n_cards_in_pack
    return pack_num, pick_num


def get_colmap(cols: T.List[str]) -> T.Dict[int, T.Dict[str, str]]:
    """Convert the columns of the csv to an object that contains
    both the content of the column name and the 'type' of the
    column (pool, pack, other)

    Args:
        cols (T.List[str]): _description_

    Returns:
        T.Dict[int, T.Dict[str, str]]: column ind mapping
    """
    pack_patt = re.compile("pack_card_(.+)")
    pool_patt = re.compile("pool_(.+)")
    colmap = {}
    for n, col in enumerate(cols):
        if col == "user_rank":
            col = "rank"
        if col == "user_match_win_rate_bucket":
            col = "user_game_win_rate_bucket"
        if n < 11:
            colmap[n] = {"name": col, "type": "basic"}
        else:
            pack_match = re.findall(pack_patt, col)
            if pack_match:
                colmap[n] = {"name": pack_match[0], "type": "pack"}
                continue
            pool_match = re.findall(pool_patt, col)
            if pool_match:
                colmap[n] = {"name": pool_match[0], "type": "pool"}
                continue
            colmap[n] = {"name": col, "type": "basic"}

    return colmap


def parse_row(
    row: T.Sequence[T.Any],
    colmap: T.Dict[int, T.Dict[str, str]],
    card_map: T.Dict[str, int],
    pack_inds: T.List[int],
) -> T.Dict[str, T.Union[str, T.List[int], float, int]]:
    # base = {colmap[n]["name"]: val for n, val in enumerate(row[:11])}
    base = {v["name"]: row[k] for k, v in colmap.items() if v["type"] == "basic"}
    pack_names = [colmap[n]["name"] for n in pack_inds if row[n] == "1"]
    # pack_names += [base["pick"]]
    np.random.shuffle(pack_names)
    pick_ind = pack_names.index(base["pick"])
    for p in pack_names:
        if p not in card_map:
            card_map[p] = len(card_map)
    pack = [card_map[i] for i in pack_names]
    base["pack"] = pack
    base["pick_ind"] = pick_ind
    base["pick_maindeck_rate"] = float(base["pick_maindeck_rate"])
    return base


def add_row_to_draft_dataset(
    row: T.Dict[str, T.Union[str, T.List[int], float, int]], dataset: T.Dict[str, T.Any]
):
    draft_id = T.cast(str, row["draft_id"])
    expansion_str = str(row["expansion"])
    expansion = expansion_dict[expansion_str]
    if draft_id not in dataset:
        dataset[draft_id] = {
            k: DRAFT_LEVEL_COLS[k](row[k]) for k in row.keys() if k in DRAFT_LEVEL_COLS
        }
        dataset[draft_id]["pack_data"] = [[] for _ in range(expansion.total_picks)]  # type: ignore
        dataset[draft_id]["pick_data"] = [0 for _ in range(expansion.total_picks)]  # type: ignore
        dataset[draft_id]["maindeck_rates"] = [0 for _ in range(expansion.total_picks)]  # type: ignore
        dataset[draft_id]["rank_id"] = (
            PlayerRank[T.cast(str, row["rank"])].value
            if row["rank"] in PlayerRank.__members__
            else PlayerRank.gold.value
        )

    pack_number = int(row["pack_number"])  # type: ignore
    pick_number = int(row["pick_number"])  # type: ignore
    index = pack_pick_to_index(pack_number, pick_number, expansion.n_picks_per_pack)
    dataset[draft_id]["pack_data"][index] = row["pack"]  # type: ignore
    dataset[draft_id]["pick_data"][index] = row["pack"][row["pick_ind"]]  # type: ignore
    dataset[draft_id]["maindeck_rates"][index] = row["pick_maindeck_rate"]  # type: ignore


def add_wheels(ds: DictDataset):
    """Adds the 'wheels' attribute to each draft round"""
    for draft, draft_dict in ds.items():
        pack_data = T.cast(T.List[T.List[int]], draft_dict["pack_data"])
        wheels: T.List[T.List[int]] = []
        for row_ind, row in enumerate(pack_data):
            if row_ind > MAX_WHEEL_ROUND:
                continue
            wheel_data = []
            pack_plus_8: T.List[int] = pack_data[row_ind + 8]
            for card_id in pack_data[row_ind]:
                if card_id in pack_plus_8:
                    wheel_data.append(1)
                else:
                    wheel_data.append(0)
            wheels.append(wheel_data)
        draft_dict["wheels"] = wheels


def parse_csv(
    csv_path: str,
    verbose: bool = False,
    card_name_to_id: T.Optional[T.Dict[str, int]] = None,
    set_id_str: str = "",
    set_id_int: int = 0,
    format_id_int: int = 0,
    limit_rows: T.Optional[int] = None,
) -> T.Tuple[DictDataset, T.Dict[str, int]]:
    dataset = {}  # type: ignore
    cols = pd.read_csv(csv_path, nrows=0).columns
    colmap = get_colmap(cols)
    if card_name_to_id is None:
        card_name_to_id = {"": 0}
    pack_inds = [ind for ind, val in colmap.items() if val["type"] == "pack"]
    with open(csv_path, "r") as f:
        consumer = csv.reader(f)
        if verbose:
            consumer = tqdm(consumer)
        for n, row in enumerate(consumer):
            if n == 0:
                continue
            parsed = parse_row(row, colmap, card_name_to_id, pack_inds)
            add_row_to_draft_dataset(parsed, dataset)
            if limit_rows is not None and n > limit_rows:
                break
    # remove some data that doesn't look right
    for_removal = set()
    removal_reasons = defaultdict(list)
    for draft_id, draft in dataset.items():
        if draft["event_match_wins"] + draft["event_match_losses"] == 0:
            for_removal.add(draft_id)
            removal_reasons[draft_id].append("invalid_event_match_wins_plus_losses")
        if 0 in draft["pick_data"]:
            for_removal.add(draft_id)
            removal_reasons[draft_id].append("invalid_pick_data")
        if draft["rank_id"] < PlayerRank.gold.value:
            for_removal.add(draft_id)
            removal_reasons[draft_id].append("rank_below_gold")
        if float(draft["user_game_win_rate_bucket"]) <= 0.5:
            for_removal.add(draft_id)
            removal_reasons[draft_id].append("win_rate_too_low")
    if verbose:
        print(f"number of drafts: {len(dataset)}")
        print(f"number to remove: {len(for_removal)}")
    for draft_id in list(for_removal):
        dataset.pop(draft_id)
        # if verbose:
        #     print(f"{draft_id} removed because: {removal_reasons[draft_id]}")
    add_wheels(dataset)
    for key in dataset.keys():
        raw = dataset[key]
        dataset[key] = WholeDraft(
            expansion=raw["expansion"],
            rank=raw["rank"],
            rank_id=raw["rank_id"],
            draft_id=raw["draft_id"],
            draft_time=raw["draft_time"],
            pick_data=raw["pick_data"],
            pack_data=raw["pack_data"],
            maindeck_rates=raw["maindeck_rates"],
            wheels=raw["wheels"],
            user_game_win_rate_bucket=raw["user_game_win_rate_bucket"],
            event_match_wins=raw["event_match_wins"],
            event_match_losses=raw["event_match_losses"],
            set_id_int=set_id_int,
            set_id_str=set_id_str,
            format_id_int=format_id_int,
        )
    return dataset, card_name_to_id


def persist_processed_dataset(
    path: str, dataset: DictDataset, card_map: T.Dict[str, int]
):
    """Pickle a DictDataset you've already processed

    Args:
        path: (str): path to persist the dataset
        dataset (DictDataset): the parsed data
        card_map (T.Dict[str, int]): mapping between card name and integer id
    """
    ds_path = os.path.join(path, "dataset.pkl")
    cm_path = os.path.join(path, "card_map.json")

    with open(ds_path, "wb") as f:
        pickle.dump(dataset, f)

    with open(cm_path, "w") as f2:
        json.dump(card_map, f2, indent=2)


def load_processed_dataset(
    path: str,
) -> T.Tuple[DictDataset, T.Dict[str, int]]:
    """Load a pickled dataset and card map

    Args:
        path (str): path containing the persisted objects

    Returns:
        T.Tuple[DictDataset, T.Dict[str, int]]: dataset and card map
    """
    ds_path = os.path.join(path, "dataset.pkl")
    cm_path = os.path.join(path, "card_map.json")

    with open(ds_path, "rb") as f:
        ds = pickle.load(f)

    with open(cm_path, "r") as f2:
        cm = json.load(f2)
    return ds, cm