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

from .constants import TOTAL_PICKS, N_CARDS_IN_PACK, N_PICKS_PER_PACK

# To simplify the data, we store a dict with the draft id as the key
# and the the draft-level data. The picks and options are transformed
# to a List[int] (ids of picked cards in our id map) and a List[List[int]]
# (each entry is the ids of the possible cards to pick at that round)
DraftDataType = T.Dict[
    str,
    T.Union[
        str,
        int,
        T.List[int],
        T.List[T.List[int]],
    ],
]
DictDataset = T.Dict[str, DraftDataType]


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
    pack_col_bounds: T.Tuple[int, int],
) -> T.Dict[str, T.Union[str, T.List[int]]]:
    # base = {colmap[n]["name"]: val for n, val in enumerate(row[:11])}
    base = {v["name"]: row[k] for k, v in colmap.items() if v["type"] == "basic"}
    pack_names = [
        colmap[n + pack_col_bounds[0]]["name"]
        for n, val in enumerate(row[pack_col_bounds[0] : pack_col_bounds[1] + 1])
        if val == "1"
    ]
    # pack_names += [base["pick"]]
    np.random.shuffle(pack_names)
    pick_ind = pack_names.index(base["pick"])
    for p in pack_names:
        if p not in card_map:
            card_map[p] = len(card_map)
    pack = [card_map[i] for i in pack_names]
    base["pack"] = pack
    base["pick_ind"] = pick_ind
    return base


def add_row_to_draft_dataset(
    row: T.Dict[str, T.Union[str, T.List[int]]], dataset: T.Dict[str, T.Any]
):
    draft_id = row["draft_id"]
    if draft_id not in dataset:
        dataset[draft_id] = {k: DRAFT_LEVEL_COLS[k](row[k]) for k in row.keys() if k in DRAFT_LEVEL_COLS}  # type: ignore
        dataset[draft_id]["pack_data"] = [[] for _ in range(TOTAL_PICKS)]  # type: ignore
        dataset[draft_id]["pick_data"] = [0 for _ in range(TOTAL_PICKS)]  # type: ignore

    pack_number = int(row["pack_number"])  # type: ignore
    pick_number = int(row["pick_number"])  # type: ignore
    index = pack_pick_to_index(pack_number, pick_number)
    dataset[draft_id]["pack_data"][index] = row["pack"]  # type: ignore
    dataset[draft_id]["pick_data"][index] = row["pack"][row["pick_ind"]]  # type: ignore


def parse_csv(
    csv_path: str, verbose: bool = True
) -> T.Tuple[DictDataset, T.Dict[str, int]]:
    dataset = {}  # type: ignore
    cols = pd.read_csv(csv_path, nrows=0).columns
    colmap = get_colmap(cols)
    card_name_to_id: T.Dict[str, int] = {"": 0}
    pack_inds = [ind for ind, val in colmap.items() if val["type"] == "pack"]
    pack_col_bounds = (min(pack_inds), max(pack_inds) + 1)
    with open(csv_path, "r") as f:
        consumer = csv.reader(f)
        if verbose:
            consumer = tqdm(consumer)
        for n, row in enumerate(consumer):
            if n == 0:
                continue
            parsed = parse_row(row, colmap, card_name_to_id, pack_col_bounds)
            add_row_to_draft_dataset(parsed, dataset)
    # remove some data that doesn't look right
    for_removal = set()
    for draft_id, draft in dataset.items():
        if draft["event_match_wins"] + draft["event_match_losses"] == 0:
            for_removal.add(draft_id)
        if 0 in draft["pick_data"]:
            for_removal.add(draft_id)
    for draft_id in list(for_removal):
        dataset.pop(draft_id)
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