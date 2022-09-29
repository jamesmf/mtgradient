import json
import typing as T
import datetime
import os

import numpy as np
import torch

from .processing import (
    round_to_pack_and_pick,
    WholeDraft,
    parse_csv,
    persist_processed_dataset,
)
from .constants import (
    N_PACKS,
    N_CARDS_IN_PACK,
    TOTAL_PICKS,
    N_PICKS_PER_PACK,
    PlayerRank,
)
from .expansion_info import MTGSet, expansion_dict

# downweight the first pick (anti-raredrafting)
# and place most of the emphasis on the rounds with
# a meaningful number of options
DEFAULT_PICK_WEIGHTING = {
    0: 0.4,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 0.9,
    6: 0.8,
    7: 0.7,
    8: 0.6,
    9: 0.5,
    10: 0.4,
    11: 0.3,
    12: 0.2,
    13: 0.1,
    14: 0.1,
    15: 0.1,
}

# weight picks at each round differently based on
# how much of the draft we've seen. At the beginning
# of the draft, we shouldn't penalize for not estimating
# well how many wins we'll get. But by the end of the
# draft, we should be able to estimate how our deck will
# perform
DEFAULT_WIN_PRED_WEIGHING = {k: (k + 1) / TOTAL_PICKS for k in range(TOTAL_PICKS)}

# weight loss for higher levels of play much higher
DEFAULT_WEIGHTING_BY_RANK = {
    PlayerRank.mythic.name: 8,
    PlayerRank.diamond.name: 8,
    PlayerRank.platinum.name: 4,
    PlayerRank.gold.name: 1,
    PlayerRank.silver.name: 0.25,
    PlayerRank.bronze.name: 0.125,
    "": 0.01,
}


class DraftDataPoint(T.TypedDict, total=False):
    history: T.List[T.List[int]]
    pool: T.List[int]
    options: T.List[int]
    wins_weight: int
    pick_weight: int
    picked: int
    num_wins: int
    draft: T.Dict[str, T.Any]
    round: int
    maindeck_rates: T.List[float]
    rank: str
    rank_id: int
    set_id_int: int
    format_id_int: int


class DraftDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: T.Dict[str, WholeDraft],
        recent_cutoff: datetime.date,
        card_id_offset: int = 0,
        recency_weight: float = 0.15,
        pick_weighting: T.Dict[int, float] = DEFAULT_PICK_WEIGHTING,
        win_pred_weighting: T.Dict[int, float] = DEFAULT_WIN_PRED_WEIGHING,
        rank_weighting: T.Dict[str, float] = DEFAULT_WEIGHTING_BY_RANK,
        total_picks: int = TOTAL_PICKS,
        n_picks_in_pack: int = N_PICKS_PER_PACK,
        static_pick_indices: T.Optional[T.Dict[int, int]] = None,
        use_all: bool = False,
    ):
        self.dataset = dataset
        self.idx_to_draft = {n: draft_id for n, draft_id in enumerate(dataset.keys())}
        self.recent_cutoff = recent_cutoff
        self.card_id_offset = card_id_offset
        self.recency_weight = recency_weight
        self.pick_weighting = pick_weighting
        self.win_pred_weighting = win_pred_weighting
        self.total_picks = total_picks
        self.n_picks_in_pack = n_picks_in_pack
        self.rank_weighting = rank_weighting
        self.static_pick_indices = static_pick_indices
        self.use_all = use_all

    def __len__(self):
        if not self.use_all:
            return len(self.dataset)
        else:
            return self.total_picks * len(self.dataset)

    def __getitem__(self, idx: int):
        if not self.use_all:
            draft = self.dataset[self.idx_to_draft[idx]]
            if self.static_pick_indices is None:
                round_num = np.random.randint(0, len(draft["pack_data"]))
            else:
                round_num = self.static_pick_indices[idx]
        else:
            draft_num = int(idx / 3 / self.n_picks_in_pack)
            draft = self.dataset[self.idx_to_draft[draft_num]]
            round_num = idx % (3 * self.n_picks_in_pack)
            # if we combine drafts with different number of rounds, lazily
            # augment the smaller drafts
            if round_num >= len(draft["pack_data"]):
                round_num = np.random.randint(0, len(draft["pack_data"]))

        pick_weight, wins_weight = self.get_weights(draft, round_num)

        return {
            "history": [
                [i + self.card_id_offset for i in hist_round]
                for hist_round in draft["pack_data"][:round_num]
            ],
            "pool": [
                pool_i + self.card_id_offset
                for pool_i in draft["pick_data"][:round_num]
            ],
            "options": [
                opt_i + self.card_id_offset for opt_i in draft["pack_data"][round_num]
            ],
            "wins_weight": wins_weight,
            "pick_weight": pick_weight,
            "picked": draft["pick_data"][round_num] + self.card_id_offset,
            "num_wins": draft["event_match_wins"],
            "draft": draft,
            "round": round_num,
            "maindeck_rates": draft["maindeck_rates"],
            "rank": draft["rank"],
            "rank_id": draft["rank_id"],
            "set_id_int": draft["set_id_int"],
            "format_id_int": draft["format_id_int"],
        }

    def get_weights(self, draft: WholeDraft, round_num: int) -> T.Tuple[float, float]:
        # pay more attention if the user wins a lot, and less if they don't
        user_wins_factor: float = np.clip(
            draft.get("user_game_win_rate_bucket", 0.5) * 2, 0.6, 1.2
        )

        # put more weight on more-winning decks
        wins_factor: float = draft.get("event_match_wins", 3) / 14 + 0.5

        # weight loss by rank
        rank = draft["rank"]
        rank_weighting = self.rank_weighting[rank]

        pack_num, pick_num = round_to_pack_and_pick(
            round_num, n_cards_in_pack=self.n_picks_in_pack
        )
        pick_weight = (
            self.pick_weighting.get(pick_num, 0.05)
            * rank_weighting
            * wins_factor
            * user_wins_factor
        )

        num_wins_weight = (
            self.win_pred_weighting.get(round_num, 1)
            * np.min([rank_weighting, 1])
            * user_wins_factor
        )
        # if they didn't play until 7wins/3losses, then don't try to predict wins at all
        mw: int = draft.get("event_match_wins", 0)
        if mw < 7 and draft.get("event_match_losses", 0) != 3:
            num_wins_weight = 0
        return (pick_weight, num_wins_weight)


class MultiDataset:

    format_ids = {"premier": 0, "traditional": 1, "quick": 2}

    def __init__(
        self,
        expansion_configs: T.List[T.Dict[str, T.Any]],
        limit_per: T.Optional[int] = None,
    ):
        self.train_datasets: T.List[DraftDataset] = []
        self.val_datasets: T.List[DraftDataset] = []
        self.test_datasets: T.List[DraftDataset] = []
        self.card_ids: T.Optional[T.Dict[str, int]] = None
        self.limit_per = limit_per

        for expansion_data in expansion_configs:
            self.add_set(
                set_id=expansion_data["set_id"],  # type: ignore
                dataset_details=expansion_data["datasets"],  # type: ignore
                set_id_int=expansion_data["set_id_int"],  # type: ignore
            )

    def add_set(
        self,
        set_id: str,
        dataset_details: T.List[T.Dict[str, T.Union[str, bool]]],
        set_id_int: int,
    ):
        expansion = expansion_dict[set_id]
        set_train_datasets = []
        set_val_datasets = []
        set_test_datasets = []

        for format_config in dataset_details:
            draft_csv_path = T.cast(str, format_config["file"])
            format_str = T.cast(str, format_config["format"])
            format_int = T.cast(int, format_config["format_id_int"])
            parsed_data, self.card_ids = parse_csv(
                draft_csv_path,
                verbose=True,
                card_name_to_id=self.card_ids,
                set_id_str=set_id,
                set_id_int=set_id_int,
                format_id_int=format_int,
                limit_rows=self.limit_per,
            )
            recent_game_cutoff = expansion.recent_game_dates[format_str]
            val_split_cutoff = expansion.val_dates[format_str]
            test_split_cutoff = expansion.test_dates[format_str]
            train_subset = {
                k: v
                for k, v in parsed_data.items()
                if v.get("draft_time", datetime.date(1999, 9, 9)) < test_split_cutoff
            }
            val_subset = {
                k: v
                for k, v in parsed_data.items()
                if v.get("draft_time", datetime.date(1999, 9, 9)) >= val_split_cutoff
                and v.get("draft_time", datetime.date(1999, 9, 9)) < test_split_cutoff
            }
            test_subset = {
                k: v
                for k, v in parsed_data.items()
                if v.get("draft_time", datetime.date(1999, 9, 9)) >= test_split_cutoff
            }

            train_draft_dataset = DraftDataset(train_subset, recent_game_cutoff)
            set_train_datasets.append(train_draft_dataset)

            if not format_config.get("training_only", False):
                val_draft_dataset = DraftDataset(
                    val_subset, val_split_cutoff, use_all=True
                )
                test_draft_dataset = DraftDataset(
                    test_subset, test_split_cutoff, use_all=True
                )

                set_val_datasets.append(val_draft_dataset)
                set_test_datasets.append(test_draft_dataset)

        self.train_datasets.extend(set_train_datasets)
        self.val_datasets.extend(set_val_datasets)
        self.test_datasets.extend(set_test_datasets)