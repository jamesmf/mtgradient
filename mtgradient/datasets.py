import json
import typing as T
import datetime

import numpy as np
import torch

from .processing import DraftDataType, round_to_pack_and_pick
from .constants import N_ROUNDS, N_CARDS_IN_PACK, TOTAL_PICKS, N_PICKS_PER_PACK

# downweight the first pick (anti-raredrafting)
# and place most of the emphasis on the rounds with
# a meaningful number of options
DEFAULT_PICK_WEIGHTING = {
    0: 0.6,
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
    "mythic": 3,
    "diamond": 2.5,
    "platinum": 1.5,
    "gold": 0.5,
    "silver": 0.25,
    "bronze": 0.125,
    "": 0.01,
}


class DraftDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: T.Dict[str, T.Any],
        recent_cutoff: datetime.date,
        recency_weight: float = 0.15,
        pick_weighting: T.Dict[int, float] = DEFAULT_PICK_WEIGHTING,
        win_pred_weighting: T.Dict[int, float] = DEFAULT_WIN_PRED_WEIGHING,
        rank_weighting: T.Dict[str, float] = DEFAULT_WEIGHTING_BY_RANK,
        num_rounds: int = N_ROUNDS,
        n_picks_in_pack: int = N_PICKS_PER_PACK,
        static_pick_indices: T.Optional[T.Dict[int, int]] = None,
        use_all: bool = False,
    ):
        self.dataset = dataset
        self.idx_to_draft = {n: draft_id for n, draft_id in enumerate(dataset.keys())}
        self.recent_cutoff = recent_cutoff
        self.recency_weight = recency_weight
        self.pick_weighting = pick_weighting
        self.win_pred_weighting = win_pred_weighting
        self.num_rounds = num_rounds
        self.n_picks_in_pack = n_picks_in_pack
        self.rank_weighting = rank_weighting
        self.static_pick_indices = static_pick_indices
        self.use_all = use_all

    def __len__(self):
        if not self.use_all:
            return len(self.dataset)
        else:
            return self.num_rounds * len(self.dataset)

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

        pick_weight, wins_weight = self.get_weights(draft, round_num)

        return {
            "history": draft["pack_data"][:round_num],
            "pool": draft["pick_data"][:round_num],
            "options": draft["pack_data"][round_num],
            "wins_weight": wins_weight,
            "pick_weight": pick_weight,
            "picked": draft["pick_data"][round_num],
            "num_wins": draft["event_match_wins"],
            "draft": draft,
            "round": round_num,
        }

    def get_weights(
        self, draft: DraftDataType, round_num: int
    ) -> T.Tuple[float, float]:
        # pay more attention if the user wins a lot, and less if they don't
        user_wins_factor: float = np.clip(draft.get("user_game_win_rate_bucket", 0.5) * 2, 0.6, 1.2)  # type: ignore

        # put more weight on more-winning decks
        wins_factor: float = draft.get("event_match_wins", 3) / 14 + 0.5  # type: ignore

        # weight loss by rank
        rank: str = draft["rank"]  # type: ignore
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
        mw: int = draft.get("event_match_wins", 0)  # type: ignore
        if mw < 7 and draft.get("event_match_losses", 0) != 3:
            num_wins_weight = 0
        return (pick_weight, num_wins_weight)
