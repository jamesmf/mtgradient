import typing as T
import datetime


class MTGSet:
    name: str
    val_dates: T.Dict[str, datetime.date]
    test_dates: T.Dict[str, datetime.date]
    recent_game_dates: T.Dict[str, datetime.date]

    n_cards_in_pack = 16
    n_packs = 3

    def __init__(self):
        self.n_picks_per_pack = self.n_cards_in_pack - 1
        self.total_picks = self.n_packs * self.n_picks_per_pack


class NEO(MTGSet):
    name = "NEO"

    def __init__(self):
        super().__init__()


class HBG(MTGSet):
    name = "HBG"

    def __init__(self):
        self.n_cards_in_pack = 15
        super().__init__()


class SNC(MTGSet):
    name = "SNC"

    def __init__(self):
        self.n_cards_in_pack = 15
        self.val_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2022, 6, 17, 10),
        }
        self.test_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.date(2022, 6, 18)
        }
        self.recent_game_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2022, 6, 11, 0),
        }
        super().__init__()


class DMU(MTGSet):
    name = "DMU"

    def __init__(self):
        self.n_cards_in_pack = 15
        self.val_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.date(2022, 9, 15),
            "traditional": datetime.datetime(2022, 9, 15, 10),
        }
        self.test_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.date(2022, 9, 16),
            "traditional": datetime.datetime(2022, 9, 16, 0),
        }
        self.recent_game_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.date(2022, 9, 11),
            "traditional": datetime.datetime(2022, 9, 11, 0),
        }
        super().__init__()


expansion_dict: T.Dict[str, MTGSet] = {
    s.name: s
    for s in (
        NEO(),
        HBG(),
        SNC(),
        DMU(),
    )
}
