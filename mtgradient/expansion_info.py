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


class MID(MTGSet):
    name = "MID"

    def __init__(self):
        self.n_cards_in_pack = 15
        self.val_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2021, 10, 16, 0),
        }
        self.test_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2021, 10, 17, 0),
        }
        self.recent_game_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2022, 10, 10, 0),
        }
        super().__init__()


class VOW(MTGSet):
    name = "VOW"

    def __init__(self):
        self.n_cards_in_pack = 15
        self.val_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2021, 12, 1, 0),
            "quick": datetime.datetime(2023, 12, 1, 0),
        }
        self.test_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2021, 12, 1, 12),
            "quick": datetime.datetime(2023, 12, 1, 0),
        }
        self.recent_game_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2021, 11, 30, 12),
            "quick": datetime.datetime(2023, 12, 1, 0),
        }
        super().__init__()


class NEO(MTGSet):
    name = "NEO"

    def __init__(self):
        self.val_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2022, 3, 9, 0),
        }
        self.test_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2022, 3, 10, 0),
        }
        self.recent_game_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2022, 3, 1, 0),
        }
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
            "premier": datetime.datetime(2022, 10, 11, 0),
            "traditional": datetime.datetime(2022, 10, 11, 0),
        }
        self.test_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2022, 10, 12, 6),
            "traditional": datetime.datetime(2022, 10, 12, 6),
        }
        self.recent_game_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2022, 10, 7, 0),
            "traditional": datetime.datetime(2022, 10, 7, 0),
        }
        super().__init__()


class ONE(MTGSet):
    name = "ONE"
    release_date = datetime.datetime(2023, 2, 7, 0),
    def __init__(self):
        self.n_cards_in_pack = 15
        self.val_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2023, 2, 25, 12),
            # "traditional": datetime.datetime(2023, 10, 11, 0),
        }
        self.test_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2023, 2, 26, 18),
            # "traditional": datetime.datetime(2023, 10, 12, 6),
        }
        self.recent_game_dates: T.Dict[str, datetime.date] = {
            "premier": datetime.datetime(2023, 2, 15, 0),
            # "traditional": datetime.datetime(2023, 10, 7, 0),
        }
        super().__init__()


expansion_dict: T.Dict[str, MTGSet] = {
    s.name: s for s in (MID(), NEO(), HBG(), SNC(), DMU(), VOW(), ONE())
}
