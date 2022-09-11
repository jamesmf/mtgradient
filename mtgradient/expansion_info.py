import typing as T


class MTGSet:

    n_cards_in_pack = 16
    n_packs = 3

    def __init__(self):
        self.n_picks_per_pack = self.n_cards_in_pack - 1
        self.total_picks = self.n_packs * self.n_picks_per_pack


class NEO(MTGSet):
    def __init__(self):
        super().__init__()


class HBG(MTGSet):
    def __init__(self):
        self.n_cards_in_pack = 15
        super().__init__()


class SNC(MTGSet):
    def __init__(self):
        self.n_cards_in_pack = 15
        super().__init__()


expansion_dict: T.Dict[str, MTGSet] = {"NEO": NEO(), "HBG": HBG(), "SNC": SNC()}
