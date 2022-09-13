from enum import Enum


class PlayerRank(Enum):
    bronze = 0
    silver = 1
    gold = 2
    platinum = 3
    diamond = 4
    mythic = 5


# May be wasteful, but static shape is helpful
N_CARDS_IN_PACK = 16
N_PICKS_PER_PACK = N_CARDS_IN_PACK - 1
N_PACKS = 3
TOTAL_PICKS = N_PACKS * N_PICKS_PER_PACK

# estimates about whether something will wheel are
# only useful in rounds where we expect people to be
# taking good cards; after a certain point, it's
# extremely noisy
MAX_WHEEL_ROUND = 4