import os
import json

import pytest

from mtgradient.processing import parse_csv

curr_path = os.path.split(os.path.abspath(__file__))[0]
test_draft_data_path = os.path.join(curr_path, "testdata", "test_premier_draft.csv")


@pytest.fixture(scope="module")
def parsed_csv():
    return parse_csv(test_draft_data_path)


def test_parsed_all_test_data(parsed_csv):
    ds, _ = parsed_csv
    assert len(ds) == 2, "incorrect number of drafts when parsing test dataset"

    assert (
        "dace9c349b224cf4a12a151d578ce870" in ds
    ), "expected match id dace9c349b224cf4a12a151d578ce870 in parsed data, not found"


def test_card_map_test_data(parsed_csv):
    _, card_map = parsed_csv
    print(card_map)
    print(f"len card map: {len(card_map)}")
    assert (
        len(card_map) == 196
    ), "incorrect number of cards found when generating card map"


def test_parse_row_1(parsed_csv):
    """Test that the first pick in the first row comes out with the
    proper options when passed through parse_row
    """
    ds, card_ids = parsed_csv
    # the first round of the first draft should have the following options
    options = [
        "Akki Ronin",
        "Assassin's Ink",
        "Bamboo Grove Archer",
        "Chainflail Centipede",
        "Go-Shintai of Ancient Wars",
        "Imperial Subduer",
        "Jukai Trainee",
        "Lethal Exploit",
        "March of Swirling Mist",
        "Moonsnare Prototype",
        "Okiba Reckoner Raid",
        "Plains",
        "Spirited Companion",
        "Thirst for Knowledge",
        "Virus Beetle",
    ]
    card_id_set = set([card_ids[i] for i in options])
    p0 = set(ds["dace9c349b224cf4a12a151d578ce870"]["pack_data"][0])
    print(p0)
    print(card_id_set)
    assert (
        len(p0.difference(card_id_set)) == 0
    ), "incorrect parsing of the p1p1 (round 0) of the first draft in the test dataset"


def test_wheel_logic(parsed_csv):
    """Test that we correctly label which cards in a pack wheel"""
    ds, card_ids = parsed_csv
    card_ids_inv = {v: k for k, v in card_ids.items()}
    card_names_pack0 = [
        card_ids_inv[i] for i in ds["dace9c349b224cf4a12a151d578ce870"]["pack_data"][0]
    ]
    print(ds["dace9c349b224cf4a12a151d578ce870"])

    assert (
        "wheels" in ds["dace9c349b224cf4a12a151d578ce870"]
    ), f"key 'wheels' not found in draft"

    wheel_data = ds["dace9c349b224cf4a12a151d578ce870"]["wheels"][0]

    cards_that_should_wheel = [
        "Akki Ronin",
        "Chainflail Centipede",
        "Jukai Trainee",
        "Moonsnare Prototype",
        "Okiba Reckoner Raid",
        "Plains",
        "Virus Beetle",
    ]
    for ind, card_name in enumerate(card_names_pack0):
        wheel = wheel_data[ind]
        if card_name in cards_that_should_wheel:
            assert wheel == 1, f"Card that should wheel {card_name} had wheel = {wheel}"


def test_maindeck_rate_in_draft(parsed_csv):
    ds, card_ids = parsed_csv
    maindeck_rates = ds["dace9c349b224cf4a12a151d578ce870"]["maindeck_rates"]
    assert (
        maindeck_rates[0] == 1
    ), f"maindeck_rate of dace9c349b224cf4a12a151d578ce870 pick 0 should be 1.0, found {maindeck_rates[0]}"
    assert (
        min(maindeck_rates) == 0
    ), f"min(maindeck_rates) of dace9c349b224cf4a12a151d578ce870 should be 0"
    assert (
        max(maindeck_rates) == 1
    ), f"max(maindeck_rates) of dace9c349b224cf4a12a151d578ce870 should be 1"


# def test_fail(parsed_csv):
#     ds, card_ids = parsed_csv
#     print(ds["dace9c349b224cf4a12a151d578ce870"])
#     assert False
