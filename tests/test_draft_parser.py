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
    assert len(ds) == 3, "incorrect number of drafts when parsing test dataset"

    assert (
        "dace9c349b224cf4a12a151d578ce870" in ds
    ), "expected match id dace9c349b224cf4a12a151d578ce870 in parsed data, not found"


def test_card_map_test_data(parsed_csv):
    _, card_map = parsed_csv
    print(card_map)
    print(f"len card map: {len(card_map)}")
    assert (
        len(card_map) == 217
    ), "incorrect number of cards found when generating card map"
