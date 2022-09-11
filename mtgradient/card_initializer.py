import typing as T
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

from mtgradient.data_retrieval.ratings import read_data


def load_raw_card_data(
    scryfall_json_path: str, card_data_dir: str, refresh: bool = False
) -> pd.DataFrame:
    """Join the scryfall card data (bulk download) to the 17Lands
    dataset.

    Args:
        scryfall_json_path (str): full path to oracle JSON
        card_rating_df (pd.DataFrame): path do directory containing card ratings df
        refresh (bool): whether to refresh the data. Defaults to False

    Returns:
        pd.DataFrame: joined DataFrame
    """

    with open(scryfall_json_path, "r") as f:
        scryfall_data = json.load(f)

    card_rating_df = read_data(card_data_dir, refresh=refresh)

    relevant_scryfall = [
        card
        for card in scryfall_data
        if card["name"] in set(card_rating_df["name"].tolist())
    ]

    scryfall_df = pd.DataFrame(relevant_scryfall)[
        [
            "arena_id",
            "name",
            "mana_cost",
            "cmc",
            "type_line",
            "oracle_text",
            "power",
            "toughness",
            "colors",
            "keywords",
            "set",
        ]
    ]
    scryfall_df

    scryfall_df[scryfall_df["name"].str.contains("antern")]

    return pd.merge(
        card_rating_df, scryfall_df, how="left", left_on="name", right_on="name"
    ).drop_duplicates("name")


numeric_17lands_data_cols = [
    "alsa",
    "ata",
    "win_rate",
    "sideboard_win_rate",
    "oh wr",
    "drawn_win_rate",
    "gd wr",
    "never_drawn_win_rate",
    "iwd",
]


def to_stat_str(stats: T.Dict[str, T.Any]) -> str:
    cmc = f"cmc_{stats.get('cmc', '')}"
    colors = stats.get("colors", [])
    if not (isinstance(colors, list) or isinstance(colors, str)):
        colors = []
    else:
        colors = list(colors)
    if len(colors) > 0:
        colors_str = f"color_{''.join(colors)}"
    else:
        colors_str = ""

    body_str = ""
    pwr = stats.get("power")
    tgh = stats.get("toughness")
    if isinstance(pwr, int) or isinstance(pwr, str):
        body_str = f"power_{pwr} "
    if isinstance(tgh, int) or isinstance(tgh, str):
        body_str += f"toughness_{tgh}"

    keywords = stats.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    keyword_str = " ".join(["kw" + i.replace(" ", "_") for i in keywords])

    return " ".join(
        [
            str(i)
            for i in [
                cmc,
                colors_str,
                body_str,
                keyword_str,
                stats.get("type_line", ""),
            ]
            if i
        ]
    )


def featurize_cards(
    card_data: pd.DataFrame,
    card_ids: T.Dict[str, int],
    n_cards: int,
    emb_dim: int,
    n_components_text: int = 50,
):
    stat_vectorizer = CountVectorizer(min_df=10)
    text_vectorizer = CountVectorizer(max_df=0.1, min_df=10)
    pca = PCA(n_components=n_components_text)
    scaler = StandardScaler()
    imputer = SimpleImputer()

    card_data["stats_str"] = card_data.apply(
        lambda x: to_stat_str(x),
        axis=1,
    )
    stats_vec = stat_vectorizer.fit_transform(card_data["stats_str"].values)
    text_vec = text_vectorizer.fit_transform(
        card_data["oracle_text"].fillna("nan").values
    )
    text_vec = pca.fit_transform(np.asarray(text_vec.todense()))
    combined = np.concatenate(
        [
            imputer.fit_transform(card_data[numeric_17lands_data_cols].values),
            np.asarray(stats_vec.todense()),
            text_vec,
        ],
        axis=1,
    )
    combined = scaler.fit_transform(combined)

    weights = np.random.normal(0, 1, size=(n_cards, emb_dim))
    for n, card_name in enumerate(card_data["name"].values):
        card_id = card_ids.get(card_name, -1)
        if card_id > 0:
            vec = combined[n]
            weights[card_id, : vec.shape[0]] = vec

    return weights
