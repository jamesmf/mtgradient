import imp
import torch
import pytorch_lightning as pl
import numpy as np


class DraftTransformer(pl.LightningModule):
    """This model is meant to accept a single
    draft round as input and predict both which
    card will be picked in that round and how
    many wins the resulting deck will get.

    The model will receive the history of the
    draft to that point as a List[List[int]]
    representing the cards available in each
    prior round. It will also receive the ids
    of every card currently in the pool.
    """

    def __init__(self, n_cards: int, emb_dim: int):
        super().__init__()

        self.n_cards = n_cards
        # embeds a card id
        self.card_embedding = torch.nn.Embedding(
            n_cards,
            emb_dim,
        )

        # embedding to identify whether an card in an attention layer
        # is part of the current round's options or if it's in the
        # pool
        self.pick_vs_pool_embedding = torch.nn.Embedding(2, emb_dim)
