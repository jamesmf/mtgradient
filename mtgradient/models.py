import typing as T
import json
import torch
import pytorch_lightning as pl
import numpy as np
from torchmetrics import Accuracy, MeanAbsoluteError
import torchmetrics.functional
import sklearn.metrics

from .constants import N_PACKS, N_CARDS_IN_PACK


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

    def __init__(
        self,
        n_cards: int,
        emb_dim: int,
        n_cards_in_pack: int = N_CARDS_IN_PACK,
        n_heads_pool: int = 8,
        n_layers_pool: int = 3,
        n_heads_pick: int = 8,
        n_layers_pick: int = 6,
        n_steps: int = 30000,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.n_cards = n_cards
        self.n_cards_in_pack = n_cards_in_pack
        self.total_picks = self.n_cards_in_pack * 3
        self.n_steps = n_steps

        # embeds a card id
        self.card_embedding = torch.nn.Embedding(
            n_cards,
            emb_dim,
        )

        # embedding to identify whether an card in an attention layer
        # is part of the current round's options or if it's in the
        # pool
        self.pick_vs_pool_embedding = torch.nn.Embedding(2, emb_dim)

        self.layernorm = torch.nn.LayerNorm(emb_dim)

        # # embedding for each round
        # self.round_embedding = torch.nn.Embedding(self.total_picks, emb_dim)
        pool_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads_pool,
            batch_first=True,
            activation="gelu",
        )
        self.pool_transformer = torch.nn.TransformerEncoder(
            pool_encoder_layer,
            num_layers=n_layers_pool,
        )
        self.wins_linear = torch.nn.Sequential(
            torch.nn.GLU(), torch.nn.Linear(int(emb_dim / 2), 1)
        )

        self.maindeck_pred_head = torch.nn.Sequential(
            torch.nn.GLU(),
            torch.nn.Linear(int(emb_dim / 2), emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(emb_dim, 1),
        )

        pick_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads_pick,
            batch_first=True,
            activation="gelu",
        )
        self.pick_transformer = torch.nn.TransformerEncoder(
            pick_encoder_layer,
            num_layers=n_layers_pick,
        )
        self.pick_linear = torch.nn.Sequential(
            torch.nn.GLU(),
            torch.nn.Linear(int(emb_dim / 2), 1),
            torch.nn.LogSoftmax(dim=1),
        )
        self.dropout = torch.nn.Dropout(dropout)
        # self.softmax = torch.nn.Softmax(dim=1)

        self.n_steps_oclr = self.n_steps

        self.pick_loss = torch.nn.NLLLoss(reduction="none")
        self.wins_loss = torch.nn.L1Loss(reduction="none")
        self.maindeck_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.metrics: T.Dict[str, T.Union[Accuracy, MeanAbsoluteError]] = {}
        for split in ("train", "val", "test"):
            acc = Accuracy(average="samples")
            setattr(self, f"{split}_accuracy", acc)
            self.metrics[split] = acc
            # for n in range(self.n_cards_in_pack * 3):
            #     accn = Accuracy(average="samples")
            #     key = f"{split}_accuracy_round_{n:0>2}"
            #     setattr(self, key, accn)
            #     self.metrics[key] = accn

            #     mae = MeanAbsoluteError()
            #     key = f"{split}_mae_round_{n:0>2}"
            #     setattr(self, key, mae)
            #     self.metrics[key] = mae

    def init_embedding(self, weights: np.ndarray):
        """Initialize the embedding layer with card data

        Args:
            weights (torch.Tensor): output from card_initializer.featurize_cards()

        Returns:
            _type_: None
        """
        t = torch.tensor(weights, dtype=torch.float)
        self.card_embedding = torch.nn.Embedding.from_pretrained(
            t.to(self.device), freeze=False, padding_idx=0
        )

    # @torch.jit.script
    def _forward(
        self,
        x: T.Dict[str, torch.Tensor],
        device: str,
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore

        history = x["history"]  # (batch_size, total_picks, n_cards_in_pack)
        pool = x["pool"]  # (batch_size, max_round_in_batch)
        pick_options = x["pick_options"]  #  (batch_size, n_cards_in_pack)

        history_emb = self.dropout(
            self.card_embedding(history)
        )  # (batch_size, max_round_in_batch, n_cards_in_pack, emb_dim)
        pool_emb = self.dropout(
            self.card_embedding(pool)
        )  # (batch_size, max_round_in_batch, emb_dim)
        pick_options_emb = self.dropout(
            self.card_embedding(pick_options)
        )  #  (batch_size, n_cards_in_pack, emb_dim)

        # take the pool data and use it to predict the number of wins
        # the pool we've drafted might win
        # TODO: add rank embedding
        pool_mask = pool == 0
        pool_mask_zero = torch.tensor(
            [[False] for _ in range(pool.shape[0])],
            dtype=torch.bool,
            device=device,
        )

        pool_mask = torch.concat((pool_mask_zero, pool_mask), dim=1)
        pool_mean = pool_emb.mean(dim=1).unsqueeze(1)
        pool_enc = self.pool_transformer(
            torch.concat((pool_mean, pool_emb), dim=1),
            src_key_padding_mask=pool_mask,
        )
        win_prob = torch.clip(self.wins_linear(pool_enc[:, 0]), 0, 7)[:, 0]

        # use the same representation to predict which cards will make the maindeck
        maindeck_preds = self.maindeck_pred_head(pool_enc[:, 1:])

        # add an embedding representing whether an index in the eventual output
        # is a current pick option or a part of the pool you've already drafted
        pick_marker = self.pick_vs_pool_embedding(
            torch.ones(pick_options_emb.shape[:2], dtype=torch.int, device=self.device)
        )
        pool_marker = self.pick_vs_pool_embedding(
            torch.zeros(pool_emb.shape[:2], dtype=torch.int, device=self.device)
        )

        pick_options_emb = pick_marker + pick_options_emb
        pool_emb = pool_marker + pool_emb

        # reduce the history of the draft down to 1 vector per round
        # history_emb = self.layernorm(
        #     torch.sum(history_emb, dim=2)
        # )  # (batch_size, max_round_in_batch, emb_dim)
        history_emb = torch.sum(history_emb, dim=2)

        # now concatenate the pick options, the pool so far and the history of cards passed
        # and pass it through the transformer
        concatenated = torch.cat([pick_options_emb, pool_emb, history_emb], dim=1)
        concatenated = self.dropout(concatenated)
        pick_enc = self.pick_transformer(concatenated)[:, : self.n_cards_in_pack]
        pick_linear_outputs = self.pick_linear(pick_enc)[:, :, 0]

        return pick_linear_outputs, win_prob, maindeck_preds

    def forward(
        self, x: T.Dict[str, torch.Tensor] = {}, *args, **kwargs
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pick, win, maindeck_preds = self._forward(x, self.device)  # type: ignore
        return pick, win, maindeck_preds

    def step(self, x: T.Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        pick_preds, win_preds, maindeck_preds = self(x)

        # compute the mask for the maindeck rates. the model shouldn't see
        # info about any future picks' maindeck_rate
        maindeck_mask = x["maindeck_rates"] != -1
        maindeck_loss_value = self.maindeck_loss(
            maindeck_preds[maindeck_mask],
            x["maindeck_rates"][maindeck_mask].reshape(-1, 1),
        )
        weighted_maindeck_loss_value = maindeck_loss_value * x["maindeck_weights"]

        self.log(f"loss/{mode}/maindeck_loss", maindeck_loss_value.mean())
        self.log(
            f"loss/{mode}/maindeck_loss_weighted", weighted_maindeck_loss_value.mean()
        )

        pick_loss = self.pick_loss(pick_preds, x["picks"])
        self.log(f"loss/{mode}/pick_loss", pick_loss.mean())
        weighted_pick = pick_loss * x["pick_weights"]
        self.log(f"loss/{mode}/pick_loss_weighted", weighted_pick.mean())

        wins_loss = self.wins_loss(win_preds, x["num_wins"])
        self.log(f"loss/{mode}/wins_loss", wins_loss.mean())
        weighted_wins = wins_loss * x["wins_weights"]
        self.log(f"loss/{mode}/wins_loss_weighted", weighted_wins.mean())
        loss = (
            weighted_pick.mean()
            + weighted_wins.mean()
            + weighted_maindeck_loss_value.mean()
        )
        self.log(f"loss/{mode}/total", loss)

        self.metrics[f"{mode}"](pick_preds, x["picks"])
        self.log(f"acc/{mode}", self.metrics[mode], on_epoch=True, on_step=True)

        if mode == "train":
            lrs = self.lr_schedulers()
            self.log(f"lr", lrs.get_last_lr()[0])  # type: ignore

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="val")

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(
            params=self.parameters(), lr=1e-6, momentum=0.8, weight_decay=1e-6
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-2,
            total_steps=self.n_steps_oclr,
            pct_start=0.2,
            div_factor=100,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def collate_batch(
    inputs: T.Sequence[
        T.Dict[str, T.Union[T.List[T.List[int]], T.List[int], torch.Tensor]]
    ],
    n_cards_in_pack: int = N_CARDS_IN_PACK,
    device="cpu",
    inference=False,
):
    output: T.Dict[str, torch.Tensor] = {}
    histories = torch.zeros(
        (len(inputs), (n_cards_in_pack - 1) * 3, n_cards_in_pack),
        dtype=torch.int,
        device=device,
    )
    pools = torch.zeros(
        (len(inputs), 3 * (n_cards_in_pack - 1)), dtype=torch.int, device=device
    )
    pick_options = torch.zeros(
        (len(inputs), n_cards_in_pack), dtype=torch.int, device=device
    )
    picks = torch.zeros((len(inputs)), dtype=torch.long, device=device)
    maindeck_rates = -1 * torch.ones(
        (len(inputs), 3 * (n_cards_in_pack - 1)), dtype=torch.float, device=device
    )
    maindeck_weights: T.List[float] = []
    for n, x in enumerate(inputs):
        history_n = x["history"]
        for m, history_round in enumerate(history_n):
            try:
                histories[n, m, : len(history_round)] = torch.tensor(history_round)  # type: ignore
            except Exception as e:
                print(history_round)
                print(inputs[n])
                raise e
        pools[n, : len(x["pool"])] = torch.tensor(x["pool"])  # type: ignore
        pick_options[n, : len(x["options"])] = torch.tensor(x["options"])  # type: ignore
        if not inference:
            try:
                picks[n] = x["options"].index(x["picked"])  # type: ignore
                maindeck_rates[n, : len(x["pool"])] = torch.tensor(x["maindeck_rates"][: len(x["pool"])], dtype=torch.float, device=device)  # type: ignore
                maindeck_weights.extend(
                    [T.cast(float, x["pick_weight"]) for _ in range(len(x["pool"]))]
                )
            except Exception as e:
                print(picks[n])
                print(x["draft"])
                raise (e)

    output["history"] = histories
    output["pool"] = pools
    output["pick_options"] = pick_options
    if not inference:
        output["picks"] = picks
        output["round"] = torch.tensor([x["round"] for x in inputs], dtype=torch.int, device=device)  # type: ignore
        output["maindeck_rates"] = maindeck_rates
        output["maindeck_weights"] = torch.tensor(
            maindeck_weights, dtype=torch.float, device=device
        )

        if "wins_weight" in inputs[0]:
            win_weights = torch.tensor([x["wins_weight"] for x in inputs], dtype=torch.float, device=device)  # type: ignore
            pick_weights = torch.tensor([x["pick_weight"] for x in inputs], dtype=torch.float, device=device)  # type: ignore
            output["wins_weights"] = win_weights
            output["pick_weights"] = pick_weights
        if "num_wins" in inputs[0]:
            wins = torch.tensor(
                [x["num_wins"] for x in inputs], dtype=torch.float, device=device
            )
            output["num_wins"] = wins

    return output
