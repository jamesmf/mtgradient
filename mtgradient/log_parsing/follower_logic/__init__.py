import requests
import json
from typing import List, Sequence, Union, Optional, Dict, Any, Tuple

import pandas as pd

from .mtga_follower import (
    Follower,
    API_ENDPOINT,
    get_config,
    JSON_START_REGEX,
    extract_time,
    json_value_matches,
    get_rank_string,
    logger,
)


ALSATrackerType = Dict[str, Dict[str, List[float]]]
ColorTrackerType = Dict[str, Dict[str, List[float]]]


class DashFollower(Follower):
    """
    Inherit from Follower but remove most of the functionality in
    favor of just keeping track of current draft status
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pack_number = ""
        self.pick_number = ""

    def picked_card(self, json_obj: Dict[str, Any], mode: str):
        if not hasattr(self, "pool_card_ids"):
            self.pool_card_ids = []
        # fail if not in (bot, combined) since I'm unfamiliar with
        # if there's another option
        if mode == "bot":
            card_id = int(json_obj["CardId"])
        if mode == "combined":
            card_id = int(json_obj["PickGrpId"])
        self.pool_card_ids.append(card_id)

    def got_pack(self, json_obj: Dict[str, Any], mode: str):
        if mode == "bot":
            picks = [int(x) for x in json_obj["DraftPack"]]
            pack_num_key = "PackNumber"
            pick_num_key = "PickNumber"
        elif mode == "human":
            picks = [int(x) for x in json_obj["PackCards"].split(",")]
            pack_num_key = "SelfPack"
            pick_num_key = "SelfPick"
        elif mode == "combined":
            picks = json_obj["CardsInPack"]
            pack_num_key = "PackNumber"
            pick_num_key = "PickNumber"
        self.pick_options = picks
        self.pack_number = json_obj[pack_num_key]
        self.pick_number = json_obj[pick_num_key]

        # in an attempt not to rewrite the Follower, only
        # keep track of certain items if they're set by
        # the app before parsing
        if (
            hasattr(self, "alsa_tracker")
            and hasattr(self, "color_tracker")
            and hasattr(self, "df")
        ):
            self.update_lanes(self.alsa_tracker, self.color_tracker, self.df)

    def _Follower__retry_post(self, *args, **kwargs):
        """
        We don't want to mess anything up, so don't talk to the API at all
        """
        r = requests.Response()
        r.status_code = 200
        return r

    def _Follower__handle_bot_draft_pack(self, json_obj: Dict[str, Any], **kwargs):
        #         super()._Follower__handle_bot_draft_pack(json_obj)
        self.got_pack(json_obj, "bot", **kwargs)

    def _Follower__handle_bot_draft_pick(self, json_obj: Dict[str, Any]):
        #         super()._Follower__handle_bot_draft_pick(json_obj)
        self.picked_card(json_obj, "bot")

    def _Follower__handle_human_draft_pack(self, json_obj: Dict[str, Any], **kwargs):
        self.got_pack(json_obj, "human", **kwargs)

    def _Follower__handle_human_draft_combined(
        self, json_obj: Dict[str, Any], **kwargs
    ):
        self.got_pack(json_obj, "combined", **kwargs)
        self.picked_card(json_obj, "combined")

    def _Follower__handle_joined_pod(self, json_obj: Dict[Any, str]):
        self.pick_options = []
        self.pool_card_ids = []

    def _Follower__handle_blob(self, full_log):
        """Attempt to parse a complete log message and send the data if relevant."""
        match = JSON_START_REGEX.search(full_log)
        if not match:
            return
        try:
            json_obj, end = self.json_decoder.raw_decode(full_log, match.start())
        except json.JSONDecodeError as e:
            logger.debug(
                f"Ran into error {e} when parsing at {self.cur_log_time}. Data was: {full_log}"
            )
            return

        json_obj = self._Follower__extract_payload(json_obj)
        #         print(json_obj)
        if type(json_obj) != dict:
            return

        try:
            maybe_time = self._Follower__maybe_get_utc_timestamp(json_obj)
            if maybe_time is not None:
                self.last_utc_time = maybe_time
        except:
            pass

        if json_value_matches(
            "Client.Connected", ["params", "messageName"], json_obj
        ):  # Doesn't exist any more
            self._Follower__handle_login(json_obj)
        elif "Event_Join" in full_log and "EventName" in json_obj:
            self._Follower__handle_joined_pod(json_obj)
        elif "DraftStatus" in json_obj:
            #             print("draft status")
            self._Follower__handle_bot_draft_pack(json_obj)
        elif "BotDraft_DraftPick" in full_log and "PickInfo" in json_obj:
            #             print("draft pick")
            self._Follower__handle_bot_draft_pick(json_obj["PickInfo"])
        elif "LogBusinessEvents" in full_log and "PickGrpId" in json_obj:
            self._Follower__handle_human_draft_combined(json_obj)
        elif "Draft.Notify " in full_log and "method" not in json_obj:
            self._Follower__handle_human_draft_pack(json_obj)
        elif "authenticateResponse" in json_obj:
            self._Follower__update_screen_name(
                json_obj["authenticateResponse"]["screenName"]
            )

    def update_lanes(
        self,
        alsa_dict: ALSATrackerType,
        color_tracker: ColorTrackerType,
        data: pd.DataFrame,
    ):
        """
        Calculate the difference in ALSA for each card and
        keep track of it by color
        """
        # if the app hasn't parsed anyting from the logs, return
        if not hasattr(self, "pack_number"):
            return
        pack_num = self.pack_number
        pick_num = self.pick_number
        # don't read into info from pack 1
        if pick_num == 1:
            return
        pp_string = f"{pack_num}_{pick_num}"
        if pp_string in alsa_dict:
            return
        if not hasattr(self, "pick_options"):
            return
        if len(self.pick_options) == 0:
            return
        card_data = data.loc[self.pick_options, ["color", "alsa"]].copy()
        card_data["pack"] = pack_num
        card_data["pick"] = pick_num

        # hypothesis: high alsa_diff implies open lane
        card_data["alsa_diff"] = card_data.alsa.apply(lambda x: pick_num - x)

        alsa_dict[pp_string] = {}
        for color, alsa_diff in card_data[["color", "alsa_diff"]].values:
            # ignore colorless cards
            if not color or str(color) == "nan":
                continue
            for color_char in color:
                if color_char not in color_tracker:
                    color_tracker[color_char] = 0
                color_tracker[color_char] += 1
            # don't read into highly multicolor cards
            if len(color) > 2:
                continue
            if alsa_diff > self.late_alsa_diff_threshold:
                if len(color) == 1:
                    alsa_weight = 1
                else:
                    alsa_weight = self.multicolor_alsa_diff_factor
                for color_char in color:
                    if color_char not in alsa_dict[pp_string]:
                        alsa_dict[pp_string][color_char] = []
                    alsa_dict[pp_string][color_char].append(alsa_diff * alsa_weight)
