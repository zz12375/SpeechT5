#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Union, List, Dict
from fairseq.data.dictionary import Dictionary

from examples.speech_recognition.new.decoders.decoder_config import DecoderConfig, FlashlightDecoderConfig
from examples.speech_recognition.new.decoders.base_decoder import BaseDecoder


class ViterbiDecoder(BaseDecoder):
    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        def get_pred(e):
            # toks = e.argmax(dim=-1).unique_consecutive()
            # return toks[toks != self.blank]
            toks = e.argmax(dim=-1)
            toks[toks == self.blank] = self.tgt_dict.unk()
            return toks
        return [[{"tokens": get_pred(x), "score": 0}] for x in emissions]


def Decoder(
    cfg: Union[DecoderConfig, FlashlightDecoderConfig], tgt_dict: Dictionary
) -> BaseDecoder:

    if cfg.type == "viterbi":
        return ViterbiDecoder(tgt_dict)
    raise NotImplementedError(f"Invalid decoder name: {cfg.name}")
