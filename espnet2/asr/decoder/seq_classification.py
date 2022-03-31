import torch
import torch.nn.functional as F
from typeguard import check_argument_types
import logging
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, x, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                x : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(x).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(x * att_w, dim=1)
        return utter_rep, att_w


class SeqClassifier(AbsDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        pool_type: str = "mean",
        attention_heads: int = 4,
        attention_dim: int = 512,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.pool_type = pool_type
        if pool_type == "att":
            self.sap = SelfAttentionPooling(encoder_output_size)

        self.output = torch.nn.Linear(encoder_output_size, vocab_size)
        self.attn = None

    def pool(self, hs_pad, hlens):
        if self.pool_type == "att":
            hs_pad_mask = (~make_pad_mask(hlens, maxlen=hs_pad.size(1)))[:, None, :].to(
                hs_pad.device
            )
            pooled, att = self.sap(hs_pad, hs_pad_mask)

        elif self.pool_type == "mean":
            pooled, att = hs_pad.mean(dim=1).unsqueeze(1), None

        elif self.pool_type == "max":
            pooled, att = hs_pad.max(dim=1).unsqueeze(1), None

        return pooled, att

    def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens):
        pooled, att = self.pool(hs_pad, hlens)
        self.attn = att
        return self.output(pooled), ys_in_lens
    
