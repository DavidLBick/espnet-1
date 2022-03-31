import torch
import torch.nn.functional as F
from typeguard import check_argument_types
import logging
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.nets_utils import to_device


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
            self.query = torch.nn.Embedding(encoder_output_size, attention_dim)
            self.value = torch.nn.Linear(encoder_output_size, attention_dim)
            self.key = torch.nn.Linear(encoder_output_size, attention_dim)
            self.mha = torch.nn.MultiheadAttention(attention_dim, attention_heads)

        self.output = torch.nn.Linear(encoder_output_size, vocab_size)

    def pool(self, hs_pad, hlens):
        if self.pool_type == "att":
            key, query, value = (
                self.key(hs_pad),
                self.query(hs_pad),
                self.value(hs_pad),
            )
            pooled, att = self.multihead_att(key, query, value)

        elif self.pool_type == "mean":
            pooled, att = hs_pad.mean(dim=1).unsqueeze(1), None

        elif self.pool_type == "max":
            pooled, att = hs_pad.max(dim=1).unsqueeze(1), None

        return pooled, att

    def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens):
        pooled, att = self.pool(hs_pad, hlens)
        return self.output(pooled), ys_in_lens

