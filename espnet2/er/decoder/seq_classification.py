"""
This file defines a sequence classification decoder that performs temporal 
sequence pooling for utterance classification

"""


import logging

import numpy as np
import torch

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class SelfAttentionPooling(torch.nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = torch.nn.Linear(input_dim, 1)
        self.softmax = torch.nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
        N: batch size, T: sequence length, H: Hidden dimension
        input:
            batch_rep : size (N, T, H)
        attention_weight:
            att_w : size (N, T, 1)
        return:
            utter_rep: size (N, H)
        """

        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w


class MTLDecoder(AbsDecoder):
    """
    Multitask decoder for joint emotion classification and continuous prediction
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        pool_type: str = "att",
        dropout_rate: float = 0.1,
        decoder_style: str = "discrete",
        discrete_pool_style: str = "independent",
        continuous_pool_style: str = "independent",
        discrete_continuous_pool_style: str = "independent",
        continuous_dim_size: int = 3,
    ):
        super().__init__()
        self.pool_type = pool_type
        self.discrete_pool_style = discrete_pool_style
        self.continuous_pool_style = continuous_pool_style
        self.discrete_continuous_pool_style = discrete_continuous_pool_style
        self.decoder_style = decoder_style
        self.encoder_output_size = encoder_output_size
        self.continuous_dim_size = continuous_dim_size

        if "discrete" in decoder_style:
            if pool_type == "att":
                self.disc_sap = (
                    SelfAttentionPooling(encoder_output_size)
                    if discrete_pool_style == "joint"
                    else torch.nn.ModuleList(
                        [
                            SelfAttentionPooling(encoder_output_size)
                            for _ in range(vocab_size)
                        ]
                    )
                )
                self.disc_processor = torch.nn.Sequential(
                    torch.nn.Linear(encoder_output_size, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(128, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(32, vocab_size),
                )

        if "continuous" in decoder_style:
            if pool_type == "att":
                self.cont_sap = (
                    SelfAttentionPooling(encoder_output_size)
                    if continuous_pool_style == "joint"
                    else torch.nn.ModuleList(
                        [
                            SelfAttentionPooling(encoder_output_size)
                            for _ in range(continuous_dim_size)
                        ]
                    )
                )
                self.cont_processor = torch.nn.Sequential(
                    torch.nn.Linear(encoder_output_size, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(128, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(32, continuous_dim_size)
                    if continuous_pool_style == "joint"
                    else torch.nn.Linear(32, 1),
                )

        if (
            "discrete" in decoder_style
            and "continuous" in decoder_style
            and discrete_continuous_pool_style == "joint"
        ):
            assert (
                discrete_pool_style == "joint" and continuous_pool_style == "joint"
            ), "discrete_continuous_pool_style must be joint if discrete_pool_style and continuous_pool_style are joint"
            self.cont_sap = self.disc_sap

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.disc_attn = None
        self.cts_attn = None

    def pool(self, hs_pad, hlens, pool_type, self_att_layer):
        """
        Perform temporal sequence pooling- Self-attention pooling,max or mean pooling
        """
        if pool_type == "att":
            hs_pad_mask = (
                (~make_pad_mask(hlens, maxlen=hs_pad.size(1)))[:, None, :]
                .to(hs_pad.device)
                .squeeze(1)
            )
            pooled, att = self_att_layer(hs_pad, hs_pad_mask)
            pooled = pooled.unsqueeze(1)

        elif pool_type == "mean":
            pooled, att = hs_pad.mean(dim=1).unsqueeze(1), None

        elif pool_type == "max":
            pooled, att = hs_pad.max(dim=1).unsqueeze(1), None

        return pooled, att

    def forward(self, hs_pad, hlens, emotion=None, emotion_cts=None):
        """
        Args:
            hs_pad: (batch, time, dim)  encoder hidden states
            hlens: (batch, )  encoder hidden states lengths
            emotion: (batch, )  emotion labels
            emotion_cts: (batch,K )  emotion continuous values, where K is the dimensionality of the emotion space
        """
        disc_logits = None  # discrete emotion logits
        cont_logits = None  # continuous emotion logits
        if "discrete" in self.decoder_style:
            if isinstance(self.disc_sap, torch.nn.ModuleList):
                if emotion is not None:
                    y_vals_unique = np.unique(emotion.cpu().numpy())
                    pooled = torch.zeros(
                        (hs_pad.shape[0], self.encoder_output_size)
                    ).to(hs_pad.device)
                    for y_val in y_vals_unique:
                        indices = (
                            torch.from_numpy(
                                np.array(
                                    [i for i, x in enumerate(emotion) if x == y_val]
                                )
                            )
                            .long()
                            .to(hs_pad.device)
                        )
                        inps = (
                            torch.index_select(hs_pad, index=indices, dim=0)
                            if len(indices) > 1
                            else hs_pad
                        )
                        lens = (
                            torch.index_select(input=hlens, index=indices, dim=0)
                            if len(indices) > 1
                            else hlens
                        )
                        assert y_val < len(
                            self.disc_sap
                        ), f"y_val {y_val} is out of range of length of self.disc_sap {len(self.disc_sap)}"
                        out, att = self.pool(
                            inps, lens, self.pool_type, self.disc_sap[y_val]
                        )
                        for i, ind in enumerate(indices):
                            pooled[ind] = out[i].squeeze(1)
                else:
                    # Get 5 sets of logits - l1,l2,l3,l4,l5: then return (5,n_classes) -> argmax -> get 5 possible outputs - score all pairs of hyp,ref to get max score ?? / Voting ??
                    # else condition output should be [1,embedding_dim] for each SAP layer- we have 5 so result is [5,embedding_dim]
                    pass
            else:
                pooled, att = self.pool(hs_pad, hlens, self.pool_type, self.disc_sap)

            # Dropout and Activation
            pooled = self.dropout(torch.nn.functional.relu(pooled))
            disc_logits = self.disc_processor(pooled)

        if "continuous" in self.decoder_style:
            if isinstance(self.cont_sap, torch.nn.ModuleList):
                out = []
                for i in range(self.continuous_dim_size):
                    pooled, att = self.pool(
                        hs_pad, hlens, self.pool_type, self.cont_sap[i]
                    )
                    # Dropout and Activation
                    pooled = self.dropout(torch.nn.functional.relu(pooled))
                    cont_logits = self.cont_processor(pooled)
                    out.append(cont_logits)
                cont_logits = torch.nn.functional.relu(torch.cat(out, dim=1))
            else:
                pooled, att = self.pool(hs_pad, hlens, self.pool_type, self.cont_sap)
                # Dropout and Activation
                pooled = self.dropout(torch.nn.functional.relu(pooled))
                cont_logits = torch.nn.functional.relu(self.cont_processor(pooled))

        return cont_logits, disc_logits


class HMTLDecoderCD(MTLDecoder, AbsDecoder):
    """
    Hierarchical Multitask decoder for joint emotion classification and continuous prediction
    Uses Predicted Continuous Emotion as Input to Discrete Emotion Classifier
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        pool_type: str = "mean",
        dropout_rate: float = 0.1,
        decoder_style: str = "discrete,continuous",
        discrete_pool_style: str = "independent",
        continuous_pool_style: str = "independent",
        discrete_continuous_pool_style: str = "independent",
        continuous_dim_size: int = 3,
        continuous_embedding_dim: int = 32,
        discrete_embedding_dim: int = 32,
    ):
        super().__init__(vocab_size=vocab_size, 
            encoder_output_size=encoder_output_size,
            pool_type=pool_type, 
            dropout_rate=dropout_rate, 
            decoder_style=decoder_style, 
            discrete_pool_style=discrete_pool_style,
            continuous_pool_style=continuous_pool_style, 
            discrete_continuous_pool_style=discrete_continuous_pool_style,
            continuous_dim_size=continuous_dim_size,
        )
        self.pool_type = pool_type
        self.discrete_pool_style = discrete_pool_style
        self.continuous_pool_style = continuous_pool_style
        self.discrete_continuous_pool_style = discrete_continuous_pool_style
        self.decoder_style = decoder_style
        self.encoder_output_size = encoder_output_size
        self.continuous_dim_size = continuous_dim_size

        if pool_type == "att":
            self.cont_sap = (
                SelfAttentionPooling(encoder_output_size)
                if continuous_pool_style == "joint"
                else torch.nn.ModuleList(
                    [
                        SelfAttentionPooling(encoder_output_size)
                        for _ in range(continuous_dim_size)
                    ]
                )
            )
        self.cont_processor = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, continuous_embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
        )
        self.cont_out = (
            torch.nn.Linear(continuous_embedding_dim, continuous_dim_size)
            if continuous_pool_style == "joint"
            else torch.nn.Linear(32, 1)
        )

        if pool_type == "att":
            self.disc_sap = (
                SelfAttentionPooling(encoder_output_size)
                if discrete_pool_style == "joint"
                else torch.nn.ModuleList(
                    [
                        SelfAttentionPooling(encoder_output_size)
                        for _ in range(vocab_size)
                    ]
                )
            )
            self.disc_processor = torch.nn.Sequential(
                torch.nn.Linear(encoder_output_size, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(128, discrete_embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
            )
            cont_embedding_size = (
                continuous_embedding_dim
                if continuous_pool_style == "joint"
                else continuous_embedding_dim * continuous_dim_size
            )
            self.disc_out = torch.nn.Linear(
                discrete_embedding_dim + cont_embedding_size, vocab_size
            )

        if (
            "discrete" in decoder_style
            and "continuous" in decoder_style
            and discrete_continuous_pool_style == "joint"
        ):
            assert (
                discrete_pool_style == "joint" and continuous_pool_style == "joint"
            ), "discrete_continuous_pool_style must be joint if discrete_pool_style and continuous_pool_style are joint"
            self.cont_sap = self.disc_sap

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.disc_attn = None
        self.cts_attn = None

    def forward(self, hs_pad, hlens, emotion=None, emotion_cts=None):
        """
        Args:
            hs_pad: (batch, time, dim)  encoder hidden states
            hlens: (batch, )  encoder hidden states lengths
            emotion: (batch, )  emotion labels
            emotion_cts: (batch,K )  emotion continuous values, where K is the dimensionality of the emotion space
        """
        disc_logits = None  # discrete emotion logits
        cont_logits = None  # continuous emotion logits

        if isinstance(self.cont_sap, torch.nn.ModuleList):
            out = []
            out_embedding = []
            for i in range(self.continuous_dim_size):
                pooled, att = self.pool(hs_pad, hlens, self.pool_type, self.cont_sap[i])
                # Dropout and Activation
                pooled = self.dropout(torch.nn.functional.relu(pooled))
                cont_embedding = self.cont_processor(pooled)
                cont_logits = self.cont_out(cont_embedding)
                out_embedding.append(cont_embedding)
                out.append(cont_logits)
            cont_logits = torch.nn.functional.relu(torch.cat(out, dim=-1))
            cont_embedding = torch.cat(out_embedding, dim=-1)
        else:
            pooled, att = self.pool(hs_pad, hlens, self.pool_type, self.cont_sap)
            # Dropout and Activation
            pooled = self.dropout(torch.nn.functional.relu(pooled))
            cont_embedding = self.cont_processor(pooled)
            cont_logits = torch.nn.functional.relu(self.cont_out(cont_embedding))

        if isinstance(self.disc_sap, torch.nn.ModuleList):
            y_vals_unique = np.unique(emotion.cpu().numpy())
            pooled = torch.zeros((hs_pad.shape[0], self.encoder_output_size)).to(
                hs_pad.device
            )
            for y_val in y_vals_unique:
                indices = (
                    torch.from_numpy(
                        np.array([i for i, x in enumerate(emotion) if x == y_val])
                    )
                    .long()
                    .to(hs_pad.device)
                )
                inps = (
                    torch.index_select(hs_pad, index=indices, dim=0)
                    if len(indices) > 1
                    else hs_pad
                )
                lens = (
                    torch.index_select(input=hlens, index=indices, dim=0)
                    if len(indices) > 1
                    else hlens
                )
                assert y_val < len(
                    self.disc_sap
                ), f"y_val {y_val} is out of range of length of self.disc_sap {len(self.disc_sap)}"
                out, att = self.pool(inps, lens, self.pool_type, self.disc_sap[y_val])
                for i, ind in enumerate(indices):
                    pooled[ind] = out[i].squeeze(1)
        else:
            pooled, att = self.pool(hs_pad, hlens, self.pool_type, self.disc_sap)

        # Dropout and Activation
        pooled = self.dropout(torch.nn.functional.relu(pooled))
        disc_embedding = self.disc_processor(pooled)
        disc_input = torch.cat([disc_embedding, cont_embedding], dim=-1)
        disc_logits = self.disc_out(disc_input)

        return cont_logits, disc_logits


class HMTLDecoderDC(AbsDecoder):
    """
    Hierarchical Multitask decoder for joint emotion classification and continuous prediction
    Uses Predicted Discrete Emotion Embedding as Input to Continuous Emotion Predictor
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        pool_type: str = "mean",
        dropout_rate: float = 0.1,
        decoder_style: str = "discrete,continuous",
        discrete_pool_style: str = "independent",
        continuous_pool_style: str = "independent",
        discrete_continuous_pool_style: str = "independent",
        continuous_dim_size: int = 3,
        continuous_embedding_dim: int = 32,
        discrete_embedding_dim: int = 32,
    ):
        super().__init__(vocab_size=vocab_size, 
            encoder_output_size=encoder_output_size,
            pool_type=pool_type, 
            dropout_rate=dropout_rate, 
            decoder_style=decoder_style, 
            discrete_pool_style=discrete_pool_style,
            continuous_pool_style=continuous_pool_style, 
            discrete_continuous_pool_style=discrete_continuous_pool_style,
            continuous_dim_size=continuous_dim_size,
        )
        self.pool_type = pool_type
        self.discrete_pool_style = discrete_pool_style
        self.continuous_pool_style = continuous_pool_style
        self.discrete_continuous_pool_style = discrete_continuous_pool_style
        self.decoder_style = decoder_style
        self.encoder_output_size = encoder_output_size
        self.continuous_dim_size = continuous_dim_size

        if "discrete" in decoder_style:
            if pool_type == "att":
                self.disc_sap = (
                    SelfAttentionPooling(encoder_output_size)
                    if discrete_pool_style == "joint"
                    else torch.nn.ModuleList(
                        [
                            SelfAttentionPooling(encoder_output_size)
                            for _ in range(vocab_size)
                        ]
                    )
                )
                self.disc_processor = torch.nn.Sequential(
                    torch.nn.Linear(encoder_output_size, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(128, discrete_embedding_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate),
                )
                self.disc_out = torch.nn.Linear(discrete_embedding_dim, vocab_size)

        if "continuous" in decoder_style:
            if pool_type == "att":
                self.cont_sap = (
                    SelfAttentionPooling(encoder_output_size)
                    if continuous_pool_style == "joint"
                    else torch.nn.ModuleList(
                        [
                            SelfAttentionPooling(encoder_output_size)
                            for _ in range(continuous_dim_size)
                        ]
                    )
                )
            self.cont_processor = torch.nn.Sequential(
                torch.nn.Linear(encoder_output_size, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(128, continuous_embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
            )
            self.cont_out = (
                torch.nn.Linear(
                    continuous_embedding_dim + discrete_embedding_dim,
                    continuous_dim_size,
                )
                if continuous_pool_style == "joint"
                else torch.nn.Linear(
                    continuous_embedding_dim + discrete_embedding_dim, 1
                )
            )

        if (
            "discrete" in decoder_style
            and "continuous" in decoder_style
            and discrete_continuous_pool_style == "joint"
        ):
            assert (
                discrete_pool_style == "joint" and continuous_pool_style == "joint"
            ), "discrete_continuous_pool_style must be joint if discrete_pool_style and continuous_pool_style are joint"
            self.cont_sap = self.disc_sap

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.disc_attn = None
        self.cts_attn = None

    def forward(self, hs_pad, hlens, emotion=None, emotion_cts=None):
        """
        Args:
            hs_pad: (batch, time, dim)  encoder hidden states
            hlens: (batch, )  encoder hidden states lengths
            emotion: (batch, )  emotion labels
            emotion_cts: (batch,K )  emotion continuous values, where K is the dimensionality of the emotion space
        """
        disc_logits = None  # discrete emotion logits
        cont_logits = None  # continuous emotion logits

        if isinstance(self.disc_sap, torch.nn.ModuleList):
            y_vals_unique = np.unique(emotion.cpu().numpy())
            pooled = torch.zeros((hs_pad.shape[0], self.encoder_output_size)).to(
                hs_pad.device
            )
            for y_val in y_vals_unique:
                indices = (
                    torch.from_numpy(
                        np.array([i for i, x in enumerate(emotion) if x == y_val])
                    )
                    .long()
                    .to(hs_pad.device)
                )
                inps = (
                    torch.index_select(hs_pad, index=indices, dim=0)
                    if len(indices) > 1
                    else hs_pad
                )
                lens = (
                    torch.index_select(input=hlens, index=indices, dim=0)
                    if len(indices) > 1
                    else hlens
                )
                assert y_val < len(
                    self.disc_sap
                ), f"y_val {y_val} is out of range of length of self.disc_sap {len(self.disc_sap)}"
                out, att = self.pool(inps, lens, self.pool_type, self.disc_sap[y_val])
                for i, ind in enumerate(indices):
                    pooled[ind] = out[i].squeeze(1)
        else:
            pooled, att = self.pool(hs_pad, hlens, self.pool_type, self.disc_sap)

        # Dropout and Activation
        pooled = self.dropout(torch.nn.functional.relu(pooled))
        disc_embedding = self.disc_processor(pooled)
        disc_logits = self.disc_out(disc_embedding)

        if isinstance(self.cont_sap, torch.nn.ModuleList):
            out = []
            for i in range(self.continuous_dim_size):
                pooled, att = self.pool(hs_pad, hlens, self.pool_type, self.cont_sap[i])
                # Dropout and Activation
                pooled = self.dropout(torch.nn.functional.relu(pooled))
                cont_embedding = self.cont_processor(pooled)
                cont_input = torch.cat((cont_embedding, disc_embedding), dim=1)
                cont_logits = self.cont_out(cont_input)
                out.append(cont_logits)
            cont_logits = torch.nn.functional.relu(torch.cat(out, dim=-1))
        else:
            pooled, att = self.pool(hs_pad, hlens, self.pool_type, self.cont_sap)
            # Dropout and Activation
            pooled = self.dropout(torch.nn.functional.relu(pooled))
            cont_embedding = self.cont_processor(pooled)
            cont_input = torch.cat((cont_embedding, disc_embedding), dim=1)
            cont_logits = torch.nn.functional.relu(self.cont_out(cont_input))

        return cont_logits, disc_logits
