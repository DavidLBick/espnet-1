"""
This describes the model for discrete and dimensional emotion recognition.
"""
import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict, List, Optional, Tuple, Union

import torch
from sklearn.metrics import accuracy_score, f1_score
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder

# from espnet2.asr.decoder.seq_classification import SeqClassifier
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.er.criterion import CCCLoss
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetERModel(AbsESPnetModel):
    """Attention based Emotion Classification/Regression model"""

    def __init__(
        self,
        vocab_size: Union[int, List[int]],
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        extract_feats_in_collect_stats: bool = True,
        disc_weights: List[float] = [0.2, 0.2, 0.2, 0.2, 0.2],
        discrete_cts_weight: List[float] = [1.0, 1.0],
        cts_weights: list = [1.0, 1.0, 1.0],
    ):
        assert check_argument_types()

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.mode = decoder.decoder_style

        self.encoder = encoder
        self.decoder = decoder

        self.dc_weight = discrete_cts_weight
        self.ct_weight = cts_weights

        if "discrete" in self.mode:
            disc_weights = torch.tensor(disc_weights)
            assert (
                len(disc_weights) == self.vocab_size
            ), f"discrete weights should be of size {self.vocab_size}"
            self.criterion_att = torch.nn.CrossEntropyLoss(
                weight=disc_weights, label_smoothing=lsm_weight, ignore_index=ignore_id
            )
        if "continuous" in self.mode:
            self.criterion_ccc = CCCLoss()
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        emotion: torch.Tensor = None,
        emotion_lengths: torch.Tensor = None,
        emotion_cts: torch.Tensor = None,
        emotion_cts_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            emotion: (Batch, Length)
            emotion_lengths: (Batch,)
        """
        assert emotion_lengths.dim() == 1, emotion_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == emotion.shape[0]
            == emotion_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, emotion.shape, emotion_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        emotion = emotion[:, : emotion_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        loss_att, acc_att, f1_att = None, None, None
        ccc_v, ccc_a, ccc_d, ccc = None, None, None, None
        stats = dict()

        # 2b. Attention decoder branch
        loss_att, acc_att, f1_att, loss_ccc, ccc = self._calc_att_loss(
            encoder_out,
            encoder_out_lens,
            emotion,
            emotion_cts,
        )
        ccc_v, ccc_a, ccc_d = (
            ccc if ccc is not None and len(ccc) == 3 else (None, None, None)
        )
        if ccc is not None:
            ccc = float(sum(ccc)) / len(ccc)
        loss = 0.0
        if loss_att is not None:
            loss += self.dc_weight[0] * loss_att
        if loss_ccc is not None:
            loss += self.dc_weight[1] * loss_ccc[0]

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["loss_ccc"] = loss_ccc.detach() if loss_ccc is not None else None
        stats["acc"] = acc_att
        stats["f1"] = f1_att
        stats["ccc_v"] = ccc_v
        stats["ccc_a"] = ccc_a
        stats["ccc_d"] = ccc_d
        stats["ccc"] = ccc
        stats["loss"] = loss.detach()

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        emotion: torch.Tensor = None,
        emotion_lengths: torch.Tensor = None,
        emotion_cts: torch.Tensor = None,
        emotion_cts_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)

        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        emotion: torch.Tensor = None,
        emotion_cts: torch.Tensor = None,
    ):

        # 1. Forward decoder
        cts_out, discrete_out = self.decoder(
            encoder_out, encoder_out_lens, emotion, emotion_cts
        )

        # 2. Compute attention loss
        loss_att = None
        acc = 0
        f1 = 0
        if "discrete" in self.mode:
            emotion = emotion.squeeze(-1)  # [B, 1] -> [B,]
            discrete_out = discrete_out.squeeze(1) # [7,1,5] -> [7,5]
            loss_att = self.criterion_att(discrete_out, emotion)
            acc = accuracy_score(
                torch.argmax(discrete_out, dim=-1).detach().cpu().numpy(),
                emotion.view(-1).cpu().numpy(),
            )
            f1 = f1_score(
                torch.argmax(discrete_out, dim=-1).detach().cpu().numpy(),
                emotion.view(-1).cpu().numpy(),
                average="macro",
            )

        loss_ccc = None
        ccc = None
        if "continuous" in self.mode:
            ccc = []
            loss_ccc = 0
            for i in range(emotion_cts.shape[-1]):
                loss, ccc_x = self.criterion_ccc(cts_out[:, i], emotion_cts[:, i])
                loss_ccc += self.ct_weight[i] * loss
                ccc.append(ccc_x)

        return loss_att, acc, f1, loss_ccc, ccc
