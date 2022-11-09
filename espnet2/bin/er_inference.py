#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.er import ERTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.utils.cli_utils import get_commandline_args


class Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("er_config.yml", "valid.ccc.best.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        er_train_config: Union[Path, str] = None,
        er_model_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        batch_size: int = 1,
        dtype: str = "float32",
        nbest: int = 1,
    ):
        assert check_argument_types()

        # 1. Build ER model
        er_model, er_train_args = ERTask.build_model_from_file(
            er_train_config, er_model_file, device
        )
        er_model.to(dtype=getattr(torch, dtype)).eval()

        self.decoder = er_model.decoder

        token_list = er_model.token_list

        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = er_train_args.token_type
        if bpemodel is None:
            bpemodel = er_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.er_model = er_model
        self.er_train_args = er_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> List[
        Union[
            Optional[str],
            Optional[List[str]],
            Optional[List[int]],
            Optional[List[float]],
            Optional[str],
        ]
    ]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, cont_emotion, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        ## Channel Selection
        if speech.shape[-1] == 2 and len(speech.shape) > 1:
            speech = speech[:, :, 0]

        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_lens = self.er_model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]

        # c. Passed the encoder result and the beam search

        cont_out, disc_out = self.decoder(enc, enc_lens)
        if disc_out is not None:
            ## if disc_out.shape[0] != 1; then we have mutiple hypotheses
            disc_out = disc_out.squeeze(1)
            tok_vals, tok_inds = torch.topk(disc_out, 1, dim=-1)
            token_int = tok_inds[:, 0].cpu().numpy().tolist()
            token_vals, token_inds = torch.topk(disc_out, 4, dim=-1)
            score = " ".join(
                [
                    f"{token_vals[:,i].cpu().numpy().tolist()[0]},{token_inds[:,i].cpu().numpy().tolist()[0]}"
                    for i in range(4)
                ]
            )

            score = tok_vals[:, 0].cpu().numpy().tolist()
            token = self.converter.ids2tokens(token_int)[0]
            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
        else:
            token = None
            text = None
            token_int = None
            score = None
        if cont_out is not None:
            emo_out = cont_out.view(-1).cpu().numpy().tolist()
        else:
            emo_out = None
        results_out = [text, token, token_int, score, emo_out]

        assert check_return_type(results_out)
        return [results_out]

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Speech2Text instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: Speech2Text instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2Text(**kwargs)


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    er_train_config: Optional[str],
    er_model_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
):
    assert check_argument_types()
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        er_train_config=er_train_config,
        er_model_file=er_model_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        dtype=dtype,
        nbest=nbest,
    )
    speech2text = Speech2Text.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = ERTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ERTask.build_preprocess_fn(speech2text.er_train_args, False),
        collate_fn=ERTask.build_collate_fn(speech2text.er_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 7 .Start for-loop
    # TODO(roshansh): Add batch decoding compat
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            results = speech2text(**batch)

            # Only supporting batch_size==1

            for key in keys:
                # key = keys[0]
                for n, (text, token, token_int, score, emo_out) in zip(
                    range(1, nbest + 1), results
                ):
                    # Create a directory: outdir/{n}best_recog
                    ibest_writer = writer[f"{n}best_recog"]
                    # Write the result to each file
                    if emo_out is not None:
                        ibest_writer["emotion_cts"][key] = " ".join(
                            [str(x) for x in emo_out]
                        )
                    if text is not None:
                        ibest_writer["text"][key] = ("").join(text).replace(" ", "")
                        ibest_writer["token"][key] = token
                        ibest_writer["token_int"][key] = str(token_int)
                        ibest_writer["score"][key] = str(score[0])


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ER Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--er_train_config",
        type=str,
        help="ER training configuration",
    )
    group.add_argument(
        "--er_model_file",
        type=str,
        help="ER model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ER model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
