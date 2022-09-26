#! /bin/bash


#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_fold01"
valid_set="valid_fold01"
test_sets="test_fold01 valid_fold01"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml
local_data_opts="--lowercase true --remove_punctuation true --remove_emo xxx_exc_fru_fea_sur"

./er.sh \
    --lang en \
    --ngpu 1 \
    --token_type word\
    --feats_type raw\
    --max_wav_duration 30 \
    --inference_nj 8 \
    --inference_er_model valid.acc.ave_10best.pth\
    --er_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --local_data_opts "${local_data_opts}" "$@"
