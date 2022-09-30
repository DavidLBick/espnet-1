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

er_config=conf/train_hubert_ll60k_conformer_continuous.yaml
inference_config=conf/decode_er.yaml
local_data_opts="--lowercase true --remove_punctuation true --remove_emo xxx_exc_fru_fea_sur_oth"
er_tag=iemocap_fold1_discrete_base  # discrete_iemocap_fold1_base

./er.sh \
    --lang en \
    --ngpu 1 \
    --token_type word\
    --feats_type raw\
    --max_wav_duration 30 \
    --inference_nj 8 \
    --inference_er_model valid.acc.ave_10best.pth\
    --er_config "${er_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --er_args "--use_wandb true --wandb_project multilabel-emorec --wandb_entity cmu-mlsp-emo --wandb_name ${er_tag}" \
    --er_tag ${er_tag} \
    --local_data_opts "${local_data_opts}" "$@"
