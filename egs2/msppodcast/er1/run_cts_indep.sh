#! /bin/bash


#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test1 valid"

er_config=conf/extracted/train_hubert_ll60k_conformer_continuous_indep.yaml
inference_config=conf/decode_er.yaml
local_data_opts=""
er_tag=msppodcast_continuous_indep_pool_base  # discrete_iemocap_fold1_base

./er.sh \
    --lang en \
    --ngpu 1 \
    --token_type word \
    --feats_type extracted \
    --max_wav_duration 30 \
    --inference_nj 5 \
    --feats_normalize null \
    --er_stats_dir exp/er_stats_msp_hubert \
    --use_continuous true \
    --use_discrete false \
    --inference_er_model valid.ccc.ave_10best.pth\
    --er_config "${er_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --er_args "--use_wandb true --wandb_project multilabel-emorec --wandb_entity cmu-mlsp-emo --wandb_name ${er_tag}" \
    --er_tag ${er_tag} \
    --local_data_opts "${local_data_opts}" "$@"
