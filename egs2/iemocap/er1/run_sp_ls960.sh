#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test valid"

asr_config=conf/tuning/finetune_hubert_960h.yaml
inference_config=conf/decode.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --nbpe 5 \
    --token_type word \
    --use-lm false \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --disable_add_sos_eos true \
    --asr_stats_dir exp/asr_stats_raw_emo_sp \
    --max_wav_duration 30 \
    --inference_nj 8 \
    --inference_asr_model valid.acc.ave_10best.pth \
    --asr_tag "sp_ls960_ft" \
    --asr_args "--wandb_project emorec_iemocap --use_wandb true --wandb_name sp_ls960_ft" \
    --asr_config "${asr_config}" \
    --use_classify true \
    --feats_normalize null \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    "$@"
