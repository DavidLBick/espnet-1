#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


nj=5
train_set="train_fold0"${nj}
valid_set="valid_fold0"${nj}
test_sets="test_fold0"${nj}
test_sets=${test_sets}" "${valid_set}
er_config=conf/tuning/train_er_conformer_lmel_att.yaml
inference_config=conf/decode.yaml

if [[ $(nvidia-smi | grep MiB | wc -l) -gt 1 ]]; then
	cuda_device=$(( $nj -1 ))
else
	cuda_device=0
fi 
echo $nj $cuda_device $train_set $valid_set $test_sets

CUDA_VISIBLE_DEVICES=${cuda_device} ./er.sh \
    --lang en \
    --ngpu 1 \
    --stage 6 \
    --stop_stage 8 \
    --token_type word \
    --er_stats_dir exp/er_stats_raw_emo_sp_fold${nj} \
    --feats_type raw \
    --max_wav_duration 30 \
    --inference_nj 8 \
    --feats_normalize null \
    --inference_er_model valid.acc.ave_10best.pth\
    --er_tag sp_conformer_lmelfeat_attpool_fold${nj} \
    --er_args "--wandb_project emorec_iemocap --use_wandb true --wandb_name sp_conformer_lmelfeat_attpool_fold${nj}" \
    --er_config "${er_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"