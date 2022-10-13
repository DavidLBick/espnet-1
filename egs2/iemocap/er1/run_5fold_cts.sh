#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


nj=$1
train_set="train_fold0"${nj}
valid_set="valid_fold0"${nj}
test_sets="test_fold0"${nj}
test_sets=${test_sets}" "${valid_set}
er_config=conf/train_hubert_ll60k_conformer_continuous.yaml
inference_config=conf/decode_er.yaml

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
    --stop_stage 6 \
    --token_type word \
    --use_continuous true \
    --use_discrete true \
    --er_stats_dir exp/er_stats_raw_emo_fold${nj} \
    --feats_type raw \
    --nj 8 \
    --max_wav_duration 30 \
    --inference_nj 8 \
    --gpu_inference true \
    --feats_normalize null \
    --inference_er_model valid.acc.ave_10best.pth \
    --er_tag conformer_continuous_base_hubertlarge_fold${nj} \
    --er_args "--wandb_project multilabel-emorec --wandb_entity cmu-mlsp-emo --use_wandb true --wandb_name conformer_continuous_base_hubertlarge_fold${nj}" \
    --er_config "${er_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}"
