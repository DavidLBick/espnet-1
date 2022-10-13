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
er_config=conf/train_hubert_ll60k_conformer_discrete.yaml
inference_config=conf/decode.yaml
inference_er_model=valid.acc.ave_10best.pth
use_discrete=true
use_continuous=true

if [[ $(nvidia-smi | grep MiB | wc -l) -gt 1 ]]; then
	cuda_device=$(( $nj -1 ))
else
	cuda_device=0
fi 

echo $nj $cuda_device $train_set $valid_set $test_sets

time CUDA_VISIBLE_DEVICES=${cuda_device} ./er.sh \
    --lang en \
    --ngpu 1 \
    --stage 7 \
    --stop_stage 9 \
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
    --er_tag conformer_discrete_base_hubertlarge_fold${nj} \
    --er_args "--wandb_project emorec_iemocap --use_wandb true --wandb_name conformer_discrete_base_hubertlarge_fold${nj}" \
    --er_config "${er_config}" \
    --use_discrete "${use_discrete}" \
    --use_continuous "${use_continuous}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}"


