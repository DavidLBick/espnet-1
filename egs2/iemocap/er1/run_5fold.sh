#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


nj=$1
train_set="train_fold0"${nj}
test_sets="test_fold0"${nj}
valid_set="valid_fold0"${nj}

# if [[ ${nj} != "1" ]]; then 
#     valid_fold=$(( ${nj} -1 ))
# else 
#     valid_fold=5
# fi
# valid_set="valid_fold0"${valid_fold}

# "${valid_set}
# er_config=conf/train_hubert_ll60k_conformer_discrete.yaml
er_config=conf/train_hubert_ll60k_conformer_continuous.yaml
#conf/tuning/train_er_conformer_lmel_att.yaml
inference_config=conf/decode_er.yaml
er_tag=sp_conformer_lmelfeat_attpool_fold${nj}_continuous_joint
# er_tag=sp_conformer_lmelfeat_attpool_fold${nj}_dicrete_joint
# inference_er_model=valid.acc.ave_10best.pth
# inference_er_model=valid.ccc.ave_10best.pth
inference_er_model=valid.ccc.best.pth
use_discrete=false
use_continuous=true

if [[ $(nvidia-smi | grep MiB | wc -l) -gt 1 ]]; then
    # av=$(nvidia-smi | grep MiB | wc -l)
	cuda_device=$(( $nj -1 ))
    # cuda_device=$av
else
	cuda_device=0
fi 

# cuda_device=0,1,2,3
echo $nj $cuda_device $train_set $valid_set $test_sets

time CUDA_VISIBLE_DEVICES=${cuda_device} ./er.sh \
    --lang en \
    --ngpu 1 \
    --stage 8 \
    --stop_stage 8 \
    --token_type word \
    --er_stats_dir exp/er_stats_raw_emo_sp_fold${nj} \
    --feats_type raw \
    --max_wav_duration 30 \
    --inference_nj 8 \
    --feats_normalize null \
    --inference_er_model "${inference_er_model}"\
    --inference_config "${inference_config}"\
    --er_tag "${er_tag}" \
    --er_args "--wandb_project emorec_iemocap --use_wandb true --wandb_name sp_conformer_lmelfeat_attpool_fold${nj}" \
    --er_config "${er_config}" \
    --use_discrete "${use_discrete}" \
    --use_continuous "${use_continuous}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}"


# CUDA_VISIBLE_DEVICES=0 ./er.sh --lang en  --ngpu 1 --stage 8 --stop_stage 8  --token_type word --er_stats_dir exp/er_stats_raw_emo_sp_fold4 --feats_type raw --max_wav_duration 30 --inference_nj 8 --feats_normalize null --inference_er_model valid.acc.ave_10best.pth --er_tag sp_conformer_lmelfeat_attpool_fold4_dicrete_joint --er_args "--wandb_project emorec_iemocap --use_wandb true --wandb_name sp_conformer_lmelfeat_attpool_fold4" --er_config conf/train_hubert_ll60k_conformer_discrete.yaml --train_set train_fold04 --valid_set valid_fold03 --test_sets test_fold04

# CUDA_VISIBLE_DEVICES=0 ./er.sh --lang en  --ngpu 1 --stage 8 --stop_stage 8  --token_type word --er_stats_dir exp/er_stats_raw_emo_sp_fold5 --feats_type raw --max_wav_duration 30 --inference_nj 8 --feats_normalize null --inference_er_model valid.acc.ave_10best.pth --er_tag sp_conformer_lmelfeat_attpool_fold5_dicrete_joint --er_args "--wandb_project emorec_iemocap --use_wandb true --wandb_name sp_conformer_lmelfeat_attpool_fold5" --er_config conf/train_hubert_ll60k_conformer_discrete.yaml --train_set train_fold05 --valid_set valid_fold04 --test_sets test_fold05