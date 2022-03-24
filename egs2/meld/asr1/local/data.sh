#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on : 
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e 
set -u
set -o pipefail 

log() {
     local fname=${BASH_SOURCE[1]##*/}
     echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


SECONDS=0
# TODO can I just remove this somehow
stage=1
stop_stage=100
# TODO replace this with PSC after done testing 
#datadir=/ocean/projects/cis220008p/shared/meld
# temporary to test on workhorse
#datadir=/home/dbick/workhorse1/data/meld
datadir=$PROJECT/corpora/meld
# TODO add switch for this vs FriendsPersona
sub_dataset=MELD.Raw
# sub_dataset=FriendsPersona
# meld/ 
#  |_ MELD.Raw/
#      |_ {train,test,dev}.tar.gz
#         |_ dia{dialogue_ID}_utt{utterance_ID}.mp4 <audio files for each utterance and dialogue combo (can have multiple utterances in one dialogue>
#         |_ {train,test,dev}_sent_emo.csv <contains emotion/sentiment labels, and dialogue_ID, utterance_ID, season, episode, speaker>
#  |_ personality_detection/
#      |_ CSV/
#          |_ friends-personality.csv
# MELD data: https://affective-meld.github.io/
# FriendsPersona data: https://github.com/emorynlp/personality-detection

log "$0 $*"
. utils/parse_options.sh # DOUBT not sure if this is always necessary 

. ./path.sh  # DOUBT seems to just update some paths to allow access to Kaldi 
. ./cmd.sh   # DOUBT seems that it sets run.pl to be the default file when backend is local, similar for slurm.pl if slurm is backend etc

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "stage 1: MELD Data Preparation"
  mkdir -p data/{train,dev,test}
    
  python local/create_speaker_id.py --fpath "${datadir}/${sub_dataset}/train_sent_emo.csv"
  
  for dset in train dev test; do
      echo "DSET ${dset}"
      audio_dir=$( ls ${datadir}/${sub_dataset}/ | grep ${dset} | grep splits )
      echo "DSET ${dset} ${audio_dir}"
      python local/parse_csv.py --label_file "${datadir}/${sub_dataset}/${dset}_sent_emo.csv" \
	      --audio_dir "${datadir}/${sub_dataset}/${audio_dir}" \
	      --data_dir "data/${dset}" 

      utils/utt2spk_to_spk2utt.pl data/${dset}/utt2spk > "data/${dset}/spk2utt"
      ## One issue in training 
      if  [[ "${dset}" == "train" ]]; then 
	      grep -v "dia125_utt3.mp4" data/${dset}/wav.scp > tmp && mv tmp data/${dset}/wav.scp
      elif [[ "${dset}" == "dev" ]]; then
	      grep -v "dia110_utt7.mp" data/${dset}/wav.scp > tmp && mv tmp data/${dset}/wav.scp
      fi
      utils/fix_data_dir.sh data/${dset}
      utils/validate_data_dir.sh --no-feats data/${dset} || exit 1

  done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
