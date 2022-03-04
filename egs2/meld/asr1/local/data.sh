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
datadir=/home/dbick/workhorse1/data/meld
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
  #TODO
  # utt2spk
  # spk2utt
  # wav.scp  will have to use a sox comand or something to convert mp4 to wav 
  # text 
  # segments 
  tmp=tmp
  # if hasn't been already 
  if [ ! -e data/${tmp}/tmp.done ]; then
    # create directories and files for Kaldi-style data prep
    rm -r data/{train,valid,test}
    mkdir -p data/{train,valid,test}
    
    python local/create_speaker_id.py --fpath "${datadir}/${sub_dataset}/train_sent_emo.csv"
    # loop through datadir and collect all the needed information 
    data_dir=(train valid test)
    label_files=(train_sent_emo.csv dev_sent_emo.csv test_sent_emo.csv)
    audio_dirs=(train_splits dev_splits_complete output_repeated_splits_test)
    for i in ${!label_files[@]}; do
      label_file=${label_files[$i]}
      audio_dir=${audio_dirs[$i]}
      out_dir=${data_dir[$i]}
      
      python local/parse_csv.py --label-file "${datadir}/${sub_dataset}/${label_file}" --audio-dir "${datadir}/${sub_dataset}/${audio_dir}" --wav-scp "data/${out_dir}/wav.scp" --utt2spk "data/${out_dir}/utt2spk" --text "data/${out_dir}/text"

      utils/utt2spk_to_spk2utt.pl data/${out_dir}/utt2spk > "data/${out_dir}/spk2utt"

    done

    # write the needed info to appropriate place: text, wav.scp, utt2spk, and segments

    # TODO dos2unix command not found, is it necessary  
    # dos2unix data/${out_dir}/wav.scp; dos2unix data/${out_dir}/utt2spk; dos2unix data/${out_dir}/text
    # TODO add this back in when done
    #touch data/${out_dir}/tmp.done
  fi
fi

for dset in test valid train; do 
  utils/fix_data_dir.sh data/${dset}
  utils/validate_data_dir.sh --no-feats data/${dset} || exit 1
done 

log "Successfully finished. [elapsed=${SECONDS}s]"


















