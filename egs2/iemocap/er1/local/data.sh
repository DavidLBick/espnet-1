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
stage=1
stop_stage=2

lowercase=false
# Convert into lowercase if "true".
remove_punctuation=false
# Remove punctuation (except apostrophes) if "true".
# Note that punctuation normalization will be performed in the "false" case. 
remove_tag=false
# Remove [TAGS] (e.g.[LAUGHTER]) if "true".
remove_emo=
# Remove the utterances with the specified emotional labels
# emotional labels: ang (anger), hap (happiness), exc (excitement), sad (sadness),
# fru (frustration), fea (fear), sur (surprise), neu (neutral), and xxx (other)

#data
datadir=/ocean/projects/iri120008p/roshansh/corpora/IEMOCAP_full_release/IEMOCAP_full_release
# IEMOCAP_full_release
#  |_ README.txt
#  |_ Documentation/
#  |_ Session{1-5}/
#      |_ sentences/wav/ ...<wav files for each utterance>
#      |_ dialog/
#          |_ transcriptions/ ...<transcription files>
#          |_ EmoEvaluation/ ...<emotion annotation files>
# In this recipe
# Sessions 1-3 & 4F (Ses01, SeS02, Ses03,and Ses04F) are used for training (6871 utterances),
# Session 4M (Ses04M) is used for validation (998 utterances), and
# Session 5 (Ses05) is used for evaluation (2170 utterances).
# Download data from here:
# https://sail.usc.edu/iemocap/

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


mkdir -p data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: IEMOCAP Data Preparation"
    # This process may take a few minutes for the first run
    # Remove "data/all/tmp.done" if you want to start all over again
    if [ -n "${remove_emo}" ]; then
        log "Remove ${remove_emo} from emotional labels"
        tmp="tmp/remove_emo"
    else
        log "Use all 9 emotional labels"
        tmp=tmp
    fi
    if [ ! -e data/all/tmp.done ];then
        mkdir -p data/{train,valid,test,all}
        echo -n "" > data/all/wav.scp; echo -n "" > data/all/utt2spk; echo -n "" > data/all/text
        for n in {1..5}; do
            for file in "${datadir}"/Session"${n}"/sentences/wav/*/*.wav; do
                utt_id=$(basename ${file} .wav)
                ses_id=$(echo "${utt_id}" | sed "s/_[FM][0-9]\{3\}//g")
                emo=$(grep ${utt_id} ${datadir}/Session${n}/dialog/EmoEvaluation/${ses_id}.txt \
                        | sed "s/^.*\t${utt_id}\t\([a-z]\{3\}\)\t.*$/\1/g")
                cts_emo=$(grep ${utt_id} ${datadir}/Session${n}/dialog/EmoEvaluation/${ses_id}.txt | cut -d $'\t' -f4\
                        | sed 's=\[==g' | sed 's=\]==g' | tr ',' ' ' | tr -s ' ') 
                if ! eval "echo ${remove_emo} | grep -q ${emo}" ; then
                        # for sentiment analysis
                        echo "${utt_id} <${emo}>" >> data/all/text
                        echo "${utt_id} ${file}" >> data/all/wav.scp
                        echo "${utt_id} ${utt_id}" >> data/all/utt2spk
                        echo "${utt_id} ${cts_emo}" | tr -s ' ' >> data/all/emotion_cts
                fi
            done
        done
        dos2unix data/all/wav.scp; dos2unix data/all/utt2spk; dos2unix data/all/text
        utils/utt2spk_to_spk2utt.pl data/all/utt2spk > "data/all/spk2utt"
        touch data/all/tmp.done
        utils/fix_data_dir.sh --utt_extra_files "emotion_cts" data/all
    fi

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        log "stage 2: IEMOCAP 5 fold data creation"
        mkdir -p data/folds
        for fold in 01 02 03 04 05; do 
            grep -e "Ses"${fold} data/all/utt2spk > data/folds/test${fold}_utts
            grep -v -e "Ses"${fold} data/all/utt2spk > data/folds/traindev${fold}_utts
            if [[ $fold != "01" ]]; then 
                valid_fold=$(( fold -1 ))
            else 
                valid_fold=5
            fi
            grep -e "Ses0"${valid_fold}"M" data/folds/traindev${fold}_utts > data/folds/valid${fold}_utts
            grep -v -e "Ses0"${valid_fold}"M" data/folds/traindev${fold}_utts > data/folds/train${fold}_utts
            nmix=$( cat data/folds/traindev${fold}_utts | wc -l)
            ndev=$( awk -v x=$nmix "BEGIN { print int(0.2*x)}")
            ntrain=$(($nmix-$ndev))
            for set in train valid test; do 
                utils/subset_data_dir.sh --utt-list data/folds/${set}${fold}_utts data/all data/${set}_fold${fold}
                utils/filter_scp.pl data/${set}_fold${fold}/utt2spk < data/all/emotion_cts > data/${set}_fold${fold}/emotion_cts
            done
        done 



fi 
log "Successfully finished. [elapsed=${SECONDS}s]"