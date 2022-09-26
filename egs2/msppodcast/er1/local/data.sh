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
stop_stage=100


datadir=/ocean/projects/iri120008p/roshansh/corpora/msppodcast

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: MSPPodcast Data Preparation"
    mkdir -p data
    mkdir -p data/{train,valid,test1}
    for dset in train valid test1; do 
        grep -iF ${dset} ${datadir}/labels/labels_concensus.csv | cut -d ',' -f 1,2,4,3,5,6 | \
        tr ',' ' ' | awk -F ' ' -v x=${datadir} '{$7=$1;$1=x"/Audios/"$1;gsub(".wav","",$7);if ($6 == "Unknown") {$6=9999};printf("s%04d_%s s%04d %s %s %s %s %s\n",$6,$7,$7,$2,$1,$3,$4,$5)}' > data/${dset}/all.data
        awk -F ' ' '{if (($3 == "H" )|| ($3 == "A" )||($3 == "S" )||($3 == "D" )||($3 == "N" )){print}}' data/${dset}/all.data > tmp && mv tmp data/${dset}/all.data 
        cut -d ' ' -f 1,2  data/${dset}/all.data | LC_ALL=C sort  > data/${dset}/utt2spk 
        cut -d ' ' -f 1,3  data/${dset}/all.data | LC_ALL=C sort  > data/${dset}/text
        cut -d ' ' -f 1,5,6,7  data/${dset}/all.data | LC_ALL=C sort  > data/${dset}/emotion_cts
        cut -d ' ' -f 1,4  data/${dset}/all.data | LC_ALL=C sort | awk -F ' ' '{print $1" sox -c 1 "$2" -t wav - | "}' > data/${dset}/wav.scp 
        utils/utt2spk_to_spk2utt.pl data/${dset}/utt2spk > data/${dset}/spk2utt 
        utils/validate_data_dir.sh --no-feats data/${dset}
        utils/fix_data_dir.sh --utt_extra_files "emotion_cts" data/${dset}
    done 
fi

log "Successfully finished. [elapsed=${SECONDS}s]"