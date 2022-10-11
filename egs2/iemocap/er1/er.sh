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
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.
feats_tag=""
# Tokenization related
token_type=word      # Tokenization type (char or bpe).

# Task and model related
use_discrete=true 
use_continuous=false 

# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# er model related
er_tag=       # Suffix to the result dir for er model training.
er_exp=       # Specify the directory path for er experiment.
               # If this option is specified, er_tag is ignored.
er_stats_dir= # Specify the directory path for er statistics.
er_config=    # Config for er model training.
er_args=      # Arguments for er model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in er config.
num_splits_er=1
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
lm_train_text=
# Upload model related
hf_repo=

batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_er_model=valid.acc.ave.pth # er model path for decoding.
                                      # e.g.
                                      # inference_er_model=train.loss.best.pth
                                      # inference_er_model=3epoch.pth
                                      # inference_er_model=valid.acc.best.pth
                                      # inference_er_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.

lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
er_speech_fold_length=800 # fold_length for speech data during er training.
er_text_fold_length=150   # fold_length for text data during er training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (word, default="word").

    # er model related
    --er_tag          # Suffix to the result dir for er model training (default="${er_tag}").
    --er_exp          # Specify the directory path for er experiment.
                       # If this option is specified, er_tag is ignored (default="${er_exp}").
    --er_stats_dir    # Specify the directory path for er statistics (default="${er_stats_dir}").
    --er_config       # Config for er model training (default="${er_config}").
    --er_args         # Arguments for er model training (default="${er_args}").
                       # e.g., --er_args "--max_epoch 10"
                       # Note that it will overwrite args in er config.
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_er_model # er model path for decoding (default="${inference_er_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --bpe_train_text # Text file path of bpe training set.
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --er_speech_fold_length # fold_length for speech data during er training (default="${er_speech_fold_length}").
    --er_text_fold_length   # fold_length for text data during er training (default="${er_text_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Use the same text as er for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi

# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi



# Set tag for naming of model directory
if [ -z "${er_tag}" ]; then
    if [ -n "${er_config}" ]; then
        er_tag="$(basename "${er_config}" .yaml)_${feats_type}"
    else
        er_tag="train_${feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        er_tag+="_${lang}_${token_type}"
    else
        er_tag+="_${token_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${er_args}" ]; then
        er_tag+="$(echo "${er_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        er_tag+="_sp"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${er_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        er_stats_dir="${expdir}/er_stats_${feats_type}_${lang}_${token_type}"
    else
        er_stats_dir="${expdir}/er_stats_${feats_type}_${token_type}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        er_stats_dir+="_sp"
    fi
fi
# The directory used for training commands
if [ -z "${er_exp}" ]; then
    er_exp="${expdir}/er_${er_tag}"
fi


if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    inference_tag+="_er_model_$(echo "${inference_er_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ -n "${speed_perturb_factors}" ]; then
           log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
           for factor in ${speed_perturb_factors}; do
               if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                   scripts/utils/perturb_data_dir_speed.sh "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
                   _dirs+="data/${train_set}_sp${factor} "
               else
                   # If speed factor is 1, same as the original
                   _dirs+="data/${train_set} "
               fi
           done
           utils/combine_data.sh "data/${train_set}_sp" ${_dirs}
        else
           log "Skip stage 2: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments data/${dset}/segments "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
                [ -f data/"${dset}"/emotion_cts ] && cp data/"${dset}"/emotion_cts "${data_feats}${_suf}/${dset}"/

            done

        elif [ "${feats_type}" = fbank_pitch ]; then
            log "[Require Kaldi] Stage 3: ${feats_type} extract: data/ -> ${data_feats}"

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # 1. Copy datadir
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

                # 2. Feature extract
                _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
                steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}${_suf}/${dset}"
                utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"

                # 3. Derive the the frame length and feature dimension
                scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                # 4. Write feats_dim
                head -n 1 "${data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' \
                    | cut -d, -f2 > ${data_feats}${_suf}/${dset}/feats_dim

                # 5. Write feats_type
                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

                # 6. Copy over continuous emotion
                [ -f data/"${dset}"/emotion_cts ] && cp data/"${dset}"/emotion_cts "${data_feats}${_suf}/${dset}"/

            done

        elif [ "${feats_type}" = fbank ]; then
            log "Stage 3: ${feats_type} extract: data/ -> ${data_feats}"
            log "${feats_type} is not supported yet."
            exit 1

        elif  [ "${feats_type}" = extracted ]; then
            log "Stage 3: ${feats_type} extract: data/ -> ${data_feats}"
            # Assumming you don't have wav.scp, but feats.scp is created by local/data.sh instead.

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # Generate dummy wav.scp to avoid error by copy_data_dir.sh
                <data/"${dset}"/cmvn.scp awk ' { print($1,"<DUMMY>") }' > data/"${dset}"/wav.scp
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

                # Derive the the frame length and feature dimension
                _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
                scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats${feats_tag}.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${data_feats}${_suf}/${dset}/feats${feats_tag}.scp |" - | \
                    awk '{ print $2 }' | cut -d, -f2 > "${data_feats}${_suf}/${dset}/feats_dim"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

                # Copy over continuous emotion
                [ -f data/"${dset}"/emotion_cts ] && cp data/"${dset}"/emotion_cts "${data_feats}${_suf}/${dset}"/
                


            done

        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do

            # Copy data dir
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            [ -f ${data_feats}/org/${dset}/emotion_cts ] && cp ${data_feats}/org/${dset}/emotion_cts "${data_feats}/${dset}"/


            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

                # utt2num_samples is created by format_wav_scp.sh
                <"${data_feats}/org/${dset}/utt2num_samples" \
                    awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                        >"${data_feats}/${dset}/utt2num_samples"
                <"${data_feats}/org/${dset}/wav.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/wav.scp"
            else
                # Get frame shift in ms from conf/fbank.conf
                _frame_shift=
                if [ -f conf/fbank.conf ] && [ "$(<conf/fbank.conf grep -c frame-shift)" -gt 0 ]; then
                    # Assume using conf/fbank.conf for feature extraction
                    _frame_shift="$(<conf/fbank.conf grep frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
                fi
                if [ -z "${_frame_shift}" ]; then
                    # If not existing, use the default number in Kaldi (=10ms).
                    # If you are using different number, you have to change the following value manually.
                    _frame_shift=10
                fi

                _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

                cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
                <"${data_feats}/org/${dset}/feats_shape" awk -F, ' { print $1 } ' \
                    | awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                        >"${data_feats}/${dset}/feats_shape"
                <"${data_feats}/org/${dset}/feats${feats_tag}.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/feats_shape"  \
                    >"${data_feats}/${dset}/feats.scp"
            fi

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"
            opt=""
            [ -f "${data_feats}/org/${dset}/emotion_cts" ] && opt="--utt_extra_files 'emotion_cts'"
            utils/fix_data_dir.sh ${opt} "${data_feats}/${dset}"
        done
        cut -d ' ' -f2 data/${train_set}/text > ${data_feats}/lm_train.txt
    fi


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        if [ "${token_type}" = word ]; then
            log "Stage 5: Generate character level token_list from ${lm_train_text}"

         # We don't use blanks or <unk> or <sos/eos> for this task
            ${python} -m espnet2.bin.tokenize_text  \
                --token_type "${token_type}" \
                --input "${data_feats}/lm_train.txt" --output "${token_list}" \
                --field 1- \
                --write_vocabulary true 
        else
            log "Error: not supported --token_type '${token_type}'"
            exit 2
        fi

    fi
else
    log "Skip the stages for data preparation"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _er_train_dir="${data_feats}/${train_set}"
        _er_valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: ER collect stats: train_set=${_er_train_dir}, valid_set=${_er_valid_dir}"

        _opts=
        if [ -n "${er_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.er_train --print_config --optim adam
            _opts+="--config ${er_config} "
        fi

        _feats_type="$(<${_er_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats${feats_tag}.scp
            _type=kaldi_ark
            _input_size="$(<${_er_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${er_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_er_train_dir}/${_scp} wc -l)" "$(<${_er_valid_dir}/${_scp} wc -l)")

        key_file="${_er_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_er_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${er_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${er_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${er_stats_dir}/run.sh"; chmod +x "${er_stats_dir}/run.sh"

        # 3. Submit jobs
        log "er collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2086
        tgt_opts=""
        if $use_discrete; then 
            tgt_opts+="--train_data_path_and_name_and_type ${_er_train_dir}/text,emotion,text "
            tgt_opts+="--valid_data_path_and_name_and_type ${_er_valid_dir}/text,emotion,text " 
        fi 
        if $use_continuous; then 
            tgt_opts+="--train_data_path_and_name_and_type ${_er_train_dir}/emotion_cts,emotion_cts,text_float "
            tgt_opts+="--valid_data_path_and_name_and_type ${_er_valid_dir}/emotion_cts,emotion_cts,text_float " 
        fi 

        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.er_train \
                --collect_stats true \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --train_data_path_and_name_and_type "${_er_train_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_er_valid_dir}/${_scp},speech,${_type}" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" ${tgt_opts} \
                ${_opts} ${er_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${er_stats_dir}"

    fi


    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        _er_train_dir="${data_feats}/${train_set}"
        _er_valid_dir="${data_feats}/${valid_set}"
        log "Stage 7: er Training: train_set=${_er_train_dir}, valid_set=${_er_valid_dir}"

        _opts=
        if [ -n "${er_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.er_train --print_config --optim adam
            _opts+="--config ${er_config} "
        fi

        _feats_type="$(<${_er_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((er_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats${feats_tag}.scp
            _type=kaldi_ark
            _fold_length="${er_speech_fold_length}"
            _input_size="$(<${_er_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${er_stats_dir}/train/feats_stats.npz "
        fi

        if [ "${num_splits_er}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${er_stats_dir}/splits${num_splits_er}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_er_train_dir}/${_scp}" \
                      "${_er_train_dir}/text" \
                      "${er_stats_dir}/train/speech_shape" \
                  --num_splits "${num_splits_er}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            if $use_discrete; then 
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,emotion,text "
            fi
            if $use_continuous; then 
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/emotion_cts,emotion_cts,text_float "
            fi 

            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_er_train_dir}/${_scp},speech,${_type} "
            if $use_discrete; then 
                _opts+="--train_data_path_and_name_and_type ${_er_train_dir}/text,emotion,text "
                _opts+="--valid_data_path_and_name_and_type ${_er_valid_dir}/text,emotion,text "

            fi
            if $use_continuous; then 
                _opts+="--train_data_path_and_name_and_type ${_er_train_dir}/emotion_cts,emotion_cts,text_float "
                _opts+="--valid_data_path_and_name_and_type ${_er_valid_dir}/emotion_cts,emotion_cts,text_float "

            fi 
            _opts+="--train_shape_file ${er_stats_dir}/train/speech_shape "
        fi

        log "Generate '${er_exp}/run.sh'. You can resume the process from stage 11 using this script"
        mkdir -p "${er_exp}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${er_exp}/run.sh"; chmod +x "${er_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "er training started... log: '${er_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${er_exp})"
        else
            jobname="${er_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${er_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${er_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.er_train \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --valid_data_path_and_name_and_type "${_er_valid_dir}/${_scp},speech,${_type}" \
                --valid_shape_file "${er_stats_dir}/valid/speech_shape" \
                --resume true \
                --init_param ${pretrained_model} \
                --ignore_init_mismatch ${ignore_init_mismatch} \
                --fold_length "${_fold_length}" \
                --fold_length "${er_text_fold_length}" \
                --output_dir "${er_exp}" \
                ${_opts} ${er_args}

    fi
else
    log "Skip the training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    er_exp="${expdir}/${download_model}"
    mkdir -p "${er_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${er_exp}/config.txt"

    # Get the path of each file
    _er_model_file=$(<"${er_exp}/config.txt" sed -e "s/.*'er_model_file': '\([^']*\)'.*$/\1/")
    _er_train_config=$(<"${er_exp}/config.txt" sed -e "s/.*'er_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_er_model_file}" "${er_exp}"
    ln -sf "${_er_train_config}" "${er_exp}"
    inference_er_model=$(basename "${_er_model_file}")

    if [ "$(<${er_exp}/config.txt grep -c lm_file)" -gt 0 ]; then
        _lm_file=$(<"${er_exp}/config.txt" sed -e "s/.*'lm_file': '\([^']*\)'.*$/\1/")
        _lm_train_config=$(<"${er_exp}/config.txt" sed -e "s/.*'lm_train_config': '\([^']*\)'.*$/\1/")

        lm_exp="${expdir}/${download_model}/lm"
        mkdir -p "${lm_exp}"

        ln -sf "${_lm_file}" "${lm_exp}"
        ln -sf "${_lm_train_config}" "${lm_exp}"
        inference_lm=$(basename "${_lm_file}")
    fi

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Decoding: training_dir=${er_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi

        # 2. Generate run.sh
        log "Generate '${er_exp}/${inference_tag}/run.sh'. You can resume the process from stage 12 using this script"
        mkdir -p "${er_exp}/${inference_tag}"; echo "${run_args} --stage 12 \"\$@\"; exit \$?" > "${er_exp}/${inference_tag}/run.sh"; chmod +x "${er_exp}/${inference_tag}/run.sh"
        
        er_inference_tool="espnet2.bin.er_inference"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${er_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _scp=feats${feats_tag}.scp
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/er_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/er_inference.JOB.log \
                ${python} -m ${er_inference_tool} \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --er_train_config "${er_exp}"/config.yaml \
                    --er_model_file "${er_exp}"/"${inference_er_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args}

            # 3. Concatenates the output files from each jobs
            for f in token token_int score text emo; do
                if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                  for i in $(seq "${_nj}"); do
                      cat "${_logdir}/output.${i}/1best_recog/${f}"
                  done | sort -k1 >"${_dir}/${f}"
                fi
            done
        done
    fi


    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Scoring"
        [ -f local/score.sh ] && local/score.sh "${er_exp}"
        log "Scoring not implemented for the test set"
    fi
else
    log "Skip the evaluation stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
