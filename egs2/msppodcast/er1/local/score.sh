#!/usr/bin/env bash
# Copyright 2021  Roshan Sharma
#           2021  Carnegie Mellon University

# # begin configuration section.
# cmd=run.pl
# stage=0
# data=data/eval2000
# #end configuration section.




[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir>"
  exit 1;
fi

er_expdir=$1

python pyscripts/utils/score_emotion.py --exp_root ${er_expdir}

exit 0
