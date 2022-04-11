#!/usr/bin/env bash

# Copyright 2021  Roshan Sharma
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys
import pandas as pd
import argparse
from collections import Counter
import glob
from sklearn.metrics import classification_report
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", required=True, help="Directory to save experiments")
args = parser.parse_args()


class_map = {"<sad>": 0, "<hap>": 1, "<ang>": 2, "<neu>": 3}



for ddir in glob.glob(args.exp_root + os.sep + "decode*"):
    for tdir in os.listdir(ddir + os.sep):
        if not os.path.exists(os.path.join(ddir, tdir, "text")):
            print(f"Skipping folder {os.path.join(ddir, tdir)} as hyp does not exist")
            continue
        with open(os.path.join("data", tdir, "text"), "r") as f:
            ref = {
            line.strip().split(" ")[0]: line.strip().split(" ")[1] for line in f.readlines()
            }
        with open(os.path.join(ddir, tdir, "text"), "r") as f:
            hyp = {
                line.strip().split(" ")[0]: line.strip().split(" ")[1]
                for line in f.readlines()
            }
        # ref = valid_ref if tdir == "valid" else test_ref
        keys = list(ref.keys())
        ref = [class_map[ref[k]] for k in keys]
        hyp = [class_map[hyp[k]] for k in keys]
        precision_metric = precision_score(ref, hyp, average="macro")
        recall_metric = recall_score(ref, hyp, average="macro")
        accuracy_metric = accuracy_score(ref, hyp)
        f1_metric = f1_score(ref, hyp, average="macro")
        # roc_metric = roc_auc_score(ref, hyp, average="macro", multi_class="ovo")
        output = f"RES: {accuracy_metric}| F-1 {f1_metric} | {recall_metric}| Precision {precision_metric}| "
        output += "\n"
        output += classification_report(ref, hyp)
        with open(os.path.join(ddir, tdir, "classification_report.txt"), "w") as f:
            f.write(output)

