#!/usr/bin/env bash

# Copyright 2021  Roshan Sharma
#           2021  Carnegie Mellon University
# Apache 2.0


import argparse
import glob
import os

import numpy as np
from scipy.stats import hmean
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    recall_score,
)


def CCC(y_true, y_pred):
    x_mean = np.nanmean(y_true, dtype="float32")
    y_mean = np.nanmean(y_pred, dtype="float32")
    denom = len(y_true) - 1
    x_var = 1.0 / denom * np.nansum((y_true - x_mean) ** 2) if len(y_true) > 1 else 0
    y_var = 1.0 / denom * np.nansum((y_pred - y_mean) ** 2) if len(y_pred) > 1 else 0
    cov = np.nanmean((y_true - x_mean) * (y_pred - y_mean))
    return round((2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2), 4)


def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred).item()

def UAR(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=0).item()


parser = argparse.ArgumentParser()
parser.add_argument("--exp_root", required=True, help="Experiment root directory")
args = parser.parse_args()


for ddir in glob.glob(args.exp_root + os.sep + "inference*"):
    for tdir in os.listdir(ddir + os.sep):

        if not os.path.exists(os.path.join(ddir, tdir, "text")) and not os.path.exists(
            os.path.join(ddir, tdir, "emotion_cts")
        ):
            print(
                f"Skipping folder {os.path.join(ddir, tdir)} as text or emotion_cts files do not exist"
            )
            continue
        print(f"Processing folder {os.path.join(ddir, tdir)}")

        output = ""

        if os.path.exists(os.path.join(ddir, tdir, "text")):
            with open(os.path.join("data", tdir, "text"), "r") as f:
                ref_disc = {
                    line.strip().split(" ")[0]: line.strip().split(" ")[1]
                    for line in f.readlines()
                }
            with open(os.path.join(ddir, tdir, "text"), "r") as f:
                hyp_disc = {
                    line.strip().split(" ")[0]: line.strip().split(" ")[1]
                    for line in f.readlines()
                }
            with open(
                os.path.join("data", "en_token_list", "word", "tokens.txt"), "r"
            ) as f:
                class_map = { line.strip() : i  for i, line in enumerate(f.readlines())}
                # class_map = {i: line[i].strip() for i, line in enumerate(f.readlines())}
            keys = list(ref_disc.keys())
            # print(class_map)
            # print(hyp_disc)
            ref_disc = [class_map[ref_disc[k]] for k in keys]
            hyp_disc = [class_map[hyp_disc[k]] for k in keys]
            uar = UAR(ref_disc, hyp_disc)
            f1 = f1_score(ref_disc, hyp_disc, average="macro", zero_division=0)
            acc = accuracy_score(ref_disc, hyp_disc)
            output += f"Discrete Result: || UAR | F1 | ACC |\n"
            output += f"{os.path.join(ddir, tdir, 'text')} Result:|| {uar:.4f} | {f1:.4f} | {acc:.4f}|\n"

        if os.path.exists(os.path.join(ddir, tdir, "emotion_cts")):
            with open(os.path.join("data", tdir, "emotion_cts"), "r") as f:
                ref_emo = {
                    line.strip().split(" ")[0]: [
                        float(x) for x in line.strip().split(" ")[1:]
                    ]
                    for line in f.readlines()
                }
            with open(os.path.join(ddir, tdir, "emotion_cts"), "r") as f:
                hyp_emo = {
                    line.strip().split(" ")[0]: [
                        float(x) for x in line.strip().split(" ")[1].split(",")
                    ]
                    for line in f.readlines()
                }

            ref_emot = np.array([ref_emo[k] for k in keys])
            hyp_emot = np.array([hyp_emo[k] for k in keys])
            classwise_ccc = [
                CCC(ref_emot[:, i], hyp_emot[:, i]) for i in range(ref_emot.shape[-1])
            ]
            ccc = np.mean(classwise_ccc)
            mae = MAE(ref_emot, hyp_emot)  
            output += f"Continuous Result: || CCC | MAE |"        
            output = f" {os.path.join(ddir, tdir, 'emotion_cts')} Result: || {ccc} | {mae} |\n"
            output += f"Classwise CCC: {classwise_ccc}"

        with open(os.path.join(ddir, tdir, "report.txt"), "w") as f:
            f.write(output)
