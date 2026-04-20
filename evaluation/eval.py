import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import re

from sklearn.metrics import precision_recall_curve,roc_curve, average_precision_score, roc_auc_score

import seaborn as sns

# ---------------- CONFIG ----------------
PASS_NUMBER = 5

base_output_folder = "./outputs"
base_test_folder = "./data/test"

experiments = ["zero_shot", "one_shot", "two_shot", "three_shot"]
datasets = ["RPM", "Fuzzy"]

os.makedirs("evaluation", exist_ok=True)

# ---------------- HELPERS ----------------

def extract_prediction(text):
    if not isinstance(text, str):
        return None

    t = text.lower()

    if re.search(r"\byes\b", t):
        return "T"
    if re.search(r"\bno\b", t):
        return "R"

    return None



def majority_vote(row):
    votes = []
    for i in range(1, PASS_NUMBER + 1):
        p = extract_prediction(row.get(f"Pass_{i}", None))
        if p:
            votes.append(p)

    if len(votes) == 0:
        return "R"

    return Counter(votes).most_common(1)[0][0]


def get_score(row):
    votes = []
    for i in range(1, PASS_NUMBER + 1):
        p = extract_prediction(row.get(f"Pass_{i}", None))
        if p == "T":
            votes.append(1)
        elif p == "R":
            votes.append(0)

    if len(votes) == 0:
        return 0.5

    return np.mean(votes)


def compute_metrics(y_true, y_pred):
    TP = FP = TN = FN = 0

    for t, p in zip(y_true, y_pred):
        if t == 'T' and p == 'T':
            TP += 1
        elif t == 'R' and p == 'R':
            TN += 1
        elif t == 'R' and p == 'T':
            FP += 1
        elif t == 'T' and p == 'R':
            FN += 1

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return {
        "Accuracy": accuracy,
        "Attack_Precision": precision,
        "Attack_Recall": recall,
        "Attack_F1": f1
    }


# ---------------- STORAGE ----------------
summary_rows = []
per_pass_rows = []
curve_data = []

# ---------------- MAIN ----------------

for dataset in datasets:

    test_file = os.path.join(base_test_folder, f"{dataset}_dataset_decoded.csv")
    test_df = pd.read_csv(test_file)

    for exp in experiments:

        raw_file = os.path.join(base_output_folder, exp, dataset, f"{exp}_{dataset}_raw_results.csv")
        dec_file = os.path.join(base_output_folder, exp, dataset, f"{exp}_{dataset}_decoded_results.csv")

        if not os.path.exists(raw_file) or not os.path.exists(dec_file):
            print(f"Missing: {exp} | {dataset}")
            continue

        raw_df = pd.read_csv(raw_file).copy()
        dec_df = pd.read_csv(dec_file).copy()

        # ---------------- ALIGNMENT FIX ----------------
        min_len = min(len(test_df), len(raw_df), len(dec_df))

        test_df = test_df.iloc[:min_len].reset_index(drop=True)
        raw_df = raw_df.iloc[:min_len].reset_index(drop=True)
        dec_df = dec_df.iloc[:min_len].reset_index(drop=True)

        y_true = test_df["label"].values
        y_true_bin = [1 if y == "T" else 0 for y in y_true]

        # ---------------- FINAL PREDICTIONS ----------------
        raw_df["Final"] = raw_df.apply(majority_vote, axis=1)
        dec_df["Final"] = dec_df.apply(majority_vote, axis=1)

        y_raw = raw_df["Final"].values
        y_dec = dec_df["Final"].values

        # ---------------- SCORES ----------------
        raw_scores = raw_df.apply(get_score, axis=1).values
        dec_scores = dec_df.apply(get_score, axis=1).values

        curve_data.append({
            "Dataset": dataset,
            "Shot": exp,
            "y_true": y_true_bin,
            "raw_scores": list(raw_scores),
            "dec_scores": list(dec_scores)
        })

        # ---------------- METRICS ----------------
        raw_metrics = compute_metrics(y_true, y_raw)
        dec_metrics = compute_metrics(y_true, y_dec)

        summary_rows.append({
            "Dataset": dataset,
            "Shot": exp,

            "Raw_Accuracy": raw_metrics["Accuracy"],
            "Decoded_Accuracy": dec_metrics["Accuracy"],

            "Raw_AttackPrecision": raw_metrics["Attack_Precision"],
            "Decoded_AttackPrecision": dec_metrics["Attack_Precision"],

            "Raw_AttackRecall": raw_metrics["Attack_Recall"],
            "Decoded_AttackRecall": dec_metrics["Attack_Recall"],

            "Raw_AttackF1": raw_metrics["Attack_F1"],
            "Decoded_AttackF1": dec_metrics["Attack_F1"],
        })

        # ---------------- PER PASS ----------------
        for i in range(1, PASS_NUMBER + 1):

            raw_preds = raw_df[f"Pass_{i}"].apply(extract_prediction)
            dec_preds = dec_df[f"Pass_{i}"].apply(extract_prediction)

            raw_preds = raw_preds.fillna("R")
            dec_preds = dec_preds.fillna("R")

            raw_m = compute_metrics(y_true, raw_preds)
            dec_m = compute_metrics(y_true, dec_preds)

            per_pass_rows.append({
                "Dataset": dataset,
                "Shot": exp,
                "Pass": f"Pass_{i}",

                "Raw_AttackPrecision": raw_m["Attack_Precision"],
                "Decoded_AttackPrecision": dec_m["Attack_Precision"],

                "Raw_AttackRecall": raw_m["Attack_Recall"],
                "Decoded_AttackRecall": dec_m["Attack_Recall"],

                "Raw_AttackF1": raw_m["Attack_F1"],
                "Decoded_AttackF1": dec_m["Attack_F1"],
            })

# ---------------- SAVE ----------------

summary_df = pd.DataFrame(summary_rows)
per_pass_df = pd.DataFrame(per_pass_rows)

summary_df.to_csv("evaluation/final_summary.csv", index=False)
per_pass_df.to_csv("evaluation/per_pass_summary.csv", index=False)

# ---------------- AUC (FIXED) ----------------

auc_rows = []

for item in curve_data:

    y_true = item["y_true"]
    raw_scores = item["raw_scores"]
    dec_scores = item["dec_scores"]

    pr_raw = average_precision_score(y_true, raw_scores)
    pr_dec = average_precision_score(y_true, dec_scores)

    roc_raw = roc_auc_score(y_true, raw_scores)
    roc_dec = roc_auc_score(y_true, dec_scores)

    auc_rows.append({
        "Dataset": item["Dataset"],
        "Shot": item["Shot"],
        "PR_AUC_Raw": pr_raw,
        "PR_AUC_Decoded": pr_dec,
        "ROC_AUC_Raw": roc_raw,
        "ROC_AUC_Decoded": roc_dec,
    })

auc_df = pd.DataFrame(auc_rows)
auc_df.to_csv("evaluation/auc_summary.csv", index=False)

# ---------------- PLOTS ----------------

def plot_metric(raw_col, dec_col, title):
    for dataset in datasets:
        sub = summary_df[summary_df["Dataset"] == dataset]
        x = np.arange(len(sub))
        w = 0.35

        plt.figure()
        plt.bar(x - w/2, sub[raw_col], width=w, label="Raw")
        plt.bar(x + w/2, sub[dec_col], width=w, label="Decoded")
        plt.xticks(x, sub["Shot"])
        plt.title(f"{dataset} - {title}")
        plt.legend()
        plt.grid()
        plt.savefig(f"evaluation/{title}_{dataset}.png")
        plt.show()


def plot_per_pass(metric):
    for dataset in datasets:
        plt.figure()

        for shot in experiments:
            sub = per_pass_df[
                (per_pass_df["Dataset"] == dataset) &
                (per_pass_df["Shot"] == shot)
            ].sort_values("Pass")

            x = np.arange(len(sub))
            plt.plot(x, sub[metric], marker='o', label=shot)

        plt.title(f"{dataset} - {metric}")
        plt.legend()
        plt.grid()
        plt.savefig(f"evaluation/{metric}_{dataset}.png")
        plt.show()


def plot_pr_roc():
    for dataset in datasets:

        plt.figure()
        for item in curve_data:
            if item["Dataset"] != dataset:
                continue

            p1, r1, _ = precision_recall_curve(item["y_true"], item["raw_scores"])
            p2, r2, _ = precision_recall_curve(item["y_true"], item["dec_scores"])

            plt.plot(r1, p1, "--", label=f"{item['Shot']} Raw")
            plt.plot(r2, p2, label=f"{item['Shot']} Decoded")

        plt.title(f"{dataset} PR Curve")
        plt.legend()
        plt.grid()
        plt.savefig(f"evaluation/pr_{dataset}.png")
        plt.show()

        plt.figure()
        for item in curve_data:
            if item["Dataset"] != dataset:
                continue

            f1, t1, _ = roc_curve(item["y_true"], item["raw_scores"])
            f2, t2, _ = roc_curve(item["y_true"], item["dec_scores"])

            plt.plot(f1, t1, "--", label=f"{item['Shot']} Raw")
            plt.plot(f2, t2, label=f"{item['Shot']} Decoded")

        plt.title(f"{dataset} ROC Curve")
        plt.legend()
        plt.grid()
        plt.savefig(f"evaluation/roc_{dataset}.png")
        plt.show()

def plot_delta():
    for dataset in datasets:
        sub = summary_df[summary_df["Dataset"] == dataset]

        for metric in ["AttackRecall", "AttackPrecision", "AttackF1"]:
            delta = sub[f"Decoded_{metric}"] - sub[f"Raw_{metric}"]

            plt.figure()
            plt.bar(sub["Shot"], delta)
            plt.axhline(0)
            plt.title(f"{dataset} Δ {metric}")
            plt.grid()
            plt.savefig(f"evaluation/delta_{dataset}_{metric}.png")
            plt.show()

# ---------------- RUN ----------------

plot_metric("Raw_AttackRecall", "Decoded_AttackRecall", "Recall")
plot_metric("Raw_AttackPrecision", "Decoded_AttackPrecision", "Precision")
plot_metric("Raw_AttackF1", "Decoded_AttackF1", "F1")

plot_per_pass("Raw_AttackRecall")
plot_per_pass("Decoded_AttackRecall")

plot_pr_roc()
plot_delta()