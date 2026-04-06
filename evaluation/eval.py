import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

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

    text = text.strip().lower()

    if text.startswith("yes"):
        return "T"
    elif text.startswith("no"):
        return "R"

    return None

def adaptive_majority_vote(row, dataset):
    """
    Returns 'T' or 'R' using dataset-specific threshold.
    - RPM: majority vote (more than half)
    - Fuzzy: if any T, return T; else R -- only because Fuzzy is very imbalanced (~2% T's)
    """
    votes = []
    for i in range(1, PASS_NUMBER + 1):
        pred = extract_prediction(row.get(f"Pass_{i}", None))
        if pred:
            votes.append(pred)

    t_count = votes.count("T")
    r_count = votes.count("R")

    if dataset == "RPM":
        threshold = (PASS_NUMBER // 2) + 1  # majority
        if t_count >= threshold:
            return "T"
        elif r_count >= threshold:
            return "R"
        else:
            return "R"  # fallback to R if no majority
    else:  # Fuzzy
        if t_count >= 1:
            return "T"
        else:
            return "R"  # fallback to R if no T

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
        "Attack_F1": f1,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }

# ---------------- MAIN ----------------

summary_rows = []
per_pass_rows = []

for dataset in datasets:
    test_file = os.path.join(base_test_folder, f"{dataset}_dataset_decoded.csv")
    test_df = pd.read_csv(test_file)

    for exp in experiments:

        raw_file = os.path.join(base_output_folder, exp, dataset, f"{exp}_{dataset}_raw_results.csv")
        dec_file = os.path.join(base_output_folder, exp, dataset, f"{exp}_{dataset}_decoded_results.csv")

        if not os.path.exists(raw_file) or not os.path.exists(dec_file):
            print(f"Missing files: {exp} | {dataset}")
            continue

        # Load predictions
        raw_df = pd.read_csv(raw_file).copy()
        dec_df = pd.read_csv(dec_file).copy()

        # ---------------- APPLY ADAPTIVE VOTE ----------------
        raw_df["Final"] = raw_df.apply(lambda row: adaptive_majority_vote(row, dataset), axis=1)
        dec_df["Final"] = dec_df.apply(lambda row: adaptive_majority_vote(row, dataset), axis=1)

        # Align lengths
        min_len = min(len(test_df), len(raw_df), len(dec_df))
        y_true = test_df["label"].iloc[:min_len]

        y_raw = raw_df["Final"].iloc[:min_len]
        y_dec = dec_df["Final"].iloc[:min_len]

        # ---------------- METRICS ----------------
        raw_metrics = compute_metrics(y_true, y_raw)
        dec_metrics = compute_metrics(y_true, y_dec)

        # ---------------- DISAGREEMENT ANALYSIS ----------------
        comp = pd.DataFrame({
            "true": y_true,
            "raw": y_raw,
            "decoded": y_dec
        })

        diff = comp[comp["raw"] != comp["decoded"]]
        decoded_better = diff[(diff["decoded"] == diff["true"]) & (diff["raw"] != diff["true"])]
        decoded_worse = diff[(diff["raw"] == diff["true"]) & (diff["decoded"] != diff["true"])]

        if len(decoded_better) > len(decoded_worse):
            insight = "Decoded helps"
        elif len(decoded_better) < len(decoded_worse):
            insight = "Decoded hurts"
        else:
            insight = "No significant difference"

        # ---------------- PRINT TABLE ----------------
        print(f"\n=== {dataset} | {exp} ===")
        print(pd.DataFrame([{
            "Raw Recall": raw_metrics["Attack_Recall"],
            "Decoded Recall": dec_metrics["Attack_Recall"],
            "Raw F1": raw_metrics["Attack_F1"],
            "Decoded F1": dec_metrics["Attack_F1"]
        }]).to_string(index=False))

        # Optional: count T occurrences per pass for inspection
        print("Raw T counts:", Counter(raw_df["Final"].iloc[:min_len]))
        print("Decoded T counts:", Counter(dec_df["Final"].iloc[:min_len]))

        # ---------------- STORE SUMMARY ----------------
        row = {
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

            "Decoded_Better": len(decoded_better),
            "Decoded_Worse": len(decoded_worse),
            "Total_Disagreements": len(diff),
            "Insight": insight
        }

        if raw_metrics["Attack_Recall"] != 0:
            row["Recall_Improvement_%"] = (
                (dec_metrics["Attack_Recall"] - raw_metrics["Attack_Recall"])
                / raw_metrics["Attack_Recall"] * 100
            )
        else:
            row["Recall_Improvement_%"] = 0

        summary_rows.append(row)

        # ---------------- PER-PASS METRICS ----------------
        for i in range(1, PASS_NUMBER + 1):
            raw_preds = raw_df[f"Pass_{i}"].apply(extract_prediction).iloc[:min_len]
            dec_preds = dec_df[f"Pass_{i}"].apply(extract_prediction).iloc[:min_len]

            raw_m = compute_metrics(y_true, raw_preds)
            dec_m = compute_metrics(y_true, dec_preds)

            per_pass_rows.append({
                "Dataset": dataset,
                "Shot": exp,
                "Pass": f"Pass_{i}",

                "Raw_Accuracy": raw_m["Accuracy"],
                "Decoded_Accuracy": dec_m["Accuracy"],

                "Raw_AttackPrecision": raw_m["Attack_Precision"],
                "Decoded_AttackPrecision": dec_m["Attack_Precision"],

                "Raw_AttackRecall": raw_m["Attack_Recall"],
                "Decoded_AttackRecall": dec_m["Attack_Recall"],

                "Raw_AttackF1": raw_m["Attack_F1"],
                "Decoded_AttackF1": dec_m["Attack_F1"],
            })

# ---------------- SAVE CSV ----------------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("evaluation/final_summary.csv", index=False)
print("\nSaved: evaluation/final_summary.csv")

per_pass_df = pd.DataFrame(per_pass_rows)
per_pass_df.to_csv("evaluation/per_pass_summary.csv", index=False)
print("Saved: evaluation/per_pass_summary.csv")
# ---------------- PLOTTING (SEPARATED BY DATASET) ----------------

def plot_metric(raw_col, dec_col, title, ylabel, filename_prefix):
    """
    Creates one plot per dataset (RPM, Fuzzy) instead of mixing them.
    """
    for dataset in datasets:
        plt.figure(figsize=(8, 5))

        sub = summary_df[summary_df["Dataset"] == dataset]

        plt.plot(sub["Shot"], sub[raw_col],
                 marker='o', linestyle='--', label="Raw")
        plt.plot(sub["Shot"], sub[dec_col],
                 marker='o', linestyle='-', label="Decoded")

        plt.title(f"{dataset} - {title}")
        plt.xlabel("Shot Type")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        filename = f"{filename_prefix}_{dataset}.png"
        plt.savefig(filename)
        plt.show()

        print(f"Saved plot: {filename}")


def plot_per_pass(metric_raw, metric_dec, title, ylabel, filename_prefix):
    """
    Creates one per-pass plot per dataset.
    """
    for dataset in datasets:
        plt.figure(figsize=(10, 6))

        for shot in experiments:
            sub = per_pass_df[
                (per_pass_df["Dataset"] == dataset) &
                (per_pass_df["Shot"] == shot)
            ]

            if sub.empty:
                continue

            x = [int(p.split("_")[1]) for p in sub["Pass"]]

            plt.plot(x, sub[metric_raw],
                     linestyle='--', marker='o',
                     label=f"{shot}-Raw")

            plt.plot(x, sub[metric_dec],
                     linestyle='-', marker='o',
                     label=f"{shot}-Decoded")

        plt.title(f"{dataset} - {title}")
        plt.xlabel("Pass Number")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        filename = f"{filename_prefix}_{dataset}.png"
        plt.savefig(filename)
        plt.show()

        print(f"Saved plot: {filename}")


# ---------------- RUN PLOTS ----------------

plot_metric("Raw_AttackRecall", "Decoded_AttackRecall",
            "Attack Recall", "Recall",
            "evaluation/attack_recall")

plot_metric("Raw_AttackPrecision", "Decoded_AttackPrecision",
            "Attack Precision", "Precision",
            "evaluation/attack_precision")

plot_metric("Raw_AttackF1", "Decoded_AttackF1",
            "Attack F1", "F1 Score",
            "evaluation/attack_f1")


plot_per_pass("Raw_AttackRecall", "Decoded_AttackRecall",
            "Per-Pass Attack Recall", "Recall",
            "evaluation/per_pass_recall")

plot_per_pass("Raw_AttackPrecision", "Decoded_AttackPrecision",
            "Per-Pass Attack Precision", "Precision",
            "evaluation/per_pass_precision")

plot_per_pass("Raw_AttackF1", "Decoded_AttackF1",
            "Per-Pass Attack F1", "F1 Score",
            "evaluation/per_pass_f1")