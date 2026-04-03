import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
PASS_NUMBER = 5

base_output_folder = "./outputs"
base_test_folder = "./data/test"

experiments = ["one_shot", "two_shot", "three_shot"]
datasets = ["RPM", "Fuzzy"]

os.makedirs("evaluation", exist_ok=True)

# ---------------- HELPERS ----------------

def extract_prediction(text):
    if not isinstance(text, str):
        return None
    if "Yes" in text:
        return "T"
    elif "No" in text:
        return "R"
    return None


def majority_vote(row):
    votes = []
    for i in range(1, PASS_NUMBER + 1):
        pred = extract_prediction(row.get(f"Pass_{i}", None))
        if pred:
            votes.append(pred)
    if not votes:
        return None
    return max(set(votes), key=votes.count)


def compute_metrics(y_true, y_pred):
    TP = FP = TN = FN = 0

    for t, p in zip(y_true, y_pred):
        if p is None:
            continue
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

        raw_df["Final"] = raw_df.apply(majority_vote, axis=1)
        dec_df["Final"] = dec_df.apply(majority_vote, axis=1)

        # Align lengths
        min_len = min(len(test_df), len(raw_df), len(dec_df))
        y_true = test_df["label"].iloc[:min_len]
        y_raw = raw_df["Final"].iloc[:min_len]
        y_dec = dec_df["Final"].iloc[:min_len]

        # Metrics
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


# ---------------- SAVE CSV ----------------
summary_df = pd.DataFrame(summary_rows)
csv_path = "evaluation/final_summary.csv"
summary_df.to_csv(csv_path, index=False)
print("\nSaved:", csv_path)


# ---------------- PLOTTING ----------------

def plot_metric(raw_col, dec_col, title, ylabel, filename):
    plt.figure(figsize=(10, 6))

    for dataset in datasets:
        sub = summary_df[summary_df["Dataset"] == dataset]
        plt.plot(sub["Shot"], sub[raw_col], marker='o', linestyle='--', label=f"{dataset}-Raw")
        plt.plot(sub["Shot"], sub[dec_col], marker='o', linestyle='-', label=f"{dataset}-Decoded")

    plt.title(title)
    plt.xlabel("Shot Type")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    print(f"Saved plot: {filename}")


plot_metric("Raw_AttackRecall", "Decoded_AttackRecall",
            "Attack Recall: Raw vs Decoded",
            "Attack Recall",
            "evaluation/attack_recall.png")

plot_metric("Raw_AttackPrecision", "Decoded_AttackPrecision",
            "Attack Precision: Raw vs Decoded",
            "Attack Precision",
            "evaluation/attack_precision.png")

plot_metric("Raw_AttackF1", "Decoded_AttackF1",
            "Attack F1: Raw vs Decoded",
            "Attack F1",
            "evaluation/attack_f1.png")