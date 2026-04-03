# code reference from https://github.com/seer-lab/flaky-prompt

import os
from openai import OpenAI
import pandas as pd
import chardet

# --- Load API key ---
key_file = os.path.join(os.path.dirname(__file__), "key.txt")
with open(key_file, "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

# --- Parameters ---
BATCH_SIZE = 23
MAX_ROWS = 300
PASS_NUMBER = 5

file_list = [
    "CAN-CarHacking/RPM_dataset_decoded.csv",
    "CAN-CarHacking/Fuzzy_dataset_decoded.csv"
]

experiment_type = "two_shot"

# ------------- SPLIT ----------------
def split_and_save_dataset(input_file, train_path, test_path, train_ratio=0.8):
    with open(input_file, "rb") as f:
        encoding = chardet.detect(f.read())['encoding']

    df = pd.read_csv(input_file, encoding=encoding)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSplit check:")
    print(train_df['label'].value_counts())
    print(test_df['label'].value_counts())

# ---------------- HELPERS ----------------
def flatten_decoded(decoded):
    if isinstance(decoded, str):
        try:
            d = eval(decoded)
            return ", ".join(f"{k}={v}" for k, v in d.items())
        except:
            return decoded
    return str(decoded)

def get_data_slice_with_min_T(data, max_rows=100, min_t=10):
    t_indices = data.index[data['label'] == 'T'].tolist()

    if len(t_indices) < min_t:
        raise ValueError("Not enough T examples")

    for t_idx in t_indices:
        start_idx = max(0, t_idx - max_rows // 2)
        end_idx = start_idx + max_rows
        slice_df = data.iloc[start_idx:end_idx]

        if (slice_df['label'] == 'T').sum() >= min_t:
            return slice_df.reset_index(drop=True)

    raise ValueError("No valid slice found")

def get_prompt_examples(data, data_column, num_batches=2):
    t_data = data[data['label'] == 'T']
    r_data = data[data['label'] == 'R']

    batches = []

    for _ in range(num_batches):
        t_sample = t_data.sample(n=1)
        r_sample = r_data.sample(n=BATCH_SIZE - 1)

        batch = pd.concat([t_sample, r_sample]).sample(frac=1).reset_index(drop=True)

        batch_str = "\n".join([
            f"{row['arbitration_id']},{row[data_column]} -> {row['label']}"
            for _, row in batch.iterrows()
        ])

        batches.append(batch_str)

    return batches

# ---------------- EXPERIMENT ----------------
def two_shot_batch_experiment(train_file, test_file, output_file, data_column):
    try:
        output_dir = os.path.dirname(output_file)

        # --- TRAIN ---
        with open(train_file, "rb") as f:
            train_encoding = chardet.detect(f.read())['encoding']

        train_data = pd.read_csv(train_file, encoding=train_encoding)
        train_data = get_data_slice_with_min_T(train_data, MAX_ROWS, 10)

        if data_column == "decoded":
            train_data['decoded'] = train_data['decoded'].apply(flatten_decoded)

        prompt_batches = get_prompt_examples(train_data, data_column, 2)

        # --- TEST ---
        with open(test_file, "rb") as f:
            test_encoding = chardet.detect(f.read())['encoding']

        data = pd.read_csv(test_file, encoding=test_encoding)

        if MAX_ROWS and len(data) > MAX_ROWS:
            data = data.iloc[:MAX_ROWS]

        if data_column == "decoded":
            data['decoded'] = data['decoded'].apply(flatten_decoded)

        results_df = data.copy()
        results_df['Batch_ID'] = None

        for i in range(1, PASS_NUMBER + 1):
            results_df[f"Pass_{i}"] = None

        base_prompt = (
            "We are trying to identify cybersecurity attacks in CAN messages.\n"
            "Below are labeled examples (T=attack, R=normal):\n"
            f"First batch:\n{prompt_batches[0]}\n"
            f"Second batch:\n{prompt_batches[1]}\n"
            "Now look at the following sequence.\n"
            "Consider the pattern across all messages.\n\n"
            "Is the last message part of an attack pattern?\n\n"
            "Answer Yes or No, and provide a brief 1-2 sentence reason.\n\n"
        )

        for pass_num in range(1, PASS_NUMBER + 1):
            print(f"\nRunning Pass {pass_num} for {test_file} ({data_column})")

            log_file_path = os.path.join(output_dir, f"logs_{data_column}_pass_{pass_num}.txt")
            open(log_file_path, "w").close()

            for i in range(BATCH_SIZE - 1, len(data)):
                window_df = data.iloc[i - BATCH_SIZE + 1:i + 1]

                batch_messages = "\n".join([
                    f"{row['arbitration_id']},{row[data_column]}"
                    for _, row in window_df.iterrows()
                ])

                test_case = base_prompt + batch_messages

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "You are a cybersecurity analyst specializing in automotive CAN traffic."},
                            {"role": "user", "content": test_case}
                        ]
                    )
                    prediction = response.choices[0].message.content.strip()
                except Exception as e:
                    prediction = f"Error: {e}"

                last_idx = window_df.index[-1]
                results_df.at[last_idx, f"Pass_{pass_num}"] = prediction
                results_df.at[last_idx, "Batch_ID"] = i

                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"\n--- Window ending at index {i} ---\n")
                    log_file.write("PROMPT:\n" + test_case + "\n")
                    log_file.write("RESPONSE:\n" + prediction + "\n")

        cols = ['arbitration_id', data_column, 'Batch_ID'] + [f'Pass_{i}' for i in range(1, PASS_NUMBER + 1)]
        results_df = results_df[cols]

        results_df.to_csv(output_file, index=False, encoding="utf-8")

    except Exception as e:
        print(f"Error: {e}")

# ---------------- MAIN ----------------
base_data_folder = "data"
train_folder = os.path.join(base_data_folder, "train")
test_folder = os.path.join(base_data_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# SPLIT
for input_file in file_list:
    name = os.path.basename(input_file)
    train_file = os.path.join(train_folder, name)
    test_file = os.path.join(test_folder, name)

    split_and_save_dataset(input_file, train_file, test_file)

# RUN
base_output_folder = "outputs"

for file_name in ["RPM_dataset_decoded.csv", "Fuzzy_dataset_decoded.csv"]:
    train_file = os.path.join(train_folder, file_name)
    test_file = os.path.join(test_folder, file_name)

    dataset_name = file_name.replace("_dataset_decoded.csv", "")

    folder = os.path.join(base_output_folder, experiment_type, dataset_name)
    os.makedirs(folder, exist_ok=True)

    two_shot_batch_experiment(train_file, test_file,
        os.path.join(folder, f"{experiment_type}_{dataset_name}_raw_results.csv"), "raw_data")

    two_shot_batch_experiment(train_file, test_file,
        os.path.join(folder, f"{experiment_type}_{dataset_name}_decoded_results.csv"), "decoded")