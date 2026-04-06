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
MAX_ROWS = 400
PASS_NUMBER = 5

file_list = [
    "CAN-CarHacking/RPM_dataset_decoded.csv",
    "CAN-CarHacking/Fuzzy_dataset_decoded.csv"
]

experiment_type = "zero_shot"

# ------------- SPLIT ----------------
def split_and_save_dataset(input_file, train_path, test_path, train_ratio=0.7):
    with open(input_file, "rb") as f:
        encoding = chardet.detect(f.read())['encoding']

    df = pd.read_csv(input_file, encoding=encoding)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

# ------------- HELPERS ----------------
def flatten_decoded(decoded):
    if isinstance(decoded, str):
        try:
            d = eval(decoded)
            return ", ".join(f"{k}={v}" for k, v in d.items())
        except:
            return decoded
    return str(decoded)

# ------------- EXPERIMENT ----------------
def zero_shot_batch_experiment(test_file, output_file, data_column, dataset_name):
    try:
        output_dir = os.path.dirname(output_file)

        # --- LOAD TEST ---
        with open(test_file, "rb") as f:
            test_encoding = chardet.detect(f.read())['encoding']
        test_data = pd.read_csv(test_file, encoding=test_encoding)

        if MAX_ROWS and len(test_data) > MAX_ROWS:
            test_data = test_data.iloc[:MAX_ROWS]

        if data_column == "decoded":
            test_data['decoded'] = test_data['decoded'].apply(flatten_decoded)

        results_df = test_data.copy()
        results_df['Batch_ID'] = None

        for i in range(1, PASS_NUMBER + 1):
            results_df[f"Pass_{i}"] = None

        # --- DYNAMIC PROMPT ---
        if "RPM" in dataset_name.upper():
            attack_definition = "An RPM attack (Spoofing) involves injecting false, sudden, and anomalous RPM values that break the smooth, continuous physical properties of the engine speed."
        else:
            attack_definition = "A Fuzzy attack involves flooding the network with random, unrecognized CAN IDs or highly chaotic data payloads."

        base_prompt = (
            f"We are analyzing automotive CAN bus traffic for '{dataset_name}' cybersecurity attacks.\n"
            f"{attack_definition}\n\n"
            "=== TARGET SEQUENCE ===\n"
            "Look at the following sequence of messages.\n"
            "Consider the pattern across ALL messages in the window.\n\n"
            f"Does this sequence contain a {dataset_name} attack (is there any injected anomaly anywhere in the sequence)?\n\n"
            "Answer Yes or No, and provide a brief 1-2 sentence reason.\n\n"
        )

        for pass_num in range(1, PASS_NUMBER + 1):
            print(f"\nRunning Pass {pass_num} for {test_file} ({data_column})")

            log_file_path = os.path.join(output_dir, f"logs_{data_column}_pass_{pass_num}.txt")
            open(log_file_path, "w").close()

            for i in range(BATCH_SIZE - 1, len(test_data)):
                window_df = test_data.iloc[i - BATCH_SIZE + 1:i + 1]

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
                            {"role": "system", "content": "You are a cybersecurity analyst. Your primary goal is to detect anomalous injected patterns in CAN traffic windows."},
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
                    log_file.write(f"\n--- Window ending at index {i} ---\nPROMPT:\n{test_case}\nRESPONSE:\n{prediction}\n")

        cols = ['arbitration_id', data_column, 'Batch_ID'] + [f'Pass_{i}' for i in range(1, PASS_NUMBER + 1)]
        results_df = results_df[cols]
        results_df.to_csv(output_file, index=False, encoding="utf-8")

    except Exception as e:
        print(f"Error: {e}")

# ------------- MAIN ----------------
base_data_folder = "data"
train_folder = os.path.join(base_data_folder, "train")
test_folder = os.path.join(base_data_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for input_file in file_list:
    name = os.path.basename(input_file)
    split_and_save_dataset(input_file, os.path.join(train_folder, name), os.path.join(test_folder, name))

base_output_folder = "outputs"

for input_file in file_list:
    file_name = os.path.basename(input_file)
    test_file = os.path.join(test_folder, file_name)
    dataset_name = file_name.replace("_dataset_decoded.csv", "")

    folder = os.path.join(base_output_folder, experiment_type, dataset_name)
    os.makedirs(folder, exist_ok=True)

    zero_shot_batch_experiment(test_file, os.path.join(folder, f"{experiment_type}_{dataset_name}_raw_results.csv"), "raw_data", dataset_name)
    zero_shot_batch_experiment(test_file, os.path.join(folder, f"{experiment_type}_{dataset_name}_decoded_results.csv"), "decoded", dataset_name)