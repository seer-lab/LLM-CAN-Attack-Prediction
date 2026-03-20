# code reference from https://github.com/seer-lab/flaky-prompt

import os
from openai import OpenAI
import pandas as pd
import chardet

# Read API key from local file
key_file = os.path.join(os.path.dirname(__file__), "key.txt")
with open(key_file, "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)
BATCH_SIZE = 23
MAX_ROWS = 500
PASS_NUMBER = 5

file_list = [
    "CAN-CarHacking/DoS_dataset_decoded.csv",
    "CAN-CarHacking/RPM_dataset_decoded.csv",
    "CAN-CarHacking/gear_dataset_decoded.csv",
    "CAN-CarHacking/Fuzzy_dataset_decoded.csv"
]

experiment_type = "one_shot"

def flatten_decoded(decoded):
    if isinstance(decoded, str):
        try:
            d = eval(decoded)
            return ", ".join(f"{k}={v}" for k,v in d.items())
        except:
            return decoded
    return str(decoded)

def get_prompt_example(data, batch_size=BATCH_SIZE):
    tr_data = data[data['label'].isin(['T','R'])]
    sample = tr_data.sample(n=min(batch_size, len(tr_data)), random_state=42)
    return "\n".join([f"{row['arbitration_id']},{row['raw_data']} -> {row['label']}" for _, row in sample.iterrows()])

def one_shot_batch_experiment(input_file, output_file, data_column, pass_number=PASS_NUMBER, batch_size=BATCH_SIZE, max_rows=MAX_ROWS):
    try:
        # Detect encoding
        with open(input_file, "rb") as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']

        # Load data and remove UNDECODABLE rows
        data = pd.read_csv(input_file, encoding=encoding)
        data = data[data['decoded'] != "UNDECODABLE"]

        if max_rows is not None and len(data) > max_rows:
            data = data.sample(n=max_rows, random_state=42)

        # Flatten decoded column if needed
        if data_column == "decoded":
            data['decoded'] = data['decoded'].apply(flatten_decoded)

        column_to_use = data_column
        results_df = data.copy()
        results_df['Batch_ID'] = None

        # Build prompt example for few-shot
        prompt_example = get_prompt_example(data, batch_size)
        base_prompt = (
            "We are trying to identify cybersecurity attacks in the Controller Area Network (CAN) of an automobile.\n"
            "Below is a labeled example (T=attack, R=normal):\n"
            f"{prompt_example}\n"
            "Now consider the following batch of messages. Are there any cybersecurity attacks in these messages?\n"
            "Answer Yes or No only. Give a single word as your answer.\n"
        )

        # Split into batches
        batches = [data.iloc[i:i+batch_size] for i in range(0, len(data), batch_size)]

        for pass_num in range(1, pass_number + 1):
            print(f"\nRunning Pass {pass_num} for {input_file} ({data_column})")

            for batch_id, batch_df in enumerate(batches):
                batch_messages = "\n".join([f"{row['arbitration_id']},{row[column_to_use]}" for _, row in batch_df.iterrows()])

                # Include all previous pass predictions in context
                previous_predictions = []
                for prev_pass in range(1, pass_num):
                    previous_predictions.append(
                        f"Pass {prev_pass}: {results_df.loc[batch_df.index[0], f'Pass_{prev_pass}']}"
                    )
                context = ""
                if previous_predictions:
                    context = "Previous predictions for this batch:\n" + "\n".join(previous_predictions) + "\n"

                test_case = base_prompt + context + batch_messages

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "You are a cybersecurity analyst specializing in automotive Controller Area Network (CAN) traffic."},
                            {"role": "user", "content": test_case}
                        ]
                    )
                    prediction = response.choices[0].message.content.strip()
                except Exception as e:
                    prediction = f"Error: {e}"
                    print(f"Error on batch {batch_id}: {e}")

                # Save predictions
                for idx in batch_df.index:
                    results_df.at[idx, f"Pass_{pass_num}"] = prediction
                    results_df.at[idx, "Batch_ID"] = batch_id

        # Keep only the relevant columns
        columns_to_keep = ['arbitration_id', data_column, 'Batch_ID'] + [f'Pass_{i}' for i in range(1, pass_number + 1)]
        results_df = results_df[columns_to_keep]

        results_df.to_csv(output_file, index=False, encoding="utf-8")

    except Exception as e:
        print(f"An error occurred: {e}")

base_output_folder = "outputs"

for input_file in file_list:
    # Get base filename without extension
    base_name = os.path.basename(input_file).replace(".csv", "")
    
    # Remove '_decoded' if it exists
    clean_name = base_name.replace("_decoded", "")
    
    # Extract dataset name (DoS, RPM, Gear, Fuzzy)
    dataset_name = clean_name.split("_")[0]
    
    # Create folders if they don't exist
    folder_path = os.path.join(base_output_folder, experiment_type, dataset_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Output file paths
    raw_output_file = os.path.join(folder_path, f"{experiment_type}_{dataset_name}_raw_results.csv")
    decoded_output_file = os.path.join(folder_path, f"{experiment_type}_{dataset_name}_decoded_results.csv")
    
    # Run experiments
    one_shot_batch_experiment(input_file, raw_output_file, "raw_data", PASS_NUMBER)
    one_shot_batch_experiment(input_file, decoded_output_file, "decoded", PASS_NUMBER)


# generate log files for prompt and response to identify training vs test data
# check test data before prompting

# (quantitative analysis)
# is the last message an attack in the batch? (go with this one)
# is there an attack in the batch? (do not go with this one)
# provide 23 messages as training and 23 message as testing?

# what sort of attack and why? (qualitative analysis)