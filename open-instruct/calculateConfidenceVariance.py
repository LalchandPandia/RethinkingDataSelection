import json
import pandas as pd
import sys

import argparse
import pickle
import sys

parser = argparse.ArgumentParser(
    description="Process perplexities csv files into LP(1) subsets."
)
parser._action_groups.pop()
required = parser.add_argument_group("Required Arguments")
optional = parser.add_argument_group("Optional Arguments")
required.add_argument(
    "--dataset",
    help="The dataset for which the perplexities were dumped.",
    required=True,
)
required.add_argument(
    "--percentage",
    help="Percentage (1-100) of original dataset to select.",
    required=True,
    type=int,
    choices=range(1, 101),
)
required.add_argument(
        "--model_name",
    help="Optional model name which is added to filenames of generated subsets.",
    default="model",
)
args = parser.parse_args()
MODEL_NAME = args.model_name


# List of file paths (adjust with your actual file paths)
file_dir = '/path/to/scratch/open_instruct/dump_pplxs/'
files = [f'{file_dir}epoch-0-train_{args.dataset}.json', f'{file_dir}epoch-1-train_{args.dataset}.json', f'{file_dir}epoch-2-train_{args.dataset}.json']
processed_dir = '/path/to/scratch/data/processed/'
# Read all JSONL files into a list of dataframes
dfs = []
for file in files:
    df = pd.read_json(file, lines=True)  # Read each JSONL file into a DataFrame
    dfs.append(df)

# Concatenate all DataFrames into one
all_data = pd.concat(dfs, ignore_index=True)

# Group by 'id' and calculate the mean and standard deviation for 'confidence'
result = all_data.groupby('id')['likelihood'].agg('mean').reset_index()

# Calculate standard deviation separately, ensuring ddof=0 for population std
std_result = all_data.groupby('id')['likelihood'].std(ddof=0).reset_index()

# Merge the mean and standard deviation results
result = result.merge(std_result, on='id', how='left')
# Rename columns for clarity
result.columns = ['id', 'confidence', 'variability']

# Handle the case where the standard deviation is NaN (i.e., only one data point per id)
#result['low variability'] = result['variability'].fillna(0)
result['id'] = result['id'].apply(lambda x: int(x) if not pd.isna(x) else 0)

second_file = f'{processed_dir}{args.dataset}.jsonl'  # Replace with the actual file path
second_data = []

# Load the second file
with open(second_file, 'r') as f:
    for line in f:
        second_data.append(json.loads(line))

second_df = pd.DataFrame(second_data)

# Ensure 'id' columns are of the same type for merging
second_df['id'] = second_df['id'].astype(int)

# Step 4: Merge the two DataFrames on 'id'
merged_df = pd.merge(result, second_df, on='id', how='left')

# Step 5: Write the merged DataFrame to a new JSONL file
output_file = f'{file_dir}output_with_mean_and_std_{args.dataset}.jsonl'

with open(output_file, 'w') as f:
    for _, row in merged_df.iterrows():
        json.dump(row.to_dict(), f)
        f.write('\n')

print(f"Output saved to {output_file}")

low_hard_to_learn = []
low_ambiguous = []
low_easy_to_learn = []

sort_pplx_p0 = merged_df.sort_values(by=["confidence"]).reset_index(drop=False)
sort_pplx_p1 = merged_df.sort_values(by=["variability"], ascending=False).reset_index(drop=False)
sort_pplx_p0_d = merged_df.sort_values(by=["confidence"], ascending=False).reset_index(drop=False)

indexes_hard_to_learn = sort_pplx_p0["id"].tolist()
indexes_ambiguous = sort_pplx_p1["id"].tolist()
indexes_easy_to_learn = sort_pplx_p0_d["id"].tolist()

slice_length = int(len(indexes_hard_to_learn) * (args.percentage / 100))

low_hard_to_learn = sort_pplx_p0[:slice_length]
low_ambiguous = sort_pplx_p1[:slice_length]
low_easy_to_learn = sort_pplx_p0_d[:slice_length]

with open(f"{processed_dir}{args.percentage}_hard_to_learn-{MODEL_NAME}_{args.dataset}.json", "w") as outfile:
    for index, row in low_hard_to_learn.iterrows():     
        data = {"dataset": row['dataset'], "id": row['id'], "messages": row['messages'], "confidence": row['confidence'], "variability":row['variability']}
        json.dump(data, outfile)
        outfile.write('\n')
with open(f"{processed_dir}{args.percentage}_ambiguous-{MODEL_NAME}_{args.dataset}.json", "w") as outfile:
    for index, row in low_ambiguous.iterrows():
        data = {"dataset": row['dataset'], "id": row['id'], "messages": row['messages'],  "confidence": row['confidence'], "variability":row['variability']}
        json.dump(data, outfile)
        outfile.write('\n')

with open(f"{processed_dir}{args.percentage}_easy_to_learn-{MODEL_NAME}_{args.dataset}.json", "w") as outfile:
    for index, row in low_easy_to_learn.iterrows():
        data = {"dataset": row['dataset'], "id": row['id'], "messages": row['messages'],  "confidence": row['confidence'], "variability":row['variability']}
        json.dump(data, outfile)
        outfile.write('\n')


