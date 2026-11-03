'''
This file calculates the baseline for cartography paper
'''
import json
import pandas as pd
import argparse
import shutil
from functools import reduce

parser = argparse.ArgumentParser(
    description="Process confidence and varaibility"
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
        "--model_name",
    help="Optional model name which is added to filenames of generated subsets.",
    default="model",
)
args = parser.parse_args()
model_name = args.model_name



# List of file paths (adjust with your actual file paths)
model_prefix = {'facebook/opt-125M':'opt_125M', 'meta-llama/Meta-Llama-3-8B':'llama_3'}
input_dir = f'/path/to/scratch/data/processed/llama3/{args.dataset}/'
files = [f'{input_dir}perplexity_p1_{model_prefix[model_name]}/100_perplexity_p1_{model_prefix[model_name]}_{args.dataset}.json',\
    f'{input_dir}perplexity_p2_{model_prefix[model_name]}/100_perplexity_p2_{model_prefix[model_name]}_{args.dataset}.json',\
        f'{input_dir}perplexity_p3_{model_prefix[model_name]}/100_perplexity_p3_{model_prefix[model_name]}_{args.dataset}.json']
#files = [f'{file_dir}high_p1/100_high_p1_{args.dataset}_train_options_answer.json', f'{file_dir}/high_p2/100_high_p2_{args.dataset}_train_options_answer.json', f'{file_dir}/high_p3/100_high_p3_{args.dataset}_train_options_answer.json']
processed_confidence_dir = f'{input_dir}confidence_{model_prefix[model_name]}/'
processed_variability_dir = f'{input_dir}variability_{model_prefix[model_name]}/'

# Read all JSONL files into a list of dataframes
dfs = []
for file in files:
    df = pd.read_json(file, lines=True)  # Read each JSONL file into a DataFrame
    df = df.drop(columns=['dataset', 'messages'])
    dfs.append(df)
    
merged_df = reduce(lambda left, right: pd.merge(left, right, on=["id"]), dfs)

# Step 3: Compute likelihood mean and std
likelihood_cols = [col for col in merged_df.columns if col.startswith('likelihood_')]
merged_df['likelihood_'+model_prefix[model_name]] = merged_df[likelihood_cols].mean(axis=1)
merged_df['likelihood_std_'+model_prefix[model_name]] = merged_df[likelihood_cols].std(axis=1, ddof=0)

merged_df = merged_df[['id', f'likelihood_{model_prefix[model_name]}', f'likelihood_std_{model_prefix[model_name]}']]
merged_df.columns = ['id', f'confidence_{model_prefix[model_name]}', f'variability_{model_prefix[model_name]}']

# Handle the case where the standard deviation is NaN (i.e., only one data point per id)
#result['low variability'] = result['variability'].fillna(0)

output_file = f'{processed_confidence_dir}100_confidence_{model_prefix[model_name]}_{args.dataset}.json'

with open(output_file, 'w') as f:
    for _, row in merged_df.iterrows():
        row_dict = row.to_dict()
        row_dict['id'] = int(row_dict['id'])
        #print('row_dict ',row_dict)
        json.dump(row_dict, f)
        f.write('\n')

print(f"Output saved to {output_file}")
shutil.copy(output_file, f'{processed_variability_dir}100_variability_{model_prefix[model_name]}_{args.dataset}.json')