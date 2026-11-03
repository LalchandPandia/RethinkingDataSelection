'''
This file contains the code to do random selection
'''


import torch, random
import argparse

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict

import json
parser = argparse.ArgumentParser(
    description="Process perplexities csv files into LP(1) subsets."
)
parser._action_groups.pop()
required = parser.add_argument_group("Required Arguments")
optional = parser.add_argument_group("Optional Arguments")


required.add_argument(
    "--dataset",
    help="percentage.",
    required=True,
)

required.add_argument(
    "--seed",
    help="random seed.",
    required=True,
)

args = parser.parse_args()
dataset = args.dataset
seed_value = args.seed
metric = 'random'
input_dir = f'/path/to/scratch/data/processed/llama3/{dataset}/{metric}/'

# Load the JSON file
def load_json(filename):
    list_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            list_data.append(data)
    return list_data

# Save the sorted data into separate files
def save_json(filename, data):
    with open(filename, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')
input_name = f'/path/to/scratch/data/processed/{dataset}_train_options_answer.json'

random.seed(seed_value)

def process_json(input_file):
    data = load_json(input_file)
    total_records = len(data)

    # Create 9 separate files for top 10%, 20%, ..., 90%
    splits = [1, 5, 10, 25, 50, 75, 100]
    for i in splits:
        sample_size = int((i / 100) * total_records)
        # Shuffle the data before sampling
        random.shuffle(data)
        total_records = len(data)
        sampled_data = random.sample(data, sample_size)
        output_filename = f'/path/to/scratch/data/processed/llama3/{dataset}/random_{seed_value}/{i}_{metric}_{dataset}.json'
        save_json(output_filename, sampled_data)
        print(f"Saved {output_filename} with {len(sampled_data)} records")


if __name__ == "__main__":
    process_json(input_name)
