'''
This file is used to sort the data on the basis of data selection criteria
And Create files with different % of the full data
'''
import json
import argparse, sys

parser = argparse.ArgumentParser(
    description="Partition data file into  subsets."
)
parser._action_groups.pop()
required = parser.add_argument_group("Required Arguments")
required.add_argument(
    "--dataset",
    help="The dataset for which the perplexities were dumped.",
    required=True,
)
required.add_argument(
    "--metric",
    help="The sub directory where files are stored.",
    required=True,
)
required.add_argument(
    "--main_file",
    help="The file containing the full data along with its utility score.",
    required=True,
)
args = parser.parse_args()
dataset = args.dataset
metric = args.metric
main_file= args.main_file

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
#main_file = "/path/to/scratch/data/processed/faith_or_fate_multiply.jsonl"
#main_file = "/path/to/scratch/data/processed/siqa_option.jsonl"
#main_file = f"/path/to/scratch/data/processed/llama2/arc_easy/p0/100_high_p0_arc_easy_new.json"
def merge_json_files(file1, file2, output_file):
    # Load the first JSON file
    data1 = load_json(file1)

    # Load the second JSON file
    data2 = load_json(file2)

    # Convert second dataset into a dictionary for quick lookup
    data2_dict = {item['id']: item for item in data2}

    # Merge based on 'id'
    merged_data = []
    for item in data1:
        id_value = item['id']
        if id_value in data2_dict:
            merged_item = {**item, **data2_dict[id_value]}
            merged_data.append(merged_item)

    # Save the merged output
    save_json(output_file, merged_data)

    print(f"Merged data saved to {output_file}")

'''
{"dataset":"siqa",
"id":9,
"messages":[
    {"role":"user","content":"Question: Kendall ran back and thanked Lee for helping her find the dog. How would you describe Kendall?\n A. grateful\n B. super\n C. amazing\nAnswer:"},
    {"role":"assistant","content":"A. grateful"}],
    "confidence_opt_125M":0.9468588034,
    "variability_opt_125M":0.0246701533,
    "pvi_opt_125M":7.2489280701,
    "vog_score_opt_125M":0.0006866455,
    "perplexity_p0_opt_125M":12.8191337585,
    "likelihood_p0_opt_125M":0.078008391,
    "perplexity_p1_opt_125M":1.0939036608,
    "likelihood_p1_opt_125M":0.9141572714,
    "perplexity_p2_opt_125M":1.0496709347,
    "likelihood_p2_opt_125M":0.9526795149,
    "perplexity_p3_opt_125M":1.0269685984,
    "likelihood_p3_opt_125M":0.973739624,
    "pvi_llama_3":7.7086410522,
    "vog_score_llama_3":0.0068664551,
    "confidence_llama_3":0.9987423817,
    "variability_llama_3":0.0009263513}
'''

output_dir=f'/path/to/scratch/data/processed/llama3/{dataset}/{metric}'


def process_json(input_file):
    data = load_json(input_file)
    
    # Sort data by decreasing order of perplexity
    #for dataset cartography sort in increasing order of perplexity
    descending = True
    if metric in ["confidence_llama_3", "confidence_opt_125M", "pvi_opt_125M", "pvi_llama_3"]:
        descending = False
    sorted_data = sorted(data, key=lambda x: x[metric], reverse=descending)
    
    total_records = len(sorted_data)
    
    # Create 9 separate files for top 10%, 20%, ..., 90%
    splits = [1, 5, 10, 25, 50, 75, 100]
    for i in splits:
        cutoff = int((i / 100) * total_records)
        subset = sorted_data[:cutoff]
        output_filename = f'{output_dir}/{i}_{metric}_{dataset}.json'
        save_json(output_filename, subset)
        print(f"Saved {output_filename} with {len(subset)} records")

process_json(main_file)
