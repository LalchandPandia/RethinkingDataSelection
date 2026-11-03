'''
This file is used to calculate perplexity and likelihood using minicons
'''
import torch
from minicons import scorer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    OPTForCausalLM,
    get_scheduler,
)
from functools import partial
from typing import List, Optional, Union
from typing import Dict, Optional, Sequence
import json

import argparse

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
    "--metric",
    help="The metric.",
    required=True,
)
required.add_argument(
    "--epoch",
    help="The epoch for which the likelihood is dumped.",
    required=True,
)
required.add_argument(
    "--model_name",
    help="The model  for which the likelihood is dumped.",
    required=True,
)

args = parser.parse_args()
epoch = args.epoch
device = 'cuda'
dataset = args.dataset
metric = args.metric
model_prefix = {'facebook/opt-125M':'opt_125M', 'meta-llama/Meta-Llama-3-8B':'llama_3'}
input_dir = f'/path/to/scratch/data/processed/llama3/{dataset}/{metric}/'
if int(epoch) ==  -1:
    model_name=args.model_name
else:
    model_name=f'/path/to/scratch/open_instruct/output/{dataset}_100_{model_prefix[args.model_name]}_full/epoch_{epoch}/'
lm = scorer.IncrementalLMScorer(model_name, device=device)#, torch_dtype=torch.bfloat16)
print(lm.model.dtype)

f1 = open(f'{input_dir}100_{metric}_{dataset}.json','w')

input_name = f'/path/to/scratch/data/processed/{dataset}_train_options_answer.json'

def encode_with_prompt_completion_format_minicons(example, tokenizer):
  
    messages = example["messages"]
    for msg in messages:
        if msg['role'] == 'user':
            question = msg['content']
        if msg['role'] == 'assistant':
            answer = msg['content']

    
    answer = answer + tokenizer.eos_token
    return question, answer
def model_Tokenizer(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path) 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=False,use_fast=False,)

    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,config=config,trust_remote_code=False,low_cpu_mem_usage=True,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
    return tokenizer, model.to(device)


stimuli = []

with open(input_name, "r") as f:
    for line in f:
        d = json.loads(line)
        question, answer = encode_with_prompt_completion_format_minicons(d, lm.tokenizer)
        stimuli.append((question, answer))

batches = DataLoader(stimuli, batch_size=32)
perplexity_list = []
likelihood_list = []
for batch in batches:
    questions = batch[0]
    answers = batch[1]

    likelihood = lm.conditional_score(questions, answers, reduction=lambda x: (x.mean()).exp().item())
    perplexity = lm.conditional_score(questions, answers, reduction=lambda x: (-x.mean()).exp().item())
    perplexity_list.append(perplexity)
    likelihood_list.append(likelihood)
perplexity_list = sum(perplexity_list, [])
likelihood_list = sum(likelihood_list, [])
count = 0
with open(input_name, "r") as f:
    for line in f:
        d = json.loads(line)
        d[metric] = perplexity_list[count]
        d['likelihood'+metric.split('perplexity',1)[1]] = likelihood_list[count]
        count+=1
        json.dump(d, f1)
        f1.write('\n')

