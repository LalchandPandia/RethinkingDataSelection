'''
This code calculates the mean value embedding from the last layer given a sentence
'''
from datasets import load_dataset
import random, os, math, json
import transformers
import torch
from functools import partial
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    OPTForCausalLM,
    get_scheduler,
)
import time
from utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_datasets,
    get_last_checkpoint_path,
)

import argparse
parser = argparse.ArgumentParser(
    description="Process perplexities csv files into LP(1) subsets."
)
parser._action_groups.pop()
required = parser.add_argument_group("Required Arguments")
optional = parser.add_argument_group("Optional Arguments")


required.add_argument(
    "--model_name_or_path",
    help="model_name_or_path.",
    required=True,
)
required.add_argument(
    "--base_tokenizer_name_or_path",
    help="base_tokenizer_name_or_path",
    required=True,
)
required.add_argument(
    "--pooling",
    help="pooling type",
    required=True,
)

required.add_argument(
    "--train_file",
    help="train file",
    required=True,
)
required.add_argument(
    "--output_file",
    help="output file",
    required=True,
)

args = parser.parse_args()
per_device_train_batch_size = 1

slurm_cpus = os.getenv("SLURM_CPUS_ON_NODE") or os.getenv("SLURM_CPUS_PER_TASK")
if slurm_cpus is not None:
    print('slurm_cpus ',slurm_cpus)
    args_preprocessing_num_workers = max(1, int(int(slurm_cpus) * 1))
else:
    import multiprocessing
    args_preprocessing_num_workers = max(1, int(multiprocessing.cpu_count() * 1))
    
def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    
    messages = example["messages"]
    ids = int(example['id'])
    for msg in messages:
        if msg['role'] == 'user':
            question = msg['content']
        if msg['role'] == 'assistant':
            answer = msg['content']
            
    if not question.endswith((' ', '\n', '\t')) and not answer.startswith((' ', '\n', '\t')):
        example_text = question + ' ' + answer
    else:
        example_text = question + answer
    example_text = example_text + tokenizer.eos_token
    #print('example_text ',example_text)
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(question, return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
        'ids': torch.tensor(ids)
    }

model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            low_cpu_mem_usage=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

device='cuda'
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer_name_or_path,trust_remote_code=False,use_fast=False,)


if isinstance(tokenizer, GPT2Tokenizer):
        # fix to allow OLMO training for subset selection
        #num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "</s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
else:
    #for llama 8B add pad token
    num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if num_added_tokens == 0:
        print("Note: pad_token was already present or shared with eos_token, no new token added.")
    else:
        print("pad_token added successfully.")
    ###adjust embeddings due to new tokens        
print(tokenizer)
model.resize_token_embeddings(len(tokenizer))
if num_added_tokens > 0:
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_added_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_added_tokens].mean(dim=0, keepdim=True)

    input_embeddings[-num_added_tokens:] = input_embeddings_avg
    output_embeddings[-num_added_tokens:] = output_embeddings_avg

train_file = args.train_file
train_dataset = load_dataset("json", data_files=train_file)
train_dataset = train_dataset["train"]
train_dataset = train_dataset.map(partial(encode_with_prompt_completion_format, tokenizer=tokenizer, max_seq_length=512), batched=False,
                                    remove_columns=[name for name in train_dataset.column_names if name not in ["input_ids", "labels", "attention_mask", "dataset", "ids"]
        ], load_from_cache_file=False, num_proc=args_preprocessing_num_workers,
        desc="Tokenizing and reformatting instruction data",)


print("After  map:", train_dataset.column_names)
train_dataset.set_format(type="pt")
train_dataset = train_dataset.filter(lambda example: (example["labels"] != -100).any())



train_dataset.set_format(type="pt", columns=["input_ids", "attention_mask", "labels", "ids"])
train_dataloader = DataLoader(
    train_dataset,
    shuffle=False, # Shuffle doesn't affect length
    collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
    batch_size=per_device_train_batch_size,
)

def mean_pool(hidden_states, attention_mask):
    B, _S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    print(pooled_outputs.shape)
    return pooled_outputs

all_train_embeds = {}
all_train_value_embeds = []
for index, train_inputs in enumerate(tqdm(train_dataloader)):
    with torch.no_grad():            
        passage_embeddings = model(input_ids=train_inputs['input_ids'].cuda(), attention_mask=train_inputs['attention_mask'].cuda(), output_hidden_states=True)
        hidden_states = passage_embeddings.hidden_states
        last_hidden = hidden_states[-1]
        if args.pooling == 'mean':
            normalized_embeddings = mean_pool(hidden_states=last_hidden,attention_mask=train_inputs['attention_mask'].cuda())
        elif args.pooling == 'last':
            
            attention_mask = train_inputs['attention_mask'].unsqueeze(-1).cuda()
            seq_lengths = attention_mask.sum(dim=1) - 1 
            batch_size = last_hidden.size(0)
            normalized_embeddings = last_hidden[torch.arange(batch_size), seq_lengths]
            #normalized_embeddings = F.normalize(last_token_embeddings, p=2, dim=1)      
        for  batch_index, _ in enumerate(train_inputs['ids']):
            all_train_embeds[str(train_inputs['ids'][batch_index].cpu().numpy())] = normalized_embeddings[batch_index].detach().cpu()

with open(args.output_file, "wb") as f:
    torch.save(all_train_embeds, f)
    
    