import torch
from transformers import ( AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq)
import deepspeed
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
import json
from functools import partial
from datetime import timedelta
from collections import defaultdict
import os

device='cuda'


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
    "--model_name",
    help="The dataset for which the perplexities were dumped.",
    required=True,
)


args = parser.parse_args()


def token_gradients(model, input_ids, targets):
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    
    valid_positions = (targets != -100).nonzero(as_tuple=True)[0]
    input_slice = slice(0, valid_positions[0].item())
    end_input_slice = valid_positions[-1].item()
    

    #embed_weights = get_embedding_matrix(model)
    embeddings = model.get_input_embeddings()
    #with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
    embedding_weights = embeddings.weight
    ###############################################
    #embeddings.weight.retain_grad()
    #################################################################
    embedding_size = embedding_weights.shape[0]
    #print('embed_weights ',embedding_weights.shape)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embedding_size,
        device=model.device,
        dtype=embeddings.weight.dtype
    )

    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embeddings.weight.dtype)
    )
    one_hot.requires_grad_()

    input_embeds = (one_hot @ embeddings.weight)
    input_embeds.requires_grad_()
    input_embeds.retain_grad()
    input_ids = input_ids.cpu().tolist()
    embeds = embeddings.weight[input_ids[:end_input_slice+1],:]#.detach()

    full_embeds = torch.cat(
        [
            embeds[:input_slice.start,:],
            input_embeds,
            embeds[input_slice.stop:,:]
        ],
        dim=0)
    full_embeds = full_embeds.unsqueeze(0)


    logits = model(inputs_embeds=full_embeds).logits
    #print('logits ',logits)
    loss = torch.nn.CrossEntropyLoss()(logits[0,:,:], targets[:end_input_slice+1])
    model.zero_grad()
    loss.backward()
    
    return one_hot.grad.clone().cpu(), input_embeds.grad.clone().cpu()


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    messages = example["messages"]
    
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
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(question, return_tensors='pt', max_length=max_seq_length, truncation=True)
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    } 


model_name=args.model_name
dataset=args.dataset
model_dir =f'/path/to/scratch/open_instruct/output/{dataset}_100_{model_name}_full'
tokenizer = AutoTokenizer.from_pretrained(model_dir,revision=None,trust_remote_code=False)

data_files = {}
dataset_args = {}
data_files["train"] = f"/path/to/scratch/data/processed/{dataset}_train_options_answer.json"

raw_datasets = load_dataset(
    "json",
    data_files=data_files,
    **dataset_args,
)

train_dataset = raw_datasets["train"]

train_dataset = train_dataset.map(partial(encode_with_prompt_completion_format, tokenizer=tokenizer, max_seq_length=512),
            batched=False,
            remove_columns=[
                name for name in train_dataset.column_names if name not in ["input_ids", "labels", "attention_mask", "id"]
            ],
            desc="Tokenizing and reformatting instruction data",)
train_dataset.set_format(type="pt")
train_dataset = train_dataset.filter(lambda example: (example["labels"] != -100).any())

count_item = 0
gradient = defaultdict(list)
for i in range(0,3):
    model_path = os.path.join(model_dir, f'epoch_{i}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,).to(device)
    eval_dataloader = DataLoader(train_dataset, batch_size=1,shuffle=False,collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),)
    model.eval()
    #with torch.no_grad():
    for step, batch  in enumerate(eval_dataloader):
        new_batch = {}
        new_batch['input_ids'] = batch['input_ids']
        new_batch['attention_mask'] = batch['attention_mask']
        new_batch['labels'] = batch['labels']
        print('batch ',batch)
        for id, input, target in zip(batch['id'], batch['input_ids'], batch['labels']):
            one_hot_grad, embed_grad = token_gradients(model, input, target)
            gradient[id.item()].append(embed_grad)
            if count_item % 500 == 0:
                print(f'done {count_item} examples ')
            count_item+=1
    del model


def compute_vog_scores():
    vog_scores = {}
    v_i_list = []  # Store all v_i for dataset normalization

    for example_id, grad_list in gradient.items():
        G_i = torch.stack(grad_list)  # Shape: (N_c, seq_len, embedding_dim)
        N_c = G_i.shape[0]

        # Compute mean gradient across checkpoints
        mu_i = torch.mean(G_i, dim=0)  # Shape: (seq_len, embedding_dim)

        # Compute variance across checkpoints
        V_i = torch.mean((G_i - mu_i) ** 2, dim=0) / torch.sqrt(torch.tensor(N_c, dtype=torch.float32))  # (seq_len, embedding_dim)

        # Compute mean variance over embeddings (averaging over seq_len and embedding_dim)
        v_i = torch.mean(V_i)  # Scalar
        v_i_list.append(v_i)

        vog_scores[example_id] = v_i


    return vog_scores
vog_scores = compute_vog_scores()

with open(f"/path/to/scratch/data/processed/llama3/{dataset}/vog_score_{model_name}/100_vog_score_{model_name}_{dataset}.json", "w") as f:
    for key, vog in vog_scores.items():
        json.dump({'id':key,f'vog_score_{model_name}':float(vog)}, f)
        f.write('\n')
    
    


