'''
This file contains the code to calculate confidence and variance using already trained checkpoints
This is primarily used to calculate cartography baseline
'''
import torch
from transformers import ( AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq)
from datasets import load_dataset
from torch.utils.data import DataLoader
import json
from functools import partial
from collections import defaultdict
import os

device='cuda'



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
    print('example_text ',example_text)
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
    }

model_dir ='/path/to/scratch/open_instruct/output/tulu_v2_7B_100_full_arc_easy'
tokenizer = AutoTokenizer.from_pretrained(model_dir,revision=None,trust_remote_code=False,use_fast=False,)



data_files = {}
dataset_args = {}

data_files["train"] = "/path/to/scratch/data/processed/llama2/arc_easy.jsonl"
raw_datasets = load_dataset(
    "json",
    data_files=data_files,
    **dataset_args,
)

train_dataset = raw_datasets["train"]

train_dataset = train_dataset.map(partial(encode_with_prompt_completion_format, tokenizer=tokenizer, max_seq_length=512, dataset='arc_challenge'),
            batched=False,
            remove_columns=[
                name for name in train_dataset.column_names if name not in ["input_ids", "labels", "attention_mask", "ids"]
            ],
            desc="Tokenizing and reformatting instruction data",)
train_dataset.set_format(type="pt")
train_dataset = train_dataset.filter(lambda example: (example["labels"] != -100).any())

def loss_per_sample(logits, labels):

    BATCH_SIZE = logits.size(dim=0)
    VOCAB_SIZE = logits.size(dim=-1)

    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, VOCAB_SIZE), shift_labels.view(-1))


    loss_tensor_per_sample = torch.tensor_split(loss, BATCH_SIZE)
    
    sample_loss = [torch.sum(tensor)/torch.count_nonzero(tensor)
                    if bool(torch.count_nonzero(tensor) > 0)
                    else torch.tensor(0.0).to(tensor.device)
                    for tensor in loss_tensor_per_sample]

    sample_perplexity = [torch.exp(s).item() for s in sample_loss]
    sample_likelihood = [torch.exp(-s).item() for s in sample_loss]
    return sample_perplexity, sample_likelihood



count_item = 0
gradient = defaultdict(list)
model_name='llama2'
dataset='arc_easy'
checkpointing_steps = 256
for epoch in range(0,3):
    model_path = os.path.join(model_dir, f'epoch_{epoch}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,).to(device)
    eval_dataloader = DataLoader(train_dataset, batch_size=1,shuffle=False,collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),)
    model.eval()
    train_data_ids_per_epoch = []
    train_data_perplexity_per_epoch = []
    train_data_confidence_per_epoch = []
    
    with torch.no_grad():
        for step, batch  in enumerate(eval_dataloader):
            new_batch = {}
            new_batch['input_ids'] = batch['input_ids'].to(device)
            new_batch['attention_mask'] = batch['attention_mask'].to(device)
            new_batch['labels'] = batch['labels'].to(device)
            outputs = model(**new_batch, use_cache=False)
            logits = outputs.logits
            labels = batch["labels"]
            embedding_size = logits.size(dim=-1)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            #loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
            shift_logits = shift_logits.view(-1, embedding_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            #loss = loss_fct(shift_logits, shift_labels)
            #list of perplexities of size BATCH_SIZE
            batch_perplexities, batch_likelihood  = loss_per_sample(logits, labels)
            train_data_ids_per_epoch.append(batch['ids'][0].item())
            #print(batch_likelihood)
            train_data_perplexity_per_epoch.append(batch_perplexities[0])
            train_data_confidence_per_epoch.append(batch_likelihood[0])
            if (step+1) % checkpointing_steps == 0 and step != 0:
                #train_data_ids_per_epoch = sum(train_data_ids_per_epoch, [])
                #train_data_perplexity_per_epoch = sum(train_data_perplexity_per_epoch, [])
                #train_data_confidence_per_epoch = sum(train_data_confidence_per_epoch, [])
                with open(f'/path/to/scratch/data/processed/{model_name}/{dataset}/cartography_opt_125M/epoch-{epoch}-{model_name}_train_{dataset}.json', 'a') as f:
                    for id, likelihood, perplexity in zip(train_data_ids_per_epoch, train_data_confidence_per_epoch, train_data_perplexity_per_epoch):
                        json.dump({'id':id, 'likelihood':likelihood, 'perplexity':perplexity}, f)#, cls=NumpyEncoder)
                        f.write('\n')
                train_data_confidence_per_epoch = []
                train_data_ids_per_epoch = []
                train_data_perplexity_per_epoch = []
        #flush the remaining data
        #train_data_ids_per_epoch = sum(train_data_ids_per_epoch, [])
        #train_data_perplexity_per_epoch = sum(train_data_perplexity_per_epoch, [])
        #train_data_confidence_per_epoch = sum(train_data_confidence_per_epoch, [])
        with open(f'/path/to/scratch/data/processed/{model_name}/{dataset}/cartography_opt_125M/epoch-{epoch}-{model_name}_train_{dataset}.json', 'a') as f:
            for id, likelihood, perplexity in zip(train_data_ids_per_epoch, train_data_confidence_per_epoch, train_data_perplexity_per_epoch):
                json.dump({'id':id, 'likelihood':likelihood, 'perplexity':perplexity}, f)#, cls=NumpyEncoder)
                f.write('\n')
    del model


#with open("/path/to/scratch/data/processed/siqa_dc.jsonl", "w") as f:
#    for key, vog in vog_scores.items():
#        json.dump({'id':key,'vog_score':float(vog)}, f)
#        f.write('\n')
    
    


