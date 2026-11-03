'''
For this code I had to create new conda with 
torch.__version__
'2.2.0+cu118'
transformers.__version__
'4.42.4'

'''
import torch, json
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModel
from functools import partial
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser(
    description="Process perplexities csv files into LP(1) subsets."
)
parser._action_groups.pop()
required = parser.add_argument_group("Required Arguments")
optional = parser.add_argument_group("Optional Arguments")
required.add_argument(
    "--train_file",
    help="train file",
    required=True,
)


args = parser.parse_args()




def collate_fn(batch):
    ids = [item['id'] for item in batch]
    texts = [item['text'] for item in batch]
    return ids, texts

def encode_with_prompt_completion_format(example):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    
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
    
    return {"text": example_text.strip(), "id": example["id"]}


model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
device='cuda'
model = model.to(device)
model.eval()
max_length = 512
passage_prefix = ""
embed = {}
train_dataset = load_dataset("json", data_files=args.train_file)["train"]
print('before map ',train_dataset.column_names)
train_dataset = train_dataset.map(partial(encode_with_prompt_completion_format), load_from_cache_file=False, num_proc=16)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

with torch.no_grad():
    for batch_ids, batch_texts in tqdm(train_dataloader, desc="Embedding"):
        batch_embeddings = model.encode(list(batch_texts), instruction=passage_prefix, max_length=max_length)
        print('Luke batch_embeddings ',batch_embeddings)
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        for i, eid in enumerate(batch_ids):
            #L-2 normalize the unit vector
            embed[str(eid)] = batch_embeddings[i]
torch.save(
    embed, "/path/to/scratch/embedding_nvidia.pt")