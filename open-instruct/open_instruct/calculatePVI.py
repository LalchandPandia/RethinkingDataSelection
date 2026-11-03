import torch
import json

from minicons import scorer
import argparse

device='cuda'


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
dataset = args.dataset
model_name = args.model_name

model_dir = f'/path/to/scratch/open_instruct/output/{dataset}_100_{model_name}_full'
null_model_dir = f'/path/to/scratch/open_instruct/output/{dataset}_100_{model_name}_full_without_input'
    
def encode_with_prompt_completion_format_minicons(example, tokenizer):
  
    messages = example["messages"]
    for msg in messages:
        if msg['role'] == 'user':
            question = msg['content']
        if msg['role'] == 'assistant':
            answer = msg['content']

    
    answer = answer + tokenizer.eos_token
    return question, answer


data_file = f"/path/to/scratch/data/processed/{dataset}_train_options_answer.json"

data_file_no_input= f"/path/to/scratch/data/processed/{dataset}_train_options_answer_without_input.json"
pvi_scores = {}
def loss_per_sample(logits, labels):

    BATCH_SIZE = logits.size(dim=0)
    VOCAB_SIZE = logits.size(dim=-1)
    #print('BATCH_SIZE ',BATCH_SIZE)
    #print('VOCAB_SIZE ',VOCAB_SIZE)

    # move labels to correct device to enable model parallelism
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
    #print('Luke sample_loss ',sample_loss)
    #sample_perplexity = [torch.exp(s).item() for s in sample_loss]
    #sample_likelihood = [torch.exp(-s).item() for s in sample_loss]

    # labels[i] maps to sample_loss[i] and sample_perplexity[i]
    return sample_loss
def evaluate(model_dir, data_file):
    lm  = scorer.IncrementalLMScorer(model_dir, device=device)
    
    lm.model.eval()
    
    scores = {}
    f = open(data_file)
    for line in f:
        d = json.loads(line)
        question, answer = encode_with_prompt_completion_format_minicons(d, lm.tokenizer)
        prefixes = [
            question,
        ]

        # these can be your answers
        continuations = [
            answer,
        ]
        likelihood = lm.conditional_score(prefixes, continuations, reduction=lambda x: x.sum().item())
        scores[str(d['id'])] = likelihood[0]
          
    del lm.model
    return scores


pvi_scores = {}
log_prob_g_prime = evaluate(model_dir, data_file)
log_prob_g = evaluate(null_model_dir, data_file_no_input)
for ids in log_prob_g_prime.keys():
    print('ids ',ids)
    pvi = -log_prob_g[ids] + log_prob_g_prime[ids]
    pvi_scores[int(ids)] = pvi

with open(f"/path/to/scratch/data/processed/llama3/{dataset}/pvi_{model_name}/100_pvi_{model_name}_{dataset}.json", "w") as f:
    for key, pvi in pvi_scores.items():
        json.dump({'id':key,f'pvi_{model_name}':float(pvi)}, f)
        f.write('\n')
    
    


