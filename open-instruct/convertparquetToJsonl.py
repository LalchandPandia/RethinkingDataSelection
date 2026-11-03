import pandas as pd
import json, sys
from datasets import load_dataset
def readPIQA():
    dataset = load_dataset('ybisk/piqa')['train']
    count = 0
    f = open('/path/to/scratch/data/processed/piqa_train.jsonl', 'w')
    for row in dataset:
        d = {}
        d['dataset'] = 'piqa'
        d['id'] = count
        if row['label'] == 0:
            solution = row['sol1']
        else:
            solution = row['sol2']
        d['messages'] = [{'role': 'user', 'content':row['goal']}, {'role':'assistant', 'content':solution}]
        count+=1
        json.dump(d, f)
        f.write('\n')

def readSIQA():
    count = 0
    dataset = load_dataset('allenai/social_i_qa')['train']
    f = open('/path/to/scratch/data/processed/siqa_train.jsonl', 'w')
    for row in dataset:
        d = {}
        d['dataset'] = 'siqa'
        d['id'] = count
        query = row['question']
        context = row['context']
        label = row['label']

        if label == '1':
            answer = row['answerA']
        if label == '2':
            answer = row['answerB']
        if label == '3':
            answer = row['answerC']
        value = [{"role":"user","content":context+' '+query}, {"role":"assistant","content":answer}]
        d['messages'] = value
        count+=1
        json.dump(d, f)
        f.write('\n')
#readSIQA(dataset)
#readPIQA(dataset)
def readARC(variant):
    if variant == 'easy':
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")['train']
    if variant == 'challenge':
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")['train']
    count = 0
    f = open('/path/to/scratch/data/processed/ai2_arc_'+variant+'.jsonl', 'w')
    for row in dataset:
        d = {}
        d['dataset'] = 'arc_'+variant
        d['id'] = count
        query = row['question']
        answer = row['choices']['text'][row['choices']['label'].index(row['answerKey'])]
        value = [{"role":"user","content": query}, {"role":"assistant","content":answer}]
        d['messages'] = value
        count+=1
        json.dump(d, f)
        f.write('\n')
readARC('easy')
readARC('challenge')
readPIQA()
readSIQA()
sys.exit()
print(df)
df = df.reset_index()
f = open('/path/to/scratch/data/processed/piqa_train.jsonl', 'w')
optionToIndex = {'A':0, 'B':1,'C':2, 'D':3,'E':4, '1':0,'2':1,'3':2,'4':3,'5':4}

for index, row in df.iterrows():
    d = {}
    print(row)
    d['dataset'] = 'piqa'
    d['id'] = row['id']
    #d['messages'] = [{'role': 'user','content':row['question']}, {'role':'assistant','content':row['choices']['text'][optionToIndex[row['answerKey']]]}]
    if row['label'] == 0:
        solution = row['sol1']
    else:
        solution = row['sol2']
    d['messages'] = [{'role': 'user', 'content':row['goal']}, {'role':'assistant', 'content':solution}]
    json.dump(d, f)
    f.write('\n')

