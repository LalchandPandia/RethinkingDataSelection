import torch
import json
import argparse
from datasets import load_dataset

from vllm import LLM, SamplingParams
parser = argparse.ArgumentParser()
parser.add_argument("--model", default='', type=str)
parser.add_argument("--dtype", default='bfloat16', type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--input_test_file", type=str)
args = parser.parse_args()
model = args.model
print('model ',model)
model_max_length = 2048
llm = LLM(model=model,  tensor_parallel_size=torch.cuda.device_count(), dtype=args.dtype, trust_remote_code=True)
tokenizer = llm.get_tokenizer()
print(tokenizer.eos_token)
print('type eos ',repr(tokenizer.eos_token))
sampling_params = SamplingParams(temperature=0, top_p=1,max_tokens=model_max_length, stop=[tokenizer.eos_token])
print(' sampling_params ',sampling_params.max_tokens)
def encode_with_prompt_completion_format(example, tokenizer):
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
            
    
    answer = answer + tokenizer.eos_token
    
    return question, answer

def run_question_answer(question):
    outputs = llm.generate(question, sampling_params)
    
    outputs = [output.outputs[0].text for output in outputs]
    return outputs

#dataset = load_dataset('allenai/social_i_qa')['validation']
def datasetFromJson(file_name):
    data_files = {}
    dataset_args = {}
    data_files["train"] = file_name
    raw_datasets = load_dataset("json",data_files=data_files,**dataset_args,)['train']
    return raw_datasets
def make_cloze_prompt(question, question_prefix="Question: ", answer_prefix="Answer:"):
    prompt = f"{question_prefix}{question}\n{answer_prefix}"
    return prompt

def convert_int_to_letter(index):
    return chr(index + ord('A'))
def make_mcq_prompt(
    question,
    choices,
    question_prefix="Question: ",
    choices_prefix="",
    answer_prefix="Answer:",
    label_prefix=" ",
    label_format=None,
):
    choice_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_format = label_format or label_prefix + "A."
    if "A" not in label_format:
        raise ValueError(
            "Label format must contain 'A' to indicate where to place the answer label"
        )
    # Use prefix space before each answer label to avoid ambiguity in tokenization
    choices_text = "\n".join(
        [
            f"{label_format.replace('A', label)} {text}"
            for label, text in zip(choice_labels, choices)
        ]
    )
    prompt = f"{question_prefix}{question}\n{choices_prefix}{choices_text}\n{answer_prefix}"
    return prompt, choices_text

def _process_doc_siqa_mcf(doc):
        print('doc ',doc)
        choices = [doc["answerA"], doc["answerB"], doc["answerC"]]
        num_choices = len(choices)
        choice_labels = ["A", "B", "C", "D", "E"][:num_choices]
        choice_answers = []
        for i, choice in enumerate(choices):
            #choice_answers = choice_labels[i]+ '. '+choice
            choice_answers.append(choice)
        query, choices_text = make_mcq_prompt(doc["context"] + " " + doc["question"], choice_answers)
        #gold = convert_int_to_letter(int(doc["label"]) - 1) +". "+choices[int(doc["label"]) - 1]
        gold = convert_int_to_letter(int(doc["label"]) - 1)
        return query, choices_text, gold
def _process_doc_arc_mcf(doc):
        num_choices = len(doc["choices"]["text"])
        choice_labels = ["A", "B", "C", "D", "E"][:num_choices]
        choices = doc['choices']
        query, choices_text  = make_mcq_prompt(doc["question"], choices['text'])
        gold = doc["answerKey"] +". "+doc['choices']['text'][doc['choices']['label'].index(doc['answerKey'])]
        return query, choices_text, gold
def _process_doc_EntityTracking_mcf(example):
    messages = example["messages"]

    for msg in messages:
        if msg['role'] == 'user':
            question = msg['content']
        if msg['role'] == 'assistant':
            answer = msg['content']
    return question, answer

def _process_doc_mcf(example):
    messages = example["messages"]

    for msg in messages:
        if msg['role'] == 'user':
            question = msg['content']
        if msg['role'] == 'assistant':
            answer = msg['content']
    return question, answer

def mcfGreedy(dataset):
    f1 = open(f"output_dir/{args.output_file}", 'w')
    count = 0
    correct = 0
    zero_shot_correct = 0
    for i, data in enumerate(dataset):
        count+=1
        #question, choices, gold = _process_doc_siqa_mcf(data)
        question,  gold = _process_doc_mcf(data)
        outputs = run_question_answer(question)
        data = {}
        data['id'] = count
        data['question'] = question
        '''
        We do not need choice list for ARC as options are part of question
        choice_list = []
        for choice in choices.split('\n'):
            choice_list.append(choice.rstrip().lstrip())
        data['choices'] = {}
        for j, choice in enumerate(choice_list):
            data['choices'][str(j)] = choice
        '''
        data['target'] = gold.rstrip().lstrip()
        data['prediction'] = outputs[0].rstrip().lstrip()
        
        if data['prediction'].strip()[:1].upper() == data['target'].strip()[:1].upper():
            zero_shot_correct += 1
        
        if data['prediction'].upper() == data['target'].upper():
            correct+=1
            
        if data['prediction'].strip()[:1].upper() == data['target'].strip()[:1].upper() and data['prediction'].upper() != data['target'].upper():
            print('zero shot correct but')
            print('data ',data)
        if data['prediction'].strip()[:1].upper() != data['target'].strip()[:1].upper() and data['prediction'].upper() == data['target'].upper():
            print('zero shot wrong but')
            print('data ',data)

        json.dump(data, f1)
        f1.write('\n')
    print('accuracy ',correct/count)
    print('accuracy without looking at answer text',zero_shot_correct/count)
def mcfGreedyEntityTracking(dataset):
    f1 = open(f"output_dir/{args.output_file}", 'w')

    count = 0
    correct = 0
    for i, example in enumerate(dataset):
        count+=1
        question, answer = _process_doc_EntityTracking_mcf(example)
        outputs = run_question_answer(question)
        data = {}
        data['id'] = count
        data['question'] = question
        data['target'] = answer
        data['prediction'] = outputs[0].rstrip().lstrip()
        data['numops'] = example['numops']
        print('data ',data)
        if data['prediction'] == data['target']:
            correct+=1
        json.dump(data, f1)
        f1.write('\n')
    print('accuracy ',correct/count)

file_name=args.input_test_file
#file_name='/path/to/scratch/data/processed/siqa_validation_options_answer_without_answer_text.json'
#file_name='/path/to/scratch/data/processed/siqa_validation_options_answer.json'
#file_name='/path/to/scratch/data/processed/entityTracking_test_options_answer_final.json'
#dataset = datasetFromJson(file_name)
#dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")['test']
dataset = datasetFromJson(file_name)
#mcfGreedyEntityTracking(dataset)
mcfGreedy(dataset)




