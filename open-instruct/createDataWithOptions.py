'''
This is the initial file which creates the data
'''
import json
import random
import argparse
from datasets import load_dataset

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
    "--split",
    help="The split to be used.",
    required=True,
)
required.add_argument('--input_flag', action='store_true', help='Set this flag to include input')
args = parser.parse_args()
input_flag = args.input_flag
input_dir = '/path/to/scratch/data/processed/'
#input_dir = './open-instruct/'

random.seed(42)
split=args.split
if args.dataset == 'arc_easy':
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")[split]
if args.dataset == 'arc_challenge':
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")[split]
if args.dataset == 'piqa':
    dataset = load_dataset('ybisk/piqa')[split]
if args.dataset == 'siqa':
    dataset = load_dataset('allenai/social_i_qa')[split]
if args.dataset == 'mathInstruct':
    dataset = load_dataset("TIGER-Lab/MathInstruct")['train']
if args.dataset == 'openBookQA':
    dataset = load_dataset('allenai/openbookqa')[split]
print('dataset ',dataset[-1])
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
    return prompt

def make_mcq_prompt_without_input(
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
    prompt = f"{answer_prefix}"
    return prompt

def _process_doc_siqa(doc, input_flag=True):
        choices = [doc["answerA"], doc["answerB"], doc["answerC"]]
        num_choices = len(choices)
        choice_labels = ["A", "B", "C", "D", "E"][:num_choices]
        if input_flag:
            query = make_mcq_prompt(doc["context"] + " " + doc["question"], choices)
        else:
            query = make_mcq_prompt_without_input(doc["context"] + " " + doc["question"], choices)
        gold = convert_int_to_letter(int(doc["label"]) - 1) +". "+choices[int(doc["label"]) - 1]
        #temporary to check if just giving option letter is enough
        gold = convert_int_to_letter(int(doc["label"]) - 1)# +". "+choices[int(doc["label"]) - 1]
        value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
        return value
    
def _process_doc_openBookQA(doc, input_flag=True):
        
        if input_flag:
            query = make_mcq_prompt(doc["question_stem"], doc["choices"]["text"])
        else:
            query = make_mcq_prompt_without_input(doc["question_stem"], doc["choices"]["text"])
        #temporary to check if just giving option letter is enough
        answer_index = ["A", "B", "C", "D"].index(doc["answerKey"].strip())
        gold = doc["answerKey"].strip() +". "+doc["choices"]["text"][answer_index]
        value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
        return value   

def convert_int_to_letter(index):
    return chr(index + ord('A'))

def make_cloze_prompt(question, question_prefix="Question: ", answer_prefix="Answer:"):
    prompt = f"{question_prefix}{question}\n{answer_prefix}"
    return prompt


def _process_doc_arc(doc, input_flag=True):
        '''
        {'id': 'NYSEDREGENTS_2004_8_31', 
        'question': 'A chemical property of a mineral is evident if the mineral',
        'choices': 
        {
            'text': 
                ['breaks easily when struck with a hammer', 
                'bubbles when acid is placed on it', 
                'is easily scratched by a fingernail',
                'reflects light from its surface'],
            'label': ['1', '2', '3', '4']
        },
        'answerKey': 'B'}
        '''
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        original_key = doc["answerKey"]
        doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
        choices = doc['choices'] 
        if input_flag:
            query = make_mcq_prompt(doc["question"], choices['text'])
        else:
            query = make_mcq_prompt_without_input(doc["question"], choices['text'])
        #query = make_mcq_prompt(doc["question"], choices['text'])
       
        gold = doc["answerKey"] +". "+doc['choices']['text'][doc['choices']['label'].index(original_key)]
        value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
        return value
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

PROMPT_DICT_QA_old = {
    "prompt_no_input": (
        "Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter.\n"
        "Question: {instruction}\n\n"
        "Options:\n{option}"
    ),
}
PROMPT_DICT_QA = {
    "prompt_no_input": (
        "Question: {instruction}\n\n"
        "Options:\n{option}"
    ),
}

def format_example_mathInstruct(example):
    prompt_no_input = PROMPT_DICT["prompt_no_input"]
    query = example['instruction']
    answer = example['output']
    value = [{"role":"user","content": prompt_no_input.format(instruction=query)}, {"role":"assistant","content":answer}]
    return value

if args.dataset == 'arc_easy' or args.dataset == 'arc_challenge':
    #format_example = format_example_arc
    format_example = _process_doc_arc

if args.dataset == 'siqa':
    #format_example = format_example_siqa
    #format_example = _process_doc_siqa_cf
    format_example = _process_doc_siqa
if args.dataset == 'openBookQA':
    format_example = _process_doc_openBookQA
if input_flag:
    print('in input_flag')
    f1 = open(f'{input_dir}{args.dataset}_{split}_options_answer.json', 'w') 
else:
    f1 = open(f'{input_dir}{args.dataset}_{split}_options_answer_without_input.json', 'w')
print('input_flag ',input_flag)
for i, data in enumerate(dataset):
    d = {}
    #print(data)
    d = {"dataset": args.dataset, "id": i, "messages": format_example(data, input_flag=input_flag)}
    #print('d ',d)
    json.dump(d,f1)
    f1.write('\n')
    
