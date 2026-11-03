from datasets import load_dataset
import json
dataset = load_dataset('mandarjoshi/trivia_qa', 'unfiltered.nocontext')

def make_mcq_prompt(
    question,
    choices="",
    question_prefix="Question: ",
    choices_prefix="",
    answer_prefix="Answer:",
    label_prefix=" ",
    label_format=None,
):
    
    
    prompt = f"{question_prefix}{question}\n{answer_prefix}"
    return prompt

def make_mcq_prompt_without_input(
    question,
    choices="",
    question_prefix="Question: ",
    choices_prefix="",
    answer_prefix="Answer:",
    label_prefix=" ",
    label_format=None,
):
    
    prompt = f"{answer_prefix}"
    return prompt

def triviaQAFormatter(example, input_flag=True):
    
    question = example['question']
    
    
    aliases = example['answer']['aliases']
    answer = example['answer']['value']
    if input_flag:
        query = make_mcq_prompt(question,  question_prefix="")
    else:
        query = make_mcq_prompt_without_input(question,  question_prefix="")
    gold = answer
    value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
    return value, aliases
for split in ['test']:
    count = 0
    with open(f'100_triviqa_{split}.json', 'w') as f:
        
        for data in dataset[split]:
            message, aliases = triviaQAFormatter(data)
            if message is None:
                continue
            new_data = {'dataset':'medmcqa', 'id':count, 'messages':message, 'aliases':aliases}
            count+=1
            json.dump(new_data, f)
            f.write('\n')
            
    