from datasets import load_dataset
import json
dataset = load_dataset('openlifescienceai/medmcqa')

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

def medMCQAFormatter(example, input_flag=True):
    if example['choice_type'] !='single':
        return None
    choices = [example[f"op{chr(i+ord('a'))}"] for i in range(4)]
    num_choices = len(choices)
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"][:num_choices]
    question = example['question']
    
    label_index = example["cop"]
    
    answer = choices[label_index]
    if input_flag:
        query = make_mcq_prompt(question, choices, question_prefix="")
    else:
        query = make_mcq_prompt_without_input(question, choices, question_prefix="")
    gold = choice_labels[label_index] +". "+answer
    value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
    return value
for split in ['train', 'test','validation']:
    count = 0
    with open(f'100_medmcqa_{split}.json', 'w') as f:
        
        for data in dataset[split]:
            message = medMCQAFormatter(data)
            if message is None:
                continue
            new_data = {'dataset':'medmcqa', 'id':count, 'messages':message}
            count+=1
            json.dump(new_data, f)
            f.write('\n')
            
    