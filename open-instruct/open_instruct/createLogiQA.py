from itertools import groupby
import json
input_dir = '/path/to/home/LogiQA-dataset/'
output_dir = './'
def read_lqa(split):

    with open(f"{input_dir}{split}.txt", "r") as f:
        lines = f.readlines()
    

    grouper = groupby(lines, key=lambda x: x in {"\n"})
    ds = dict(enumerate((list(j) for i, j in grouper if not i), 1)).values()
    print(ds)
    count = 0
    with open(f'{output_dir}100_logiqa_{split}.json','w') as f1:
        for row in ds:
            formatted_row = dict()
            formatted_row["answer_idx"] = ord(row[0][0]) - ord("a")
            formatted_row["question"] = f"{row[1].strip()} {row[2].strip()}"
            formatted_row['id'] = count
            choices = list()
            for choice in row[3:]:
                choices.append(choice[2:].strip())
            formatted_row["choices"] = choices
            json.dump(formatted_row, f1)
            f1.write('\n')
            count+=1

read_lqa('Train')
read_lqa('Test')
read_lqa('Eval')
