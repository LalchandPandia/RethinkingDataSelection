import json
with open('/path/to/scratch/open_instruct/dump_pplxs/epoch-0-train_data.json') as f:
    d = json.load(f)
e = []
for item in d:
    s = {}
    for key, value in item.items():
        if key!='confidence':
            s['id'] = value
        else:
            s['confidence'] = value
        e.append(s)

with open('/path/to/scratch/open_instruct/dump_pplxs/epoch-0-train_data_updated.json', 'w') as f:
    for item in e:
        json.dump(item,f)
        f.write('\n')

