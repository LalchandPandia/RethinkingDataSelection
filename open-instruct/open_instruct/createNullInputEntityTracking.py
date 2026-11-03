import json
f = open('/path/to/scratch/data/processed/llama3/entityTracking/high_p0/100_high_p0_entityTracking.json')
f1 = open('/path/to/scratch/data/processed/llama3/entityTracking/low_pvi/100_low_pvi_entityTracking_nullInput.json', 'w')
keys_to_exclude = ['perplexity', 'likelihood', 'count']
for line in f:
    data = json.loads(line)
    new_data = {key: value for key, value in data.items() if key not in keys_to_exclude}
    
    null_input = data['messages'][0]['content']
    last_sentence = null_input.strip().split('.')[-1].strip()
    new_data['messages'][0]['content'] = last_sentence
    json.dump(new_data, f1)
    f1.write('\n')
