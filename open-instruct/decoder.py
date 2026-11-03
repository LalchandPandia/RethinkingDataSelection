import numpy as np, json



class TrainingDynamics:
    def __init__(self, id, logits, golds):
        self.id = id
        self.logits = logits
        self.golds = golds
    def getId(self):
        return self.id
    def getLogits(self):
        return np.asarray(self.logits)
    def getGolds(self):
        return np.asarray(self.golds)
    def setId(self):
        self.id = self.id
    def setLogits(self):
        self.logits = self.logits.tolist()
    def setGolds(self):
        self.golds = self.golds.tolist()

def TrainingDynamicsDecoder(obj):
    
    return TrainingDynamics(obj['id'], obj['logits'], obj['golds'])
    
import re
def fix_invalid_numbers(line):
    # Regex to find invalid numbers like -1. and replace them with -1.0
    return re.sub(r"(-?\d+)\.(?!\d)", r"\1.0", line)
def fix_empty_values(line):
    # Fix empty values like `,,` or trailing commas
    return re.sub(r",,", r",0.0,", line)
    return re.sub(r",\s*([\]}])", r",0.0\1", line)
count = 0
f = open('/path/to/scratch/open_instruct/dump_pplxs/epoch-0-train_data.json')
for i, line in enumerate(f):
    line = fix_empty_values(line)
    line = fix_invalid_numbers(line.strip())
    try:
        d = json.loads(line, object_hook=TrainingDynamicsDecoder)
    except json.JSONDecodeError as e:
        print(f"Error on line {i}: {e}")
        count+=1
        continue
print('count ',count)
    
    
