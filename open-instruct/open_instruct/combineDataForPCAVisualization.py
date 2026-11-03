import numpy as np, json, csv
from collections import defaultdict
import os

import argparse
parser = argparse.ArgumentParser(
    description="Process perplexities csv files into LP(1) subsets."
)
parser._action_groups.pop()
required = parser.add_argument_group("Required Arguments")
optional = parser.add_argument_group("Optional Arguments")


required.add_argument(
    "--percentage",
    help="percentage.",
    required=True,
)
required.add_argument(
    "--embedding_pca_file",
    help="embedding pca file.",
    required=True,
)
required.add_argument(
    "--pooling",
    help="pooling.",
    required=True,
)

required.add_argument(
    "--model",
    help="model.",
    required=True,
)

args = parser.parse_args()
percentage = args.percentage
columns = ['confidence_llama_3', 'confidence_opt_125M', 'random', 'perplexity_p0_llama_3', 'perplexity_p0_opt_125M', \
    'pvi_llama_3', 'pvi_opt_125M', 'variability_llama_3', 'variability_opt_125M', 'vog_score_llama_3', 'vog_score_opt_125M']
path_name_dict = {f'{percentage}_confidence_llama_3_siqa.json':'confidence_llama_3',
                  f'{percentage}_confidence_opt_125M_siqa.json':'confidence_opt_125M',\
    f'{percentage}_random_siqa.json':'random', f'{percentage}_perplexity_p0_llama_3_siqa.json':'perplexity_p0_llama_3',\
        f'{percentage}_perplexity_p0_opt_125M_siqa.json':'perplexity_p0_opt_125M', \
    f'{percentage}_pvi_llama_3_siqa.json':'pvi_llama_3',f'{percentage}_pvi_opt_125M_siqa.json': 'pvi_opt_125M',\
        f'{percentage}_variability_llama_3_siqa.json': 'variability_llama_3',\
            f'{percentage}_variability_opt_125M_siqa.json':'variability_opt_125M',\
        f'{percentage}_vog_score_llama_3_siqa.json':'vog_score_llama_3', \
            f'{percentage}_vog_score_opt_125M_siqa.json':'vog_score_opt_125M'}
path_dir = f'./full_data/siqa/{percentage}/'
data_path_dir = f'./full_data/siqa/{percentage}/data/'
ids_dict = defaultdict(list)
for file_name in os.listdir(data_path_dir):
    full_path = os.path.join(data_path_dir, file_name)
    f = open(full_path)
    for line in f:
        data = json.loads(line)
        ids_dict[path_name_dict[file_name]].append(data['id'])
for key in ids_dict:
    ids_dict[key] = set(str(i) for i in ids_dict[key])
print(len(ids_dict['variability_llama_3']))
#keys: ids, pca, variance, variance_ratio
pca_file = np.load(args.embedding_pca_file)
f1 = open(path_dir+f'pca_{args.model}_{args.pooling}.csv','w')
f2 = open(path_dir+f'selection_indicator_{percentage}.csv','w')
field_indicator = columns+['id']
field_writer = csv.DictWriter(f2, delimiter='\t', fieldnames=field_indicator)
field_writer.writeheader()
fields = ['id','pca_1', 'pca_2','pca_3']
pca_writer = csv.DictWriter(f1, delimiter='\t', fieldnames=fields)
pca_writer.writeheader()
ids = pca_file['ids']
pcas = pca_file['pca']
print('variance ',pca_file['variance'])
print('variance_ratio ',pca_file['variance_ratio'])
f3 = open(f'./open-instruct/embeddings/variance_{args.model}_{args.pooling}.csv', 'w')
variance_writer = csv.DictWriter(f3, delimiter='\t', fieldnames=['variance_1', 'variance_2','variance_3', 'variance_ratio_1', 'variance_ratio_2','variance_ratio_3'])
variance_writer.writeheader()
variance_writer.writerow({'variance_1':pca_file['variance'][0],'variance_2':pca_file['variance'][1],\
    'variance_3':pca_file['variance'][2], 'variance_ratio_1':pca_file['variance_ratio'][0],'variance_ratio_2':pca_file['variance_ratio'][1],'variance_ratio_3':pca_file['variance_ratio'][2]})

def idCheck(id, fieldname):
   
    if id in ids_dict[fieldname]:
        return True
    return False
for id_val, pca_vals in zip(ids, pcas):
    pca_writer.writerow({'id':id_val, 'pca_1':pca_vals[0], 'pca_2':pca_vals[1],'pca_3':pca_vals[2]})
    field_writer.writerow({'id':id_val,\
        'confidence_llama_3':idCheck(id_val,'confidence_llama_3'), 'confidence_opt_125M':idCheck(id_val, 'confidence_opt_125M'),\
            'random':idCheck(id_val,'random'), 'perplexity_p0_llama_3':idCheck(id_val, 'perplexity_p0_llama_3'),\
                'perplexity_p0_opt_125M':idCheck(id_val, 'perplexity_p0_opt_125M'), \
    'pvi_llama_3':idCheck(id_val, 'pvi_llama_3'), 'pvi_opt_125M':idCheck(id_val, 'pvi_opt_125M'),
    'variability_llama_3':idCheck(id_val, 'variability_llama_3'), 'variability_opt_125M': idCheck(id_val, 'variability_opt_125M'),\
        'vog_score_llama_3':idCheck(id_val, 'vog_score_llama_3'), 'vog_score_opt_125M':idCheck(id_val, 'vog_score_opt_125M')})    