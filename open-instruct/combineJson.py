import pandas as pd
from functools import reduce
main_file_name = '/path/to/scratch/data/processed/siqa_train_options_answer.json'
main_file_name='/path/to/scratch/data/processed/entityTracking_train_options_answer_final.json'

'''
The following list contains path of dataset with opt 125M specific cartography, pvi, vog, p0 
'''
file_list = ['/path/to/scratch/data/processed/llama3/entityTracking/low_confidence/100_hard_to_learn_entityTracking_opt_125M.json', '/path/to/scratch/data/processed/llama3/entityTracking/low_pvi/100_low_pvi_entityTracking_opt_125M_train_options_answer.json',  '/path/to/scratch/data/processed/llama3/entityTracking/high_vog/100_high_vog_entityTracking_opt_125M_train_options_answer.json', '/path/to/scratch/data/processed/llama3/entityTracking/perplexity_p0_opt_125M/100_perplexity_p0_opt_125M_entityTracking_train_options_answer.json']

'''
The following list contains path of dataset with llama 3 specific cartography, pvi, vog, p0 
'''
file_list_llama3 = ['/path/to/scratch/data/processed/llama3/entityTracking/high_p0/100_high_p0_entityTracking.json', '/path/to/scratch/data/processed/llama3/entityTracking/high_vog/100_high_vog_entityTracking_llama3_train_options_answer.json',  '/path/to/scratch/data/processed/llama3/entityTracking/low_confidence/100_hard_to_learn_entityTracking_llama3.json', '/path/to/scratch/data/processed/llama3/entityTracking/low_pvi/100_low_pvi_entityTracking_llama3_train_options_answer.json'] 
def renameColumn(dataFrame, flag):
    dataFrame.rename(columns={"likelihood": f"likelihood_{flag}", "perplexity": f"perplexity_{flag}", "confidence": f"confidence_{flag}", "variability": f"variability_{flag}", "vog_score":f"vog_score_{flag}", "pvi":f"pvi_{flag}"}, inplace=True)
    return dataFrame

# Read JSON Lines files
main_file_name_df = pd.read_json(main_file_name, lines=True)

cartography_opt_125M_df = pd.read_json(file_list[0], lines=True)
cartography_opt_125M_df = cartography_opt_125M_df[["id", "confidence_opt_125M", "variability_opt_125M"]]
pvi_opt_125M_df = pd.read_json(file_list[1], lines=True)
vog_opt_125M_df = pd.read_json(file_list[2], lines=True)
high_p0_opt_125M_df = pd.read_json(file_list[3], lines=True)
high_p0_opt_125M_df  = high_p0_opt_125M_df[["id","perplexity", "likelihood"]]
#high_p1_opt_125M_df = pd.read_json(file_list[4], lines=True)
#high_p1_opt_125M_df = high_p1_opt_125M_df[["id","perplexity", "likelihood"]]
#high_p2_opt_125M_df = pd.read_json(file_list[5], lines=True)
#high_p2_opt_125M_df = high_p2_opt_125M_df[["id","perplexity", "likelihood"]]
#high_p3_opt_125M_df = pd.read_json(file_list[6], lines=True)
#high_p3_opt_125M_df = high_p3_opt_125M_df[["id","perplexity", "likelihood"]]

#cartography_opt_125M_df = renameColumn(cartography_opt_125M_df, 'opt_125M')
pvi_opt_125M_df = renameColumn(pvi_opt_125M_df, 'opt_125M')
vog_opt_125M_df = renameColumn(vog_opt_125M_df, 'opt_125M')
high_p0_opt_125M_df = renameColumn(high_p0_opt_125M_df, 'p0_opt_125M')
#high_p1_opt_125M_df = renameColumn(high_p1_opt_125M_df, 'p1_opt_125M')
#high_p2_opt_125M_df = renameColumn(high_p2_opt_125M_df, 'p2_opt_125M')
#high_p3_opt_125M_df = renameColumn(high_p3_opt_125M_df, 'p3_opt_125M')

#for llama3 df
high_p0_llama_3_df = pd.read_json(file_list_llama3[0], lines=True)[["id","perplexity", "likelihood"]]
cartography_llama_3_df = pd.read_json(file_list_llama3[2], lines=True)
pvi_llama_3_df = pd.read_json(file_list_llama3[3], lines=True)
pvi_llama_3_df = renameColumn(pvi_llama_3_df, 'llama_3')
vog_llama_3_df = pd.read_json(file_list_llama3[1], lines=True)
vog_llama_3_df = renameColumn(vog_llama_3_df, 'llama_3')

dfs = [main_file_name_df, cartography_opt_125M_df, pvi_opt_125M_df, vog_opt_125M_df, high_p0_opt_125M_df, high_p0_llama_3_df,  pvi_llama_3_df, vog_llama_3_df, cartography_llama_3_df]
#dfs = [main_file_name_df, high_p0_llama_3_df, vog_llama_3_df, cartography_llama_3_df]#, pvi_llama_3_df]
# Merge on 'id'
merged_df = reduce(lambda left, right: pd.merge(left, right, on='id', how='inner'), dfs)
#merged_df = pd.merge(main_file_name_df, cartography_opt_125M_df, pvi_opt_125M_df, vog_opt_125M_df, high_p0_opt_125M_df, high_p1_opt_125M_df, high_p2_opt_125M_df, high_p3_opt_125M_df,  on='id', how='inner')  # Change 'how' if needed

# Save back to JSON Lines
merged_df.to_json('/path/to/scratch/data/processed/llama3/entityTracking/100_all_scores_entityTracking_train_options_answer_merged.json', orient='records', lines=True)

