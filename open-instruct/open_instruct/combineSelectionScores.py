import argparse
import pandas as pd, os
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
    "--dataset",
    help="percentage.",
    required=True,
)



args = parser.parse_args()
percentage = args.percentage
dataset=args.dataset
merged_df = None

main_file_name = f'/path/to/scratch/data/processed/{args.dataset}_train_options_answer.json'

columns = ['confidence_llama_3', 'confidence_opt_125M',  'perplexity_p0_llama_3', 'perplexity_p0_opt_125M', \
    'pvi_llama_3', 'pvi_opt_125M', 'variability_llama_3', 'variability_opt_125M', 'vog_score_llama_3', 'vog_score_opt_125M']#,'random',]
#temporarily using opt metrics
#columns = [ 'confidence_opt_125M',  'perplexity_p0_opt_125M', \
#     'pvi_opt_125M',  'variability_opt_125M',  'vog_score_opt_125M']#,'random',]
path_name_dict = {f'{percentage}_confidence_llama_3_{dataset}.json':'confidence_llama_3',
                  f'{percentage}_confidence_opt_125M_{dataset}.json':'confidence_opt_125M',\
     f'{percentage}_perplexity_p0_llama_3_{dataset}.json':'perplexity_p0_llama_3',\
        f'{percentage}_perplexity_p0_opt_125M_{dataset}.json':'perplexity_p0_opt_125M', \
    f'{percentage}_pvi_llama_3_{dataset}.json':'pvi_llama_3',f'{percentage}_pvi_opt_125M_{dataset}.json': 'pvi_opt_125M',\
        f'{percentage}_variability_llama_3_{dataset}.json': 'variability_llama_3',\
            f'{percentage}_variability_opt_125M_{dataset}.json':'variability_opt_125M',\
        f'{percentage}_vog_score_llama_3_{dataset}.json':'vog_score_llama_3', \
            f'{percentage}_vog_score_opt_125M_{dataset}.json':'vog_score_opt_125M',f'{percentage}_random_{dataset}.json':'random'}

path_name_dict = {f'{percentage}_confidence_llama_3_{dataset}.json':'confidence_llama_3',
                  f'{percentage}_confidence_opt_125M_{dataset}.json':'confidence_opt_125M',\
     f'{percentage}_perplexity_p0_llama_3_{dataset}.json':'perplexity_p0_llama_3',\
        f'{percentage}_perplexity_p0_opt_125M_{dataset}.json':'perplexity_p0_opt_125M', \
    f'{percentage}_pvi_llama_3_{dataset}.json':'pvi_llama_3',f'{percentage}_pvi_opt_125M_{dataset}.json': 'pvi_opt_125M',\
        f'{percentage}_variability_llama_3_{dataset}.json': 'variability_llama_3',\
            f'{percentage}_variability_opt_125M_{dataset}.json':'variability_opt_125M',\
        f'{percentage}_vog_score_llama_3_{dataset}.json':'vog_score_llama_3', \
            f'{percentage}_vog_score_opt_125M_{dataset}.json':'vog_score_opt_125M',f'{percentage}_random_{dataset}.json':'random'}

path_name_dict = {f'{percentage}_confidence_opt_125M_{dataset}.json':'confidence_opt_125M',\
        f'{percentage}_perplexity_p0_opt_125M_{dataset}.json':'perplexity_p0_opt_125M',\
            f'{percentage}_pvi_opt_125M_{dataset}.json': 'pvi_opt_125M',\
            f'{percentage}_variability_opt_125M_{dataset}.json':'variability_opt_125M',\
                f'{percentage}_vog_score_opt_125M_{dataset}.json':'vog_score_opt_125M'}
#,\                    f'{percentage}_random_{dataset}.json':'random'}
path_dir = f'/path/to/scratch/data/processed/llama3/{dataset}/'

for metric in columns:
    full_path = os.path.join(path_dir, metric)
    files = os.listdir(full_path)
    print('full_path ',full_path)
    full_path = os.path.join(full_path, files[0])
    print('full_path ',full_path)
    #df = pd.read_csv(full_path, delimiter='\t')
    df = pd.read_json(full_path, lines=True)
    print(df.columns)
    selected_columns = ['id', metric]
    print(selected_columns)
    df = df[selected_columns]
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on='id', how="outer")
main_file_name_df = pd.read_json(main_file_name, lines=True)
merged_df = pd.merge(main_file_name_df, merged_df)
print('merged_df ',merged_df.columns)
merged_df.to_json(f"{path_dir}merged_output.json", orient='records', lines=True)