import math
import csv
mac_cost = 2
model_param = {}
model_param['Llama_2_7b'] = {'n_layer':32, 'n_ctx':4096, 'd_attn': 128*32, 'd_model':4096, 'n_vocab':32000, 'd_ffn':11008, 'total_parameters': 6738149376}
model_param['Llama_2_13b'] = {'n_layer':40, 'n_ctx':4096, 'd_attn': 128*40, 'd_model':5120, 'n_vocab':32000, 'd_ffn':13824}
model_param['Llama_2_70b'] = {'n_layer':80, 'n_ctx':4096, 'd_attn': 128*80, 'd_model':8192, 'n_vocab':32000, 'd_ffn':28672}

model_param['OPT_125M'] = {'n_layer':12, 'n_ctx':2048, 'd_attn': 64*12, 'd_model':768, 'n_vocab':50268, 'd_ffn':3072, 'total_parameters':125117952}

def FLOPCalculator(total_tokens, model_name):
    total_parameters = mac_cost*model_param[model_name]['d_model']*model_param[model_name]['n_layer']*(2*model_param[model_name]['d_attn']+0.5*model_param[model_name]['d_ffn']) +  2*model_param[model_name]['n_vocab']*model_param[model_name]['d_model']

    total_parameters = model_param[model_name]['total_parameters']
    flops_per_token = mac_cost*total_parameters + mac_cost*model_param[model_name]['n_layer']*model_param[model_name]['n_ctx']*model_param[model_name]['d_attn']
    return flops_per_token, total_tokens*flops_per_token

flops_per_token, total_cost = FLOPCalculator(math.pow(10,8), 'Llama_2_7b')
print(flops_per_token)
num_epochs = 3
forward_backward = 3
model_param['Llama_2_7b']['flops_per_token'] = flops_per_token*forward_backward

#flops_per_token, total_cost = FLOPCalculator(math.pow(10,8), 'Llama_2_13b')
#print(total_cost)
#flops_per_token, total_cost = FLOPCalculator(math.pow(10,8), 'Llama_2_70b')
#print(total_cost)
flops_per_token, total_cost = FLOPCalculator(math.pow(10,8), 'OPT_125M')
model_param['OPT_125M']['flops_per_token'] = flops_per_token
tokens = {'arc_easy': {}, 'arc_challenge':{},'piqa':{},'siqa':{}}
tokens['arc_easy']['10'] = 6410
tokens['arc_easy']['20'] = 13215
tokens['arc_easy']['30'] = 20436
tokens['arc_easy']['40'] = 27771
tokens['arc_easy']['50'] = 35750
tokens['arc_easy']['60'] = 43875
tokens['arc_easy']['70'] = 52068
tokens['arc_easy']['80'] = 61128
tokens['arc_easy']['90'] = 70661
tokens['arc_easy']['100'] = 81439

tokens['arc_challenge']['10'] = 3294
tokens['arc_challenge']['20'] = 6903
tokens['arc_challenge']['30'] = 11109
tokens['arc_challenge']['40'] = 15277
tokens['arc_challenge']['50'] = 19633
tokens['arc_challenge']['60'] = 24190
tokens['arc_challenge']['70'] = 29128
tokens['arc_challenge']['80'] = 34422
tokens['arc_challenge']['90'] = 39804
tokens['arc_challenge']['100'] = 46029

tokens['piqa']['10'] = 28649
tokens['piqa']['20'] = 70207
tokens['piqa']['30'] = 123601
tokens['piqa']['40'] = 187768
tokens['piqa']['50'] = 258477
tokens['piqa']['60'] = 338493
tokens['piqa']['70'] = 423292
tokens['piqa']['80'] = 512787
tokens['piqa']['90'] = 606049
tokens['piqa']['100'] = 703172

tokens['siqa']['10'] = 109865
tokens['siqa']['20'] = 222367
tokens['siqa']['30'] = 339035
tokens['siqa']['40'] = 458761
tokens['siqa']['50'] = 581168
tokens['siqa']['60'] = 705637
tokens['siqa']['70'] = 831987
tokens['siqa']['80'] = 960755
tokens['siqa']['90'] = 1092478
tokens['siqa']['100'] = 1230298
field_columns = ['percentage', 'dataset', 'Total_FLOPS', 'Data_Selection_FLOPS', 'Fine_Tuning_FLOPS', 'Method']
f = open('results/FLOPS.csv', 'w')
writer = csv.DictWriter(f, fieldnames=field_columns,delimiter='\t')
writer.writeheader()
method_names = ['high_p0','high_p1','high_p2','high_p3', 'high_vog', 'hard_to_learn','ambiguous', 'low_pvi', 'random']

def data_selection_flops_per_token(method_name):
    if method_name == 'high_p0':
        return model_param['OPT_125M']['flops_per_token']
    if method_name == 'high_p1':
        return model_param['OPT_125M']['flops_per_token']*forward_backward
    if method_name == 'high_p2':
        return model_param['OPT_125M']['flops_per_token']*forward_backward*2
    if method_name == 'high_p3':
        return model_param['OPT_125M']['flops_per_token']*forward_backward*3
    if method_name == 'high_vog':
        return model_param['Llama_2_7b']['flops_per_token']*num_epochs
    if method_name == 'hard_to_learn':
        return model_param['Llama_2_7b']['flops_per_token']*num_epochs
    if method_name == 'ambiguous':
        return model_param['Llama_2_7b']['flops_per_token']*num_epochs
    if method_name == 'low_pvi':
        return model_param['Llama_2_7b']['flops_per_token']*2*num_epochs
    if method_name == 'random':
        return 0
    
def FT_flops_per_token():
    return model_param['Llama_2_7b']['flops_per_token']*num_epochs

        
    
    
    
def format_large_number(num):
    if num >= 1e12:
        return f"{num / 1e12:.2f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    else:
        return str(num)
    
for p in [10,20,30,40,50,60,70,80,90,100]:
    for data in tokens.keys():
        for method in method_names:
            data_Selection_FLOPS = data_selection_flops_per_token(method)*tokens[data][str(100)]
            fine_Tuning_FLOPS = FT_flops_per_token()*tokens[data][str(p)] 
            total_flops = fine_Tuning_FLOPS+data_Selection_FLOPS
            writer.writerow({'percentage':p, 'dataset':data, 'Method':method, 'Data_Selection_FLOPS':data_Selection_FLOPS, 'Fine_Tuning_FLOPS':fine_Tuning_FLOPS, 'Total_FLOPS': total_flops})
