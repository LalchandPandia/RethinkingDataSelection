import csv, json
f = open('results/scores_per_method.csv', 'w')
writer= csv.DictWriter(f, fieldnames=['method', 'id','dataset', 'score'])
writer.writeheader()
method_types = ['high_p0','high_p1','high_p2','high_p3', 'high_vog', 'hard_to_learn', 'low_pvi']
percentage = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
dataset_name = ['arc_easy', 'arc_challenge','piqa', 'siqa']
for dataset in dataset_name:
    for method_name in method_types:
        f = open(f'results/scores/{dataset}/100_{method_name}_{dataset}.json')
        if method_name == 'high_vog':
            metric = 'vog_score'
        elif method_name == 'hard_to_learn':
            metric_list = ['hard_to_learn', 'ambiguous']
        elif method_name == 'low_pvi':
            metric = 'pvi'
        else:
            metric = 'perplexity'
        for row in f:
            data = json.loads(row)
            if method_name == 'hard_to_learn':
                d = {'method':metric_list[0], 'id':data['id'], 'dataset':dataset,'score':data['confidence']}
                writer.writerow(d)
                d = {'method':metric_list[1], 'id':data['id'], 'dataset':dataset,'score':data['variability']}
            else:
                d = {'method':method_name, 'id':data['id'], 'dataset':dataset,'score':data[metric]}
                 
            writer.writerow(d)
        
    