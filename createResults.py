import csv
f = open('Llama_2_7b_scores.csv', 'w')
writer= csv.DictWriter(f, fieldnames=['method', 'percentage_data_used','dataset', 'accuracy'])
writer.writeheader()
method_types = ['high_p0','high_p1','high_p2','high_p3', 'high_vog', 'hard_to_learn', 'ambiguous', 'random']#'Low_PVI', ]
percentage = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
dataset_name = ['arc_easy', 'arc_challenge', 'piqa', 'siqa']
results = {}
####ARC-Easy#######################################################
results['accuracy_low_p3_arc_easy'] = [0.764, 0.788, 0.793, 0.796, 0.798, 0.784, 0.795, 0.786, 0.790, 0.797, 0.812]
results['accuracy_high_p3_arc_easy'] = [0.764, 0.817, 0.806, 0.811, 0.803, 0.812, 0.801, 0.801, 0.803, 0.801, 0.812]

results['accuracy_low_p0_arc_easy'] = [0.764, 0.782, 0.789, 0.804, 0.793, 0.793, 0.790, 0.789, 0.796, 0.796, 0.812]
results['accuracy_high_p0_arc_easy'] = [0.764, 0.8, 0.805, 0.810, 0.807, 0.801, 0.803, 0.799, 0.794, 0.799, 0.812]

results['accuracy_low_p1_arc_easy'] = [0.764, 0.782, 0.791, 0.788, 0.797, 0.798, 0.792, 0.795, 0.788, 0.799, 0.812]
results['accuracy_high_p1_arc_easy'] = [0.764, 0.821, 0.810, 0.822, 0.804, 0.804, 0.801, 0.804, 0.804, 0.806, 0.812]

results['accuracy_low_p2_arc_easy'] = [0.764, 0.788, 0.784, 0.801, 0.792, 0.793, 0.791, 0.796, 0.801, 0.795, 0.812]
results['accuracy_high_p2_arc_easy'] = [0.764, 0.824, 0.797, 0.813, 0.808, 0.799, 0.811, 0.801, 0.801, 0.793, 0.812]

results['accuracy_hard_to_learn_arc_easy'] = [0.764, 0.806, 0.794, 0.802, 0.813, 0.805, 0.809, 0.813, 0.805, 0.807, 0.812]
results['accuracy_ambiguous_arc_easy'] = [0.764, 0.810, 0.806, 0.802, 0.804, 0.800, 0.796, 0.790, 0.790, 0.800, 0.812]

results['accuracy_high_vog_arc_easy'] = [0.764, 0.778, 0.769, 0.787, 0.782, 0.782, 0.791, 0.780, 0.781, 0.781, 0.812]

results['accuracy_random_arc_easy'] =  [0.764, 0.805, 0.785, 0.8, 0.796, 0.800, 0.800, 0.801, 0.801, 0.798, 0.812]
############ARC-Challenge##############################################
results['accuracy_low_p3_arc_challenge'] = [0.435, 0.474, 0.499, 0.512, 0.513, 0.522, 0.533, 0.533, 0.529, 0.539, 0.547]
results['accuracy_high_p3_arc_challenge'] = [0.435, 0.468, 0.532, 0.543, 0.529, 0.540, 0.537, 0.554, 0.538, 0.542, 0.547]

results['accuracy_low_p0_arc_challenge'] = [0.435, 0.468, 0.505, 0.509, 0.509, 0.510, 0.536, 0.523, 0.541, 0.529, 0.547]
results['accuracy_high_p0_arc_challenge'] = [0.435, 0.471, 0.523, 0.539, 0.541, 0.545, 0.539, 0.544, 0.538, 0.538, 0.547]

results['accuracy_low_p1_arc_challenge'] = [0.435, 0.471, 0.503, 0.503, 0.518, 0.523, 0.529, 0.513, 0.527, 0.519, 0.547]
results['accuracy_high_p1_arc_challenge'] = [0.435, 0.471, 0.526, 0.549, 0.529, 0.541, 0.538, 0.557, 0.549, 0.535, 0.547]

results['accuracy_low_p2_arc_challenge'] = [0.435, 0.468, 0.506, 0.509, 0.514, 0.535, 0.519, 0.526, 0.531, 0.543, 0.547]
results['accuracy_high_p2_arc_challenge'] = [0.435, 0.466, 0.533, 0.531, 0.536, 0.538, 0.551, 0.553, 0.540, 0.535, 0.547]

results['accuracy_hard_to_learn_arc_challenge'] = [0.435, 0.463, 0.515, 0.527, 0.529, 0.534, 0.533, 0.552, 0.526, 0.533, 0.547]
results['accuracy_ambiguous_arc_challenge'] = [0.435, 0.474, 0.524, 0.527, 0.527, 0.532, 0.533, 0.526, 0.525, 0.527, 0.547]

results['accuracy_random_arc_challenge'] = [0.435, 0.463, 0.518, 0.521, 0.530, 0.523, 0.527, 0.537, 0.520, 0.540, 0.547]

results['accuracy_high_vog_arc_challenge'] = [0.435, 0.465, 0.516, 0.518, 0.519, 0.503, 0.509, 0.521, 0.491, 0.499, 0.547]
##############PIQA#########################################
results['accuracy_low_p3_piqa'] = [0.780, 0.780, 0.785, 0.788, 0.795, 0.796, 0.793, 0.806, 0.804, 0.806, 0.804]
results['accuracy_high_p3_piqa'] = [0.780, 0.8, 0.807, 0.804, 0.795, 0.793, 0.789, 0.804, 0.806, 0.805, 0.804]

results['accuracy_low_p0_piqa'] = [0.780, 0.778, 0.786, 0.791,  0.781, 0.784, 0.792, 0.801, 0.799, 0.807, 0.804]
results['accuracy_high_p0_piqa'] = [0.780, 0.794, 0.795,  0.799, 0.791, 0.793, 0.797, 0.794, 0.796, 0.801, 0.804]

results['accuracy_low_p1_piqa'] = [0.780, 0.786, 0.785, 0.794, 0.783, 0.788, 0.799, 0.794, 0.793, 0.803, 0.804]
results['accuracy_high_p1_piqa'] = [0.780, 0.795, 0.798, 0.800, 0.793, 0.799, 0.801, 0.798, 0.798, 0.800, 0.804]

results['accuracy_high_p2_piqa'] = [0.780, 0.799, 0.793, 0.794, 0.793, 0.798, 0.792, 0.797,  0.801, 0.803, 0.804]
results['accuracy_low_p2_piqa'] = [0.780, 0.789, 0.785, 0.784, 0.785, 0.785, 0.792, 0.788, 0.799, 0.797, 0.804]

results['accuracy_hard_to_learn_piqa'] = [0.780, 0.791, 0.794, 0.791, 0.786, 0.801, 0.798, 0.797, 0.795, 0.801, 0.804]
results['accuracy_ambiguous_piqa'] = [0.780, 0.784, 0.785, 0.788, 0.788, 0.785, 0.798, 0.794, 0.792, 0.793, 0.804]

results['accuracy_high_vog_piqa'] = [0.780, 0.775, 0.764, 0.777, 0.773, 0.775, 0.782, 0.779, 0.786, 0.784, 0.804]

results['accuracy_random_piqa'] = [0.780, 0.794, 0.795, 0.792, 0.792, 0.799, 0.805, 0.798, 0.802, 0.801, 0.804]
###########SIQA########################################################
results['accuracy_low_p3_siqa'] = [0.461, 0.523, 0.539, 0.548, 0.558, 0.557, 0.556, 0.564, 0.576, 0.573, 0.58]
results['accuracy_high_p3_siqa']  = [0.461, 0.562,0.568, 0.573, 0.568, 0.566, 0.570, 0.561, 0.564, 0.575, 0.58]

results['accuracy_low_p0_siqa'] = [0.461, 0.534, 0.527, 0.542, 0.548, 0.561, 0.558, 0.550, 0.568, 0.571, 0.58]
results['accuracy_high_p0_siqa'] = [0.461, 0.583,0.575, 0.572, 0.574, 0.587, 0.570, 0.571, 0.570, 0.566, 0.58]

results['accuracy_low_p1_siqa'] = [0.461, 0.528, 0.531, 0.534, 0.535, 0.561, 0.573, 0.566, 0.569, 0.572, 0.58]
results['accuracy_high_p1_siqa'] = [0.461, 0.572, 0.565, 0.534, 0.567, 0.571, 0.573, 0.579, 0.568, 0.575, 0.58]

results['accuracy_low_p2_siqa'] = [0.461, 0.536, 0.522, 0.543, 0.542, 0.556, 0.561, 0.567, 0.580, 0.570, 0.58]
results['accuracy_high_p2_siqa'] = [0.461, 0.577, 0.569, 0.571, 0.564, 0.557, 0.571, 0.570, 0.567, 0.579, 0.58]

results['accuracy_hard_to_learn_siqa'] = [0.461, 0.551, 0.563, 0.555, 0.560, 0.564, 0.564, 0.570, 0.573, 0.569, 0.58]
results['accuracy_ambiguous_siqa'] = [0.461, 0.553, 0.550, 0.560, 0.561, 0.574, 0.567, 0.566, 0.561, 0.569, 0.58]

results['accuracy_high_vog_siqa'] = [0.461, 0.535, 0.540, 0.546, 0.548, 0.544, 0.553, 0.557, 0.545, 0.550, 0.58]

results['accuracy_random_siqa'] = [0.461, 0.577, 0.572, 0.571, 0.573, 0.578, 0.570, 0.571, 0.574, 0.565, 0.58]
data_percentage = [0, 10, 20, 30, 40, 50,  60, 70, 80, 90, 100]
for dataset in dataset_name:
    for method_type in method_types:
        for percentage in data_percentage:
            print(f'accuracy_{method_type}_{dataset}')
            d = {'method':method_type, 'percentage_data_used':percentage, 'dataset':dataset, 'accuracy':results[f'accuracy_{method_type}_{dataset}'][int(percentage/10)]} 
            writer.writerow(d)


