import matplotlib.pyplot as plt

####ARC-Easy#######################################################
accuracy_low_arc_easy = [0.764, 0.788, 0.793, 0.796, 0.798, 0.784, 0.795, 0.786, 0.790, 0.797, 0.812]
accuracy_high_arc_easy = [0.764, 0.817, 0.806, 0.811, 0.803, 0.812, 0.801, 0.801, 0.803, 0.801, 0.812]

accuracy_low_arc_easy_p0 = [0.764, 0.782, 0.789, 0.804, 0.793, 0.793, 0.790, 0.789, 0.796, 0.796, 0.812]
accuracy_high_arc_easy_p0 = [0.764, 0.8, 0.805, 0.810, 0.807, 0.801, 0.803, 0.799, 0.794, 0.799, 0.812]

accuracy_low_arc_easy_p1 = [0.764, 0.782, 0.791, 0.788, 0.797, 0.798, 0.792, 0.795, 0.788, 0.799, 0.812]
accuracy_high_arc_easy_p1 = [0.764, 0.821, 0.810, 0.822, 0.804, 0.804, 0.801, 0.804, 0.804, 0.806, 0.812]

accuracy_low_arc_easy_p2 = [0.764, 0.788, 0.784, 0.801, 0.792, 0.793, 0.791, 0.796, 0.801, 0.795, 0.812]
accuracy_high_arc_easy_p2 = [0.764, 0.824, 0.797, 0.813, 0.808, 0.799, 0.811, 0.801, 0.801, 0.793, 0.812]

accuracy_hard_to_learn_arc_easy = [0.764, 0.806, 0.794, 0.802, 0.813, 0.805, 0.809, 0.813, 0.805, 0.807, 0.812]
accuracy_ambiguous_arc_easy = [0.764, 0.810, 0.806, 0.802, 0.804, 0.800, 0.796, 0.790, 0.790, 0.800, 0.812]

############ARC-Challenge##############################################
accuracy_low_arc_challenge = [0.435, 0.474, 0.499, 0.512, 0.513, 0.522, 0.533, 0.533, 0.529, 0.539, 0.547]
accuracy_high_arc_challenge = [0.435, 0.468, 0.532, 0.543, 0.529, 0.540, 0.537, 0.554, 0.538, 0.542, 0.547]

accuracy_low_arc_challenge_p0 = [0.435, 0.468, 0.505, 0.509, 0.509, 0.510, 0.536, 0.523, 0.541, 0.529, 0.547]
accuracy_high_arc_challenge_p0 = [0.435, 0.471, 0.523, 0.539, 0.541, 0.545, 0.539, 0.544, 0.538, 0.538, 0.547]

accuracy_low_arc_challenge_p1 = [0.435, 0.471, 0.503, 0.503, 0.518, 0.523, 0.529, 0.513, 0.527, 0.519, 0.547]
accuracy_high_arc_challenge_p1 = [0.435, 0.471, 0.526, 0.549, 0.529, 0.541, 0.538, 0.557, 0.549, 0.535, 0.547]

accuracy_low_arc_challenge_p2 = [0.435, 0.468, 0.506, 0.509, 0.514, 0.535, 0.519, 0.526, 0.531, 0.543, 0.547]
accuracy_high_arc_challenge_p2 = [0.435, 0.466, 0.533, 0.531, 0.536, 0.538, 0.551, 0.553, 0.540, 0.535, 0.547]

accuracy_hard_to_learn_arc_challenge = [0.435, 0.463, 0.515, 0.527, 0.529, 0.534, 0.533, 0.552, 0.526, 0.533, 0.547]
accuracy_ambiguous_arc_challenge= [0.435, 0.474, 0.524, 0.527, 0.527, 0.532, 0.533, 0.526, 0.525, 0.527, 0.547]

##############PIQA#########################################
accuracy_low_piqa = [0.780, 0.780, 0.785, 0.788, 0.795, 0.796, 0.793, 0.806, 0.804, 0.806, 0.804]
accuracy_high_piqa = [0.780, 0.8, 0.807, 0.804, 0.795, 0.793, 0.789, 0.804, 0.806, 0.805, 0.804]

accuracy_low_piqa_p0 = [0.780, 0.778, 0.786, 0.791,  0.781, 0.784, 0.792, 0.801, 0.799, 0.807, 0.804]
accuracy_high_piqa_p0 = [0.780, 0.794, 0.795,  0.799, 0.791, 0.793, 0.797, 0.794, 0.796, 0.801, 0.804]

accuracy_low_piqa_p1 = [0.780, 0.786, 0.785, 0.794, 0.783, 0.788, 0.799, 0.794, 0.793, 0.803, 0.804]
accuracy_high_piqa_p1 = [0.780, 0.795, 0.798, 0.800, 0.793, 0.799, 0.801, 0.798, 0.798, 0.800, 0.804]

accuracy_hard_to_learn_piqa = [0.780, 0.791, 0.794, 0.791, 0.786, 0.801, 0.798, 0.797, 0.795, 0.801, 0.804]
accuracy_ambiguous_piqa = [0.780, 0.784, 0.785, 0.788, 0.788, 0.785, 0.798, 0.794, 0.792, 0.793, 0.804]


###########SIQA########################################################
accuracy_low_siqa = [0.461, 0.523, 0.539, 0.548, 0.558, 0.557, 0.556, 0.564, 0.576, 0.573, 0.58]
accuracy_high_siqa = [0.461, 0.562,0.568, 0.573, 0.568, 0.566, 0.570, 0.561, 0.564, 0.575, 0.58]

accuracy_low_siqa_p0 = [0.461, 0.534, 0.527, 0.542, 0.548, 0.561, 0.558, 0.550, 0.568, 0.571, 0.58]
accuracy_high_siqa_p0 = [0.461, 0.583,0.575, 0.572, 0.574, 0.587, 0.570, 0.571, 0.570, 0.566, 0.58]

accuracy_low_siqa_p1 = [0.461, 0.528, 0.531, 0.534, 0.535, 0.561, 0.573, 0.566, 0.569, 0.572, 0.58]
accuracy_high_siqa_p1 = [0.461, 0.572, 0.565, 0.534, 0.567, 0.571, 0.573, 0.579, 0.568, 0.575, 0.58]


data_percentage = [0, 10, 20, 30, 40, 50,  60, 70, 80, 90, 100]


color_list = ['k','y','m','g','c','r','b','lime']
marker_list = ['o','s','^','x','d','p','*','8']
linestyle_list = ['solid','dashed','dashdot','dotted']
fig,ax =  plt.subplots(2,2)
ax[0][0].set_xlabel('% of data used')
ax[0][0].set_ylabel('Accuracy on ARC-Easy')
ax[0][0].set_title("Target: Llama 7B, base: OPT 125M")
ax[0][0].set_xticks(data_percentage)
#ax[0].set_yticks([0.764, 0.779, 0.809, 0.810, 0.810, 0.812])
ax[0][0].plot(data_percentage, accuracy_low_arc_easy, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[0],label='low_p3',marker=marker_list[0])
ax[0][0].plot(data_percentage, accuracy_high_arc_easy, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[1],label='high_p3',marker=marker_list[3])

ax[0][0].plot(data_percentage, accuracy_low_arc_easy_p0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[0],label='low_p0',marker=marker_list[0])
ax[0][0].plot(data_percentage, accuracy_high_arc_easy_p0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[1],label='high_p0',marker=marker_list[3])

ax[0][0].plot(data_percentage, accuracy_low_arc_easy_p1, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[0],label='low_p1',marker=marker_list[0])
ax[0][0].plot(data_percentage, accuracy_high_arc_easy_p1, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[1],label='high_p1',marker=marker_list[3])

ax[0][0].plot(data_percentage, accuracy_low_arc_easy_p2, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[0],label='low_p2',marker=marker_list[0])
ax[0][0].plot(data_percentage, accuracy_high_arc_easy_p2, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='high_p2',marker=marker_list[3])



ax[0][0].plot(data_percentage, accuracy_hard_to_learn_arc_easy, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[0],label='hl',marker=marker_list[0])
ax[0][0].plot(data_percentage, accuracy_ambiguous_arc_easy, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[1],label='amb',marker=marker_list[3])



#ax[0][0].axhline(y=0.764, color='r', linestyle='-', label='zero-shot')
ax[0][1].set_xlabel('% of data used')
ax[0][1].set_ylabel('Accuracy on ARC-Challenge')
ax[0][1].set_title("Target: Llama 7B, base: OPT 125M")
ax[0][1].set_xticks(data_percentage)

ax[0][1].plot(data_percentage, accuracy_low_arc_challenge, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[0],label='low_p3',marker=marker_list[0])
ax[0][1].plot(data_percentage, accuracy_high_arc_challenge, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[1],label='high_p3',marker=marker_list[3])

ax[0][1].plot(data_percentage, accuracy_low_arc_challenge_p0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[0],label='low_p0',marker=marker_list[0])
ax[0][1].plot(data_percentage, accuracy_high_arc_challenge_p0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[1],label='high_p0',marker=marker_list[3])

ax[0][1].plot(data_percentage, accuracy_low_arc_challenge_p1, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[0],label='low_p1',marker=marker_list[0])
ax[0][1].plot(data_percentage, accuracy_high_arc_challenge_p1, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[1],label='high_p1',marker=marker_list[3])

ax[0][1].plot(data_percentage, accuracy_low_arc_challenge_p2, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[0],label='low_p2',marker=marker_list[0])
ax[0][1].plot(data_percentage, accuracy_high_arc_challenge_p2, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='high_p2',marker=marker_list[3])

ax[0][1].plot(data_percentage, accuracy_hard_to_learn_arc_challenge, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[0],label='hl',marker=marker_list[0])
ax[0][1].plot(data_percentage, accuracy_ambiguous_arc_challenge, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[1],label='amb',marker=marker_list[3])

#ax[0][1].axhline(y=0.435, color='r', linestyle='-', label='zero-shot')

ax[1][0].set_xlabel('% of data used')
ax[1][0].set_ylabel('Accuracy on PIQA')
ax[1][0].set_title("Target: Llama 7B, base: OPT 125M")
ax[1][0].set_xticks(data_percentage)

ax[1][0].plot(data_percentage, accuracy_low_piqa, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[0],label='low_p3',marker=marker_list[0])
ax[1][0].plot(data_percentage, accuracy_high_piqa, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[1],label='high_p3',marker=marker_list[3])

ax[1][0].plot(data_percentage, accuracy_low_piqa_p0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[0],label='low_p0',marker=marker_list[0])
ax[1][0].plot(data_percentage, accuracy_high_piqa_p0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[1],label='high_p0',marker=marker_list[3])

ax[1][0].plot(data_percentage, accuracy_low_piqa_p1, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[0],label='low_p1',marker=marker_list[0])
ax[1][0].plot(data_percentage, accuracy_high_piqa_p1, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[1],label='high_p1',marker=marker_list[3])



ax[1][0].plot(data_percentage, accuracy_hard_to_learn_piqa, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[0],label='hl',marker=marker_list[0])
ax[1][0].plot(data_percentage, accuracy_ambiguous_piqa, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[1],label='amb',marker=marker_list[3])

#ax[1][0].axhline(y=0.780, color='r', linestyle='-', label='zero-shot')


ax[1][1].set_xlabel('% of data used')
ax[1][1].set_ylabel('Accuracy on SIQA')
ax[1][1].set_title("Target: Llama 7B, base: OPT 125M")
ax[1][1].set_xticks(data_percentage)

ax[1][1].plot(data_percentage, accuracy_low_siqa, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[0],label='low_p3',marker=marker_list[0])
ax[1][1].plot(data_percentage, accuracy_high_siqa, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[1],label='high_p3',marker=marker_list[3])

ax[1][1].plot(data_percentage, accuracy_low_siqa_p0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[0],label='low_p0',marker=marker_list[0])
ax[1][1].plot(data_percentage, accuracy_high_siqa_p0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[1],label='high_p0',marker=marker_list[3])

ax[1][1].plot(data_percentage, accuracy_low_siqa_p1, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[0],label='low_p1',marker=marker_list[0])
ax[1][1].plot(data_percentage, accuracy_high_siqa_p1, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[1],label='high_p1',marker=marker_list[3])


#ax[1][1].axhline(y=0.461, color='r', linestyle='-', label='zero-shot')





#plt.axhline(y=0.810, color='g', linestyle='-', label='next-10%-low')

plt.legend(loc='lower right', bbox_to_anchor=(0.1, -0.3),fancybox=True, shadow=True, ncol=6)
plt.tight_layout()
plt.show()

