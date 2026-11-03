import matplotlib.pyplot as plt
import sys
data_percentage = [10, 20, 30, 40, 50, 60, 70, 80, 90]
accuracy_low_arc_easy_P0 = [42.69085743586222, 54.54731227344937, 57.63576183566341, 62.11485519435671, 66.66446915329827, 71.35061894257863, 70.03602072957962, 76.23122771170404, 87.50303670777215]
accuracy_low_arc_easy_P1 = [1.0687883541319105, 1.1774896852175394, 1.4095360432730781, 1.8814205233256023, 2.5318358240127563, 3.2769179844856264, 4.0720345526649835, 5.191146086520619, 6.99695680965612]
accuracy_low_arc_easy_P2 = [1.050303824742635, 1.1306090047624375, 1.2733496280952736, 1.5441923322942521, 1.9333706519868639, 2.425367908036267, 3.026009202306233, 3.820976499915123, 5.001306195612307]
accuracy_low_arc_easy_P3 = [1.0433302805158826, 1.1103953968154059, 1.2379689428541396, 1.4835294402970207, 1.8497824711269804, 2.308902377552456, 2.8781928239549908, 3.640048234065374, 4.7641354317135285]

accuracy_high_arc_easy_P0 = [352.6507940928141, 265.1886739370558, 216.63548657311333, 178.042969294654, 161.39362427224054, 148.64767681298432, 138.19031178943695, 128.90757571869426, 121.94595909907494]
accuracy_high_arc_easy_P1 = [62.81055998060438, 42.149736080169674, 32.444858871036104, 26.53982988357544, 22.636593731986153, 19.722677585902037, 17.376828038049123, 15.439642750223477, 13.8671469006715]
accuracy_high_arc_easy_P2 = [42.58587696499295, 28.533928807576498, 22.155827799196597, 18.27684028625488, 15.599532765282525, 13.582299946855615, 11.979271637901427, 10.677009949949053, 9.625336938963995]
accuracy_high_arc_easy_P3 = [39.20380917019314, 26.497336875067816, 20.66057890291567, 17.07126613405016, 14.579200237909953, 12.702662798210426, 11.205807049842107, 9.99197623444928, 9.01268732017941]

accuracy_low_arc_challenge_P0 =  [61.66756805643305, 62.62589084086397, 53.14806232879411, 49.24770832595142, 53.22677000520054, 52.96983651361593, 58.02866994070976, 64.00496574167433, 78.3491849880351]
accuracy_low_arc_challenge_P1 =  [1.865406474551639, 2.5368666002034073, 3.5943744364069468, 4.45776656516713, 5.7126074317103, 6.882856658069992, 8.382867353172594, 10.504246922045446, 13.296376996741158]
accuracy_low_arc_challenge_P2 = [1.1646919433061067, 1.5420800789589304, 2.228485608100891, 2.983893153118074, 3.840957235351658, 4.81055717336794, 6.011396804988612, 7.516610813540454, 9.691685613004312]
accuracy_low_arc_challenge_P3 =  [1.1563641960556443, 1.5110633629854484, 2.1660373630808363, 2.901346996059887, 3.716318421372361, 4.657049223847254, 5.817027532277893, 7.256911288959354, 9.340182141964107]

accuracy_high_arc_challenge_P0 = [617.4260938146093, 405.32892531664385, 305.55006396734893, 251.20434763607562, 211.25669598963265, 187.53691160980884, 166.10813869766196, 149.62532400259093, 140.00483095278923]
accuracy_high_arc_challenge_P1 = [137.25600991807542, 86.24636108672138, 65.90541766152452, 53.7512705907459, 45.542281765417464, 39.72950831646358, 35.05895716903463, 31.390062725610573, 28.256824917802746]
accuracy_high_arc_challenge_P2 = [97.32383308754311, 62.098448603677106, 47.38758773519032, 38.8323579570591, 32.99123942660091, 28.698445493524726, 25.346463295996266, 22.626410482982017, 20.324305916140307]
accuracy_high_arc_challenge_P3 = [92.75759846455342, 59.277998141643714, 45.269008018721394, 37.13415146087373, 31.573053904210944, 27.469321542276532, 24.271913496775035, 21.670540896474318, 19.468782048949894]

accuracy_low_piqa_P0 = [17.00494749171626, 23.110554575624118, 28.2232328474016, 32.21623878327333, 34.832086314481444, 37.0285593997215, 40.98573751546398, 48.84620716701283, 59.19688192190577]
accuracy_low_piqa_P1 = [4.158993777616064, 5.496276247249783, 6.611429028917538, 7.652873574845822, 8.6839586301765, 9.784432132364714, 11.027957524747245, 12.523029322111856, 14.623021847617585]
accuracy_low_piqa_P2 = [3.6528245050073926, 4.822545726354043, 5.80969972347044, 6.738457292101011, 7.655361448078904, 8.633562056369886, 9.7374491473231, 11.051865261353846, 12.858454685557769]
accuracy_low_piqa_P3 = [3.543588548769202, 4.681847049537318, 5.649209348158439, 6.5551768841162685, 7.4557552617320715, 8.415061908682265, 9.496735541974202, 10.781358847181588, 12.547143294296465]

accuracy_high_piqa_P0 = [638.4704895019531, 390.24541885286, 294.7919918761203, 237.24747496028573, 199.40593390514502, 173.71764785407112, 155.21045478969953, 140.61802712985025, 128.2407165955744]
accuracy_high_piqa_P1 = [104.32781659321041, 67.88186994655617, 52.9183429741302, 44.30774616466371, 38.50434915548285, 34.222326054251944, 30.87142873480447, 28.118454766661923, 25.753800142700037]
accuracy_high_piqa_P2 = [85.99381503431343, 56.65653208632561, 44.524091607316194, 37.47950125517449, 32.68937029159057, 29.128964401541865, 26.32712352600321, 24.009708482706795, 22.0081830737295]
accuracy_high_piqa_P3 = [83.00978383420048, 54.84585461832587, 43.15665169717263, 36.36066887561807, 31.731848513121165, 28.286871269642784, 25.56931713128599, 23.321626170981848, 21.37740937792575]

accuracy_low_siqa_P0 = [1.0, 2006.0615306813643, 1762.0817834423365, 1597.427063690692, 1578.3211644176379, 1542.4882636119908, 1433.687586108738, 1369.8897600995097, 1787.8023780134283]
accuracy_low_siqa_P1 = [1.0, 1.061166669251283, 1.4562442558296758, 1.9456794167646068, 2.562180320326706, 3.397986447616072, 4.47020595828704, 6.021169249395417, 9.198441795012897]
accuracy_low_siqa_P2 = [1.0, 1.0321210393717246, 1.2222407766174892, 1.5335792122273273, 1.9388793009478544, 2.4706391255964744, 3.1728206132565098, 4.197571320479875, 6.213332468322596]
accuracy_low_siqa_P3 = [1.0, 1.0266210649478082, 1.1952136219540934, 1.4767791214138404, 1.8504770312113734, 2.3391680852638728, 2.9877519590672064, 3.9412039160344885, 5.791852031680519]

accuracy_high_siqa_P0 = [25466.48076512863, 15298.792043226702, 10506.963021496105, 8075.443146394385, 6733.019269032258, 5861.165652081119, 5181.493830988925, 4693.072388235844, 4617.300240805498]
accuracy_high_siqa_P1 = [2448.077102549809, 1241.3468623548808, 833.1905456656386, 627.6187900048071, 503.6104354206583, 420.5133935063111, 360.927763705348, 316.0925931708028, 281.09589763388055]
accuracy_high_siqa_P2 = [1330.0564207740974, 676.1979212125808, 454.5888895654017, 342.78814455885544, 275.2564032968523, 229.97368269001524, 197.47281295130688, 172.98902136378214, 153.88626810988896]
accuracy_high_siqa_P3 = [1288.4738561909414, 654.5354465738951, 439.8954202541986, 331.64137899112075, 266.26962786400185, 222.44890133173513, 191.00212623002912, 167.3184102932713, 148.8445027195629]

'''
accuracy_low_arc_challenge = [0.468,0.506, 0.522, 0.533, 0.527, 0.55]
accuracy_high_arc_challenge = [0.471,0.506, 0.525, 0.533, 0.545, 0.55]
accuracy_low_piqa = [0.790, 0.785, 0.799, 0.788, 0.788, 0.788, 0.806, 0.804, 0.806, 0.804]
accuracy_high_piqa = [0.789,0.786, 0.798, 0.788, 0.787, 0.783, 0.804, 0.806, 0.805, 0.804]

accuracy_low_siqa = [0.523, 0.539, 0.548, 0.558, 0.557, 0.556, 0.58]
accuracy_high_siqa = [0.562,0.568, 0.573, 0.568, 0.566, 0.570, 0.58]
'''




color_list = ['k','y','m','g','c','r','b','lime']
marker_list = ['o','s','^','x','d','p','*','8']
linestyle_list = ['solid','dashed','dashdot','dotted', 'solid','dashed','dashdot','dotted']
fig,ax =  plt.subplots(2,2)
ax[0][0].set_xlabel('% of data used')
ax[0][0].set_ylabel('Avg perplexity on ARC-Easy')
ax[0][0].set_title("Perplexity: base: OPT 125M")
ax[0][0].set_xticks([0,10,20,30,40,50, 60, 70, 80, 90, 100])
#ax[0].set_yticks([0.764, 0.779, 0.809, 0.810, 0.810, 0.812])
ax[0][0].plot(data_percentage, accuracy_low_arc_easy_P0, '-ok',color=color_list[0],markersize=7,linestyle=linestyle_list[0],label='lp-P0',marker=marker_list[0])
ax[0][0].plot(data_percentage, accuracy_low_arc_easy_P1, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='lp-P1',marker=marker_list[1])
ax[0][0].plot(data_percentage, accuracy_low_arc_easy_P2, '-ok',color=color_list[2],markersize=7,linestyle=linestyle_list[2],label='lp-P2',marker=marker_list[2])
ax[0][0].plot(data_percentage, accuracy_low_arc_easy_P3, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[3],label='lp-P3',marker=marker_list[3])

ax[0][0].plot(data_percentage, accuracy_high_arc_easy_P0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[4],label='hp-P0',marker=marker_list[4])
ax[0][0].plot(data_percentage, accuracy_high_arc_easy_P1, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[5],label='hp-P1',marker=marker_list[5])
ax[0][0].plot(data_percentage, accuracy_high_arc_easy_P2, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[6],label='hp-P2',marker=marker_list[6])
ax[0][0].plot(data_percentage, accuracy_high_arc_easy_P3, '-ok',color=color_list[7],markersize=7,linestyle=linestyle_list[7],label='hp-P3',marker=marker_list[7])

ax[0][1].set_xlabel('% of data used')
ax[0][1].set_ylabel('Avg perplexity on ARC-Challenge')
ax[0][1].set_title("Perplexity: base: OPT 125M")
ax[0][1].plot(data_percentage, accuracy_low_arc_challenge_P0, '-ok',color=color_list[0],markersize=7,linestyle=linestyle_list[0],label='lp-P0',marker=marker_list[0])
ax[0][1].plot(data_percentage, accuracy_low_arc_challenge_P1, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='lp-P1',marker=marker_list[1])
ax[0][1].plot(data_percentage, accuracy_low_arc_challenge_P2, '-ok',color=color_list[2],markersize=7,linestyle=linestyle_list[2],label='lp-P2',marker=marker_list[2])
ax[0][1].plot(data_percentage, accuracy_low_arc_challenge_P3, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[3],label='lp-P3',marker=marker_list[3])

ax[0][1].plot(data_percentage, accuracy_high_arc_challenge_P0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[4],label='hp-P0',marker=marker_list[4])
ax[0][1].plot(data_percentage, accuracy_high_arc_challenge_P1, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[5],label='hp-P1',marker=marker_list[5])
ax[0][1].plot(data_percentage, accuracy_high_arc_challenge_P2, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[6],label='hp-P2',marker=marker_list[6])
ax[0][1].plot(data_percentage, accuracy_high_arc_challenge_P3, '-ok',color=color_list[7],markersize=7,linestyle=linestyle_list[7],label='hp-P3',marker=marker_list[7])

ax[1][0].set_xlabel('% of data used')
ax[1][0].set_ylabel('Avg perplexity on PIQA')
ax[1][0].set_title("Perplexity: base: OPT 125M")
ax[1][0].plot(data_percentage, accuracy_low_piqa_P0, '-ok',color=color_list[0],markersize=7,linestyle=linestyle_list[0],label='lp-P0',marker=marker_list[0])
ax[1][0].plot(data_percentage, accuracy_low_piqa_P1, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='lp-P1',marker=marker_list[1])
ax[1][0].plot(data_percentage, accuracy_low_piqa_P2, '-ok',color=color_list[2],markersize=7,linestyle=linestyle_list[2],label='lp-P2',marker=marker_list[2])
ax[1][0].plot(data_percentage, accuracy_low_piqa_P3, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[3],label='lp-P3',marker=marker_list[3])

ax[1][0].plot(data_percentage, accuracy_high_piqa_P0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[4],label='hp-P0',marker=marker_list[4])
ax[1][0].plot(data_percentage, accuracy_high_piqa_P1, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[5],label='hp-P1',marker=marker_list[5])
ax[1][0].plot(data_percentage, accuracy_high_piqa_P2, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[6],label='hp-P2',marker=marker_list[6])
ax[1][0].plot(data_percentage, accuracy_high_piqa_P3, '-ok',color=color_list[7],markersize=7,linestyle=linestyle_list[7],label='hp-P3',marker=marker_list[7])

ax[1][1].set_xlabel('% of data used')
ax[1][1].set_ylabel('Avg perplexity on SIQA')
ax[1][1].set_title("Perplexity: base: OPT 125M")
ax[1][1].plot(data_percentage, accuracy_low_siqa_P0, '-ok',color=color_list[0],markersize=7,linestyle=linestyle_list[0],label='lp-P0',marker=marker_list[0])
ax[1][1].plot(data_percentage, accuracy_low_siqa_P1, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='lp-P1',marker=marker_list[1])
ax[1][1].plot(data_percentage, accuracy_low_siqa_P2, '-ok',color=color_list[2],markersize=7,linestyle=linestyle_list[2],label='lp-P2',marker=marker_list[2])
ax[1][1].plot(data_percentage, accuracy_low_siqa_P3, '-ok',color=color_list[3],markersize=7,linestyle=linestyle_list[3],label='lp-P3',marker=marker_list[3])

ax[1][1].plot(data_percentage, accuracy_high_siqa_P0, '-ok',color=color_list[4],markersize=7,linestyle=linestyle_list[4],label='hp-P0',marker=marker_list[4])
ax[1][1].plot(data_percentage, accuracy_high_siqa_P1, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[5],label='hp-P1',marker=marker_list[5])
ax[1][1].plot(data_percentage, accuracy_high_siqa_P2, '-ok',color=color_list[6],markersize=7,linestyle=linestyle_list[6],label='hp-P2',marker=marker_list[6])
ax[1][1].plot(data_percentage, accuracy_high_siqa_P3, '-ok',color=color_list[7],markersize=7,linestyle=linestyle_list[7],label='hp-P3',marker=marker_list[7])

#ax[0][0].legend(loc='lower right', bbox_to_anchor=(1, -0.2), fancybox=True, shadow=True, ncol=8)

#ax[0][0].plot(data_percentage, accuracy_low_arc_easy_P1, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='high-perplexity',marker=marker_list[1])
#ax[0][0].plot(data_percentage, accuracy_low_arc_easy_P1, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='high-perplexity',marker=marker_list[1])

#ax[0][0].axhline(y=0.764, color='r', linestyle='-', label='zero-shot')

#ax[0][1].set_xlabel('% of data used')
#ax[0][1].set_ylabel('Accuracy on ARC-Challenge')
#ax[0][1].set_title("Target: Llama 7B, base: OPT 125M")
#ax[0][1].set_xticks([0,10,20,30,40, 50,100])
#ax[1].set_yticks([0.764, 0.779, 0.809, 0.810, 0.810, 0.812])
#ax[0][1].plot(data_percentage, accuracy_low_arc_challenge, '-ok',color=color_list[0],markersize=7,linestyle=linestyle_list[0],label='low-perplexity',marker=marker_list[0])
#ax[0][1].plot(data_percentage, accuracy_high_arc_challenge, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='high-perplexity',marker=marker_list[1])
#ax[0][1].axhline(y=0.435, color='r', linestyle='-', label='zero-shot')
#data_percentage_temp = [10, 20, 30, 40, 50,  60, 70, 80, 90, 100]
#ax[1][0].set_xlabel('% of data used')
#ax[1][0].set_ylabel('Accuracy on PIQA')
#ax[1][0].set_title("Target: Llama 7B, base: OPT 125M")
#ax[1][0].set_xticks([10,20,30,40, 50,60, 70, 80, 90, 100])
#ax[1].set_yticks([0.764, 0.779, 0.809, 0.810, 0.810, 0.812])
#ax[1][0].plot(data_percentage_temp, accuracy_low_piqa, '-ok',color=color_list[0],markersize=7,linestyle=linestyle_list[0],label='low-perplexity',marker=marker_list[0])
#ax[1][0].plot(data_percentage_temp, accuracy_high_piqa, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='high-perplexity',marker=marker_list[1])
#ax[1][0].axhline(y=0.780, color='r', linestyle='-', label='zero-shot')

#data_percentage_temp = [10, 20, 30, 40, 50,  60, 100]
#ax[1][1].set_xlabel('% of data used')
#ax[1][1].set_ylabel('Accuracy on SIQA')
#ax[1][1].set_title("Target: Llama 7B, base: OPT 125M")
#ax[1][1].set_xticks([10,20,30,40, 50,60,100])
#ax[1].set_yticks([0.764, 0.779, 0.809, 0.810, 0.810, 0.812])
#ax[1][1].plot(data_percentage_temp, accuracy_low_siqa, '-ok',color=color_list[0],markersize=7,linestyle=linestyle_list[0],label='low-perplexity',marker=marker_list[0])
#ax[1][1].plot(data_percentage_temp, accuracy_high_siqa, '-ok',color=color_list[1],markersize=7,linestyle=linestyle_list[1],label='high-perplexity',marker=marker_list[1])
#ax[1][1].axhline(y=0.461, color='r', linestyle='-', label='zero-shot')





#plt.axhline(y=0.810, color='g', linestyle='-', label='next-10%-low')
plt.legend(loc='lower right', bbox_to_anchor=(0.05, -0.2),fancybox=True, shadow=True, ncol=8)
#plt.legend(loc='lower right', bbox_to_anchor=(1, -0.2), fancybox=True, shadow=True, ncol=8)
lines, labels = ax[0][0].get_legend_handles_labels()
print(labels)
print(lines)
#fig.legend(lines,labels,loc='lower right', bbox_to_anchor=(0.3, -0.18),fancybox=True, shadow=True, ncol=8)
#fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=8, fancybox=True, shadow=True)
plt.tight_layout()
plt.show()
sys.exit()
