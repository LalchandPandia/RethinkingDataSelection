from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_json('100_high_p3-opt-125M_arc_easy.json',lines=True)
fig,ax =  plt.subplots(2,2)
color_list = ['k','y','m','g','c','r','b','lime']
marker_list = ['o','s','^','x','d','p','*','8']
linestyle_list = ['solid','dashed','dashdot','dotted', 'solid','dashed','dashdot','dotted']

p0 = df['P0'].values[:, np.newaxis]
p1 = df['P1'].values[:, np.newaxis]
p2 = df['P2'].values[:, np.newaxis]
p3 = df['P3'].values[:, np.newaxis]
df['P0_log'] = np.log1p(df['P0'])
df['P1_log'] = np.log1p(df['P1'])
df['P2_log'] = np.log1p(df['P2'])
df['P3_log'] = np.log1p(df['P3'])
sns.kdeplot( data = df, x='P0_log', ax = ax[0][0],  multiple='stack', bw_adjust=0.5)
sns.kdeplot(data =df, x='P1_log',ax = ax[0][1], multiple='stack', bw_adjust=0.5)
sns.kdeplot(data = df, x = 'P2_log', ax = ax[1][0], multiple='stack', bw_adjust=0.5)
sns.kdeplot(data =df, x = 'P3_log', ax = ax[1][1], multiple='stack', bw_adjust=0.5)
plt.show()
#ax[0][0] = df['P0'].plot.kde()
#ax[0][1] = df['P1'].plot.kde()
#ax[1][0] = df['P2'].plot.kde()
#ax[1][1] = df['P3'].plot.kde()
#plt.show()
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(p0)
dens = kde.score_samples(p0)
ax[0][0].plot(p0, dens, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[5],label='lp-P0',marker=marker_list[5])
print(type(ax[0][0].plot(p0, dens, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[5],label='lp-P0',marker=marker_list[5])
))
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(p1)
dens = kde.score_samples(p1)
ax[0][1].plot(p1, dens, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[5],label='lp-P0',marker=marker_list[5])

kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(p2)
dens = kde.score_samples(p2)
ax[1][0].plot(p2, dens, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[5],label='lp-P0',marker=marker_list[5])

kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(p3)
dens = kde.score_samples(p3)
ax[1][1].plot(p3, dens, '-ok',color=color_list[5],markersize=7,linestyle=linestyle_list[5],label='lp-P0',marker=marker_list[5])
plt.show()

