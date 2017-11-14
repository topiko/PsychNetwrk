import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from utils import part_corr

fname = 'ACTFöräldrar_network.csv'  #"Eliksiirit_networkanalysis.csv"
df = pd.read_csv(fname, na_values = [' ']) #, dtype = float)

pval = .1
corr_thres = .0
leave_out = ['Tutkimusnumero', 'GROUP']

smbq_mean = df['SMBQ_PRE'].mean()
print(smbq_mean)

df = df[df['SMBQ_PRE'] > smbq_mean]

keep = ['AAQ_PRE', 'CFQ_PRE', 'FFMQ_obs','FFMQ_des',
        'FFMQ_awa', 'FFMQ_jud', 'FFMQ_rea', 'DASS_stress_PRE',
        'DASS_anxiety_PRE', 'DASS_depression_PRE', 'ISI_PRE',
        'SMBQ_FT_pre', 'SMBQ_HL_pre', 'SMBQ_SP_pre', 'SMBQ_MT_pre']
cols = [col for col in df.columns if col not in leave_out]
cols = [col for col in df.columns if col in keep]

df = df[cols]

cols = [col.split('_0')[0] for col in df.columns]
df.columns = cols

corr_df = pd.DataFrame(index = cols, columns = cols)
corr_s_df = pd.DataFrame(index = cols, columns = cols)
network_df = pd.DataFrame(index = range(len(cols)**2),
                          columns = ['source', 'target',
                                     'corr', 'part_corr', 'corr_sign', 'sign_part_corr',
                                     'significance', 'significance_part'])

for i,col_r in enumerate(cols):
    for j,col_c in enumerate(cols):
        x,y = df[col_r], df[col_c]
        col_list = [col for col in df.columns if col not in [col_c, col_r]]
        arr_tmp = df[col_list].values

        nans = np.logical_or(x.isnull(), y.isnull())
        cor, sig = pearsonr(x[~nans], y[~nans])
        corr_df.loc[col_r, col_c] = abs(cor)
        part_corr_val, sig_part = part_corr(x.values, y.values, arr_tmp)
        #print(col_c, col_r, cor, part_corr_val)
        corr_s_df.loc[col_r, col_c] = sig
        n = i*len(cols) + j
        network_df.loc[n, ['source', 'target',
                           'corr', 'part_corr', 'corr_sign', 'sign_part_corr',
                           'significance', 'significance_part']] = col_r, col_c, \
                                              abs(cor), abs(part_corr_val), \
                                              np.sign(cor), np.sign(part_corr_val), \
                                              sig, sig_part

mask = corr_s_df < pval
corr_df[mask]

weight = 'part_corr' #'corr' #'part_corr'
signi = 'significance_part' #'significance' # 'significance_part'
sign = 'sign_part_corr' #'corr_sign'

network_df = network_df[(network_df[signi] < pval) &
                        (network_df['source'] != network_df['target']) &
                        (corr_thres < network_df[weight])] \
                        [['source', 'target', weight, sign]]
G = nx.from_pandas_dataframe(network_df, 'source', 'target', edge_attr=[weight, sign])

plt.figure(figsize = (20,20))
edges1 = [x for x in G.edges(data=True) if 0 < x[2][sign]]
edges2 = [x for x in G.edges(data=True) if x[2][sign] < 0]
edge_width1 = [20*G[u][v][weight] for u,v in G.edges() if 0 < G[u][v][sign]]
edge_width2 = [20*G[u][v][weight] for u,v in G.edges() if G[u][v][sign] < 0]
node_sizes = [len(G.neighbors(n))*300 for n in G.nodes()]

pos=nx.spectral_layout(G)
#pos=nx.shell_layout(G)
pos=nx.spring_layout(G)
pos=nx.fruchterman_reingold_layout(G)
#nx.draw_graphviz(G, prog='neato')
nx.draw_networkx(G, pos = pos, node_color = 'black', node_size=node_sizes,
                 with_labels = False, alpha = .25)

nx.draw_networkx_edges(G, pos, edgelist=edges1, edge_color='b',
                       alpha=0.4, width=edge_width1)
nx.draw_networkx_edges(G, pos, edgelist=edges2, edge_color='r',
                       alpha=0.4, width=edge_width2)
nx.draw_networkx_labels(G, pos, font_size=22, font_color='black')


plt.axis('off')
plt.savefig('network.png')
plt.show()
