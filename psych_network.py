import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_csv("Eliksiirit_networkanalysis.csv", na_values = [' ']) #, dtype = float)

pval = .1
corr_thres = .15
leave_out = ['Tutkimusnumero']
cols = [col for col in df.columns if col not in leave_out]
df = df[cols]

cols = [col.split('_0')[0] for col in df.columns]
df.columns = cols

corr_df = pd.DataFrame(index = cols, columns = cols)
corr_s_df = pd.DataFrame(index = cols, columns = cols)
network_df = pd.DataFrame(index = range(len(cols)**2),
                          columns = ['source', 'target', 'weight', 'sig'])

for i,col_r in enumerate(cols):
    for j,col_c in enumerate(cols):
        x,y = df[col_r], df[col_c]
        nans = np.logical_or(x.isnull(), y.isnull())
        cor, sig = pearsonr(x[~nans], y[~nans])
        corr_df.loc[col_r, col_c] = cor
        corr_s_df.loc[col_r, col_c] = sig
        n = i*len(cols) + j
        network_df.loc[n, ['source', 'target', 'weight', 'sig']] = col_r, col_c, cor, sig

mask = corr_s_df < pval
corr_df[mask]
network_df = network_df[(network_df['sig'] < pval) &
                        (network_df['source'] != network_df['target']) &
                        (corr_thres < network_df['weight'])] \
                        [['source', 'target', 'weight']]
G = nx.from_pandas_dataframe(network_df, 'source', 'target', 'weight')

plt.figure(figsize = (20,20))
edge_width = [20*G[u][v]['weight'] for u,v in G.edges()]
pos=nx.spectral_layout(G)
#pos=nx.shell_layout(G)
#pos=nx.spring_layout(G)
pos=nx.fruchterman_reingold_layout(G)
#nx.draw_graphviz(G, prog='neato')
nx.draw_networkx(G, pos = pos, node_color = 'blue', node_size=2000,
                 with_labels = False, alpha = .25, width=edge_width)

#nx.draw_networkx_edges(G, pos, edgelist=greater_than_thres, edge_color='r',
#                       alpha=0.4, width=6)
nx.draw_networkx_labels(G, pos, font_size=22, font_color='black')


plt.axis('off')
plt.show()
