# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:18:33 2025

@author: joonc
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Parameters
M = 0.5  # fraction by which US trade is reduced

# Ensure Figures directory exists
fig_dir = 'Figures'
os.makedirs(fig_dir, exist_ok=True)

# 1. Load the 19×19 export matrix for 2022
df = pd.read_csv(os.path.join(fig_dir, 'bilateral_trade_19x19_2022.csv'), index_col=0)

# 2. Build the graph with reciprocal weights, storing trade_volume
G = nx.Graph()
G.add_nodes_from(df.index)
for i in df.index:
    for j in df.columns:
        if i == j:
            continue
        vol = df.at[i, j]
        if vol > 0:
            G.add_edge(i, j, weight=1.0/vol, trade_volume=vol)

# 3. Apply reduction M to all edges involving the United States
for u, v, d in G.edges(data=True):
    if u == 'United States' or v == 'United States':
        new_vol = d['trade_volume'] * (1 - M)
        d['trade_volume'] = new_vol
        d['weight'] = 1.0 / new_vol

# 4. Compute layout for consistency
pos = nx.spring_layout(G, seed=42)

# 5. Plot and save adjusted original network
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=800,
        node_color='skyblue', edge_color='gray', alpha=0.5)
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels={(u, v): f"{int(d['trade_volume']):,}" for u, v, d in G.edges(data=True)},
    font_size=8
)
plt.title(f'Original Trade Network (US trade ×{1-M:.2f})')
plt.axis('off')
plt.tight_layout()
orig_path = os.path.join(fig_dir, f'original_trade_network_M{M:.2f}.png')
plt.savefig(orig_path, dpi=300)
plt.show()

# 6. Compute MST on modified graph
mst = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')

# 7. Plot and save adjusted MST
plt.figure(figsize=(12, 8))
nx.draw(mst, pos, with_labels=True, node_size=800,
        node_color='lightgreen', edge_color='black', width=2)
nx.draw_networkx_edge_labels(
    mst, pos,
    edge_labels={(u, v): f"{int(d['trade_volume']):,}" for u, v, d in mst.edges(data=True)},
    font_size=10
)
plt.title(f'MST After US Trade Reduction (×{1-M:.2f})')
plt.axis('off')
plt.tight_layout()
mst_path = os.path.join(fig_dir, f'trade_mst_M{M:.2f}.png')
plt.savefig(mst_path, dpi=300)
plt.show()

# 8. Print summary
total_cost = sum(d['weight'] for _, _, d in mst.edges(data=True))
total_trade = sum(d['trade_volume'] for _, _, d in mst.edges(data=True))
print(f"Total reciprocal-cost (MST) after M={M}: {total_cost:.6f}")
print(f"Total trade volume (MST) after M={M}: {total_trade:,.0f} USD")

# Display file paths
print(f"Saved original network plot to: {orig_path}")
print(f"Saved MST plot to: {mst_path}")
