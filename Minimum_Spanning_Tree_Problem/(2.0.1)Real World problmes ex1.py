import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# 0. Make sure your 19×19 CSV is at this location:
FIG_DIR   = 'Figures'
CSV_NAME  = 'bilateral_trade_19x19_2022.csv'
CSV_PATH  = os.path.join(FIG_DIR, CSV_NAME)

# 1. Load the 19×19 export matrix for 2022
df = pd.read_csv(CSV_PATH, index_col=0)

# 2. Build the graph: weight = 1 / trade_volume, keep trade_volume on edges
G = nx.Graph()
G.add_nodes_from(df.index)
for i in df.index:
    for j in df.columns:
        if i == j:
            continue
        vol = df.at[i, j]
        if vol > 0:
            G.add_edge(i, j, weight=1.0/vol, trade_volume=vol)

# 3. Compute a fixed layout
pos = nx.spring_layout(G, seed=42)

# 4. DRAW & SAVE: Full Trade Network
plt.figure(figsize=(12,8))
nx.draw(G, pos,
        with_labels=True,
        node_color='skyblue',
        node_size=800,
        edge_color='gray',
        alpha=0.5)
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels={(u,v):f"{d['trade_volume']:,}" for u,v,d in G.edges(data=True)},
    font_size=8
)
plt.title('Original Trade Network (2022 exports in USD)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'original_trade_network.png'), dpi=300)
plt.show()

# 5. COMPUTE MST (minimize reciprocal = maximize trade)
mst = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')

# 6. DRAW & SAVE: MST
plt.figure(figsize=(12,8))
nx.draw(mst, pos,
        with_labels=True,
        node_color='lightgreen',
        node_size=800,
        edge_color='black',
        width=2)
nx.draw_networkx_edge_labels(
    mst, pos,
    edge_labels={(u,v):f"{d['trade_volume']:,}" for u,v,d in mst.edges(data=True)},
    font_size=10
)
plt.title('Minimum Trade‐Based Spanning Tree')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'trade_mst.png'), dpi=300)
plt.show()

# 7. SUMMARY
total_cost  = sum(d['weight']      for _,_,d in mst.edges(data=True))
total_trade = sum(d['trade_volume'] for _,_,d in mst.edges(data=True))
print(f"Total reciprocal‐cost (MST): {total_cost:.6f}")
print(f"Total trade volume (MST): {total_trade:,.0f} USD")
