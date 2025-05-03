import networkx as nx
import matplotlib.pyplot as plt

# Define the graph
G = nx.Graph()

# Adding weighted edges (road connections with construction costs)
edges = [
    ('LA', 'SD', 120),
    ('LA', 'LV', 270),
    ('LA', 'SF', 380),
    ('LA', 'SAC', 385),
    ('SD', 'LV', 330),
    ('LV', 'SF', 570),
    ('SF', 'SAC', 90),
    ('SAC', 'LV', 550),
    ('SD', 'PHX', 360),
    ('PHX', 'LV', 290),
    ('PHX', 'LA', 370)
]

G.add_weighted_edges_from(edges)

# Plot Original Graph
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_size=1200, node_color='skyblue')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Original Road Network (Costs in millions USD)')
plt.show()

# Calculate MST using Kruskal's algorithm
mst = nx.minimum_spanning_tree(G, algorithm='kruskal')

# Plot MST Graph
plt.figure(figsize=(10, 7))
nx.draw(mst, pos, with_labels=True, node_size=1200, node_color='lightgreen')
mst_labels = nx.get_edge_attributes(mst, 'weight')
nx.draw_networkx_edge_labels(mst, pos, edge_labels=mst_labels)
plt.title('Minimum Spanning Tree (Optimized Road Network)')
plt.show()

# Display total cost of MST
total_cost = sum(mst_labels.values())
print(f'Total construction cost (MST): {total_cost} million USD')
