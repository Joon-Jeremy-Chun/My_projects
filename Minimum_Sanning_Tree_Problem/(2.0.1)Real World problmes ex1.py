import networkx as nx
import matplotlib.pyplot as plt
import random

# ---------------------------
# 1) Define 20 real countries
# ---------------------------
countries = [
    'USA', 'China', 'Japan', 'Germany', 'UK', 'France', 
    'Italy', 'Canada', 'SouthKorea', 'Brazil', 'India', 
    'Russia', 'Australia', 'Spain', 'Mexico', 'Indonesia', 
    'Turkey', 'SaudiArabia', 'Netherlands', 'Switzerland'
]

# ---------------------------------------------
# 2) Generate random trade volumes (fake data)
# ---------------------------------------------
# We'll create roughly 35 random edges to ensure
# the graph is likely to be connected.
random.seed(42)  # for reproducibility
num_edges = 35
edges = []

# We'll pick random pairs of countries without repeating
# to avoid duplicates. We'll store used pairs in a set.
used_pairs = set()

while len(edges) < num_edges:
    # Randomly pick two different countries
    c1 = random.choice(countries)
    c2 = random.choice(countries)
    if c1 != c2:
        # Sort to avoid (c1, c2) vs (c2, c1) duplication
        pair = tuple(sorted([c1, c2]))
        if pair not in used_pairs:
            used_pairs.add(pair)
            # Random trade volume in billions, e.g. 1 to 900
            trade_volume = random.randint(50, 900)
            # Convert trade volume to cost (the smaller, the more preferable)
            # so we do cost = 1 / trade_volume
            cost = 1.0 / trade_volume
            
            edges.append((c1, c2, cost))

# -----------------------------------------
# 3) Build the weighted graph with NetworkX
# -----------------------------------------
G = nx.Graph()
G.add_weighted_edges_from(edges)

# Make sure we have all countries as nodes, 
# even if some didn't get an edge by random selection.
for country in countries:
    if country not in G:
        G.add_node(country)

# ------------------------------
# 4) Plot the original graph (optional)
# ------------------------------
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue')
# Edge labels = cost (reciprocal of trade volume)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title("Original Trade Network (Costs = 1/TradeVolume)")
plt.show()

# ------------------------------
# 5) Compute the MST
# ------------------------------
mst = nx.minimum_spanning_tree(G, algorithm='kruskal')

# ------------------------------
# 6) Plot the MST
# ------------------------------
plt.figure(figsize=(10, 8))
pos_mst = nx.spring_layout(mst, seed=42)
nx.draw(mst, pos_mst, with_labels=True, node_size=800, node_color='lightgreen')
mst_labels = nx.get_edge_attributes(mst, 'weight')
nx.draw_networkx_edge_labels(mst, pos_mst, edge_labels=mst_labels, font_color='blue')
plt.title("Minimum Spanning Tree of the Trade Network")
plt.show()

# ------------------------------
# 7) Identify the hub country
# ------------------------------
# The "hub" is the node with the highest degree in the MST.
degrees = dict(mst.degree())
hub_country = max(degrees, key=degrees.get)
print("Hub country in the MST:", hub_country)
print("Degree (number of MST edges):", degrees[hub_country])

# ------------------------------
# 8) (Optional) Print total MST cost
# ------------------------------
total_cost = sum(mst_labels.values())
print(f"Total MST cost (sum of 1/tradeVolume): {total_cost:.4f}")
