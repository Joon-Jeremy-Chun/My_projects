import networkx as nx
import matplotlib
# Static backend for saving images
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq
from networkx.utils import UnionFind
import matplotlib.patches as mpatches
import os
import glob
from PIL import Image

# Styling constants
NODE_SIZE = 800
ALPHA = 0.3
ORIG_COLOR = 'skyblue'
KRUSKAL_COLOR = 'lightgreen'
PRIM_COLOR = 'lightcoral'
PAUSE_TIME = 2  # seconds per frame
FIG_DIR = 'Figures'

# Ensure output directory exists
os.makedirs(FIG_DIR, exist_ok=True)

# Create graph and weighted edges
G = nx.Graph()
edges = [
    ('LA', 'SD', 120), ('LA', 'LV', 270), ('LA', 'SF', 380),
    ('LA', 'SAC', 385), ('SD', 'LV', 330), ('LV', 'SF', 570),
    ('SF', 'SAC', 90), ('SAC', 'LV', 550), ('SD', 'PHX', 360),
    ('PHX', 'LV', 290), ('PHX', 'LA', 370)
]
G.add_weighted_edges_from(edges)
pos = nx.spring_layout(G, seed=42)

# Generate frames for Kruskal's algorithm

def save_kruskal_frames(G, pos):
    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
    uf = UnionFind(G.nodes())
    chosen = []
    frame = 0
    for u, v, d in edges_sorted:
        if uf[u] != uf[v]:
            uf.union(u, v)
            chosen.append((u, v, d))
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"Kruskal's Algorithm - Step {frame}", fontsize=14)
            nx.draw(G, pos, with_labels=True, node_size=NODE_SIZE,
                    node_color=ORIG_COLOR, alpha=ALPHA, ax=ax)
            T = nx.Graph()
            T.add_edges_from(chosen)
            nx.draw(T, pos, with_labels=True, node_size=NODE_SIZE,
                    node_color=KRUSKAL_COLOR, width=2, ax=ax)
            labels = nx.get_edge_attributes(T, 'weight')
            nx.draw_networkx_edge_labels(T, pos, edge_labels=labels, ax=ax)
            handles = [
                mpatches.Patch(color=ORIG_COLOR, label='Original Graph'),
                mpatches.Patch(color=KRUSKAL_COLOR, label='MST Edges')
            ]
            ax.legend(handles=handles, loc='upper left')
            plt.tight_layout()
            path = os.path.join(FIG_DIR, f'kruskal_{frame:02d}.png')
            fig.savefig(path)
            plt.close(fig)
            frame += 1
    return frame

# Generate frames for Prim's algorithm

def save_prim_frames(G, pos, start='LA'):
    visited = {start}
    heap = []
    for u, v, d in G.edges(start, data=True):
        heapq.heappush(heap, (d['weight'], start, v))
    chosen = []
    frame = 0
    while heap:
        w, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        chosen.append((u, v, {'weight': w}))
        for _, nbr, d in G.edges(v, data=True):
            if nbr not in visited:
                heapq.heappush(heap, (d['weight'], v, nbr))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"Prim's Algorithm - Step {frame}", fontsize=14)
        nx.draw(G, pos, with_labels=True, node_size=NODE_SIZE,
                node_color=ORIG_COLOR, alpha=ALPHA, ax=ax)
        T = nx.Graph()
        T.add_edges_from(chosen)
        nx.draw(T, pos, with_labels=True, node_size=NODE_SIZE,
                node_color=PRIM_COLOR, width=2, ax=ax)
        labels = nx.get_edge_attributes(T, 'weight')
        nx.draw_networkx_edge_labels(T, pos, edge_labels=labels, ax=ax)
        handles = [
            mpatches.Patch(color=ORIG_COLOR, label='Original Graph'),
            mpatches.Patch(color=PRIM_COLOR, label='Tree Edges')
        ]
        ax.legend(handles=handles, loc='upper left')
        plt.tight_layout()
        path = os.path.join(FIG_DIR, f'prim_{frame:02d}.png')
        fig.savefig(path)
        plt.close(fig)
        frame += 1
    return frame

# Generate and save frames
num_kruskal = save_kruskal_frames(G, pos)
num_prim = save_prim_frames(G, pos)
print(f'Saved {num_kruskal} Kruskal frames and {num_prim} Prim frames to {FIG_DIR}')

# Create separate GIFs for each algorithm
kruskal_frames = sorted(glob.glob(os.path.join(FIG_DIR, 'kruskal_*.png')))
prim_frames = sorted(glob.glob(os.path.join(FIG_DIR, 'prim_*.png')))

# Save Kruskal animation GIF
kruskal_images = [Image.open(f) for f in kruskal_frames]
kruskal_gif = os.path.join(FIG_DIR, 'kruskal_animation.gif')
kruskal_images[0].save(kruskal_gif, save_all=True, append_images=kruskal_images[1:], duration=PAUSE_TIME*1000, loop=0)
print(f'Kruskal GIF saved to {kruskal_gif}')

# Save Prim animation GIF
prim_images = [Image.open(f) for f in prim_frames]
prim_gif = os.path.join(FIG_DIR, 'prim_animation.gif')
prim_images[0].save(prim_gif, save_all=True, append_images=prim_images[1:], duration=PAUSE_TIME*1000, loop=0)
print(f'Prim GIF saved to {prim_gif}')
