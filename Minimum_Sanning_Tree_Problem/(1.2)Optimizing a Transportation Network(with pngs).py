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
import time

# Styling constants
NODE_SIZE = 800
ALPHA = 0.3
ORIG_COLOR = 'skyblue'
KRUSKAL_COLOR = 'lightgreen'
PRIM_COLOR = 'lightcoral'
PAUSE_TIME = 0.2  # 2초 대기
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
            # Label which algorithm and step number
            ax.set_title(f"Kruskal's Algorithm - Step {frame}")
            # Draw original graph faded
            nx.draw(G, pos, with_labels=True, node_size=NODE_SIZE,
                    node_color=ORIG_COLOR, alpha=ALPHA, ax=ax)
            # Draw current MST edges
            T = nx.Graph()
            T.add_edges_from(chosen)
            nx.draw(T, pos, with_labels=True, node_size=NODE_SIZE,
                    node_color=KRUSKAL_COLOR, width=2, ax=ax)
            # Draw edge weights
            labels = nx.get_edge_attributes(T, 'weight')
            nx.draw_networkx_edge_labels(T, pos, edge_labels=labels, ax=ax)
            # Add legend
            handles = [mpatches.Patch(color=ORIG_COLOR, label='Original Graph'),
                       mpatches.Patch(color=KRUSKAL_COLOR, label='MST Edges')]
            ax.legend(handles=handles, loc='upper left')
            plt.tight_layout()
            # Save frame
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
        # Label which algorithm and step number
        ax.set_title(f"Prim's Algorithm - Step {frame}")
        # Draw original graph faded
        nx.draw(G, pos, with_labels=True, node_size=NODE_SIZE,
                node_color=ORIG_COLOR, alpha=ALPHA, ax=ax)
        # Draw current tree edges
        T = nx.Graph()
        T.add_edges_from(chosen)
        nx.draw(T, pos, with_labels=True, node_size=NODE_SIZE,
                node_color=PRIM_COLOR, width=2, ax=ax)
        # Draw edge weights
        labels = nx.get_edge_attributes(T, 'weight')
        nx.draw_networkx_edge_labels(T, pos, edge_labels=labels, ax=ax)
        # Add legend
        handles = [mpatches.Patch(color=ORIG_COLOR, label='Original Graph'),
                   mpatches.Patch(color=PRIM_COLOR, label='Tree Edges')]
        ax.legend(handles=handles, loc='upper left')
        plt.tight_layout()
        # Save frame
        path = os.path.join(FIG_DIR, f'prim_{frame:02d}.png')
        fig.savefig(path)
        plt.close(fig)
        frame += 1
    return frame

# Generate and save frames
num_kruskal = save_kruskal_frames(G, pos)
num_prim = save_prim_frames(G, pos)
print(f'Saved {num_kruskal} Kruskal frames and {num_prim} Prim frames to {FIG_DIR}')

# Playback saved frames as animation

def play_frames(pattern):
    files = sorted(glob.glob(os.path.join(FIG_DIR, pattern)))
    plt.figure(figsize=(8, 6))
    for fpath in files:
        img = plt.imread(fpath)
        plt.imshow(img)
        plt.axis('off')
        plt.pause(PAUSE_TIME)
        plt.clf()
    plt.close()

# Play Kruskal then Prim
play_frames('kruskal_*.png')
play_frames('prim_*.png')
