import networkx as nx
from fingerprint_matching import example  
from math import sqrt
import os
from itertools import product


def euclidean_distance(p1, p2):
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def build_mst(minutiae_list, root=None):
    if not root:
        root = minutiae_list[0]

    tree = nx.Graph()
    visited = set()
    remaining = set(minutiae_list)

    tree.add_node(root)
    visited.add(root)
    remaining.remove(root)

    while remaining:
        min_dist = float('inf')
        closest_pair = (None, None)
        for u in visited:
            for v in remaining:
                d = euclidean_distance(u, v)
                if d < min_dist:
                    min_dist = d
                    closest_pair = (u, v)
        u, v = closest_pair
        tree.add_edge(u, v, weight=min_dist)
        visited.add(v)
        remaining.remove(v)

    return tree

def assign_levels(tree, root):
    levels = {}
    queue = [(root, 0)]
    visited = set()

    while queue:
        node, level = queue.pop(0)
        if node not in visited:
            visited.add(node)
            levels[node] = level
            for neighbor in tree.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, level + 1))
    return levels

def compute_matching_score(tree1, tree2, level_weights={0: 0.4, 1: 0.3, 2: 0.1, 3: 0.1, 4: 0.1}):
    levels1 = assign_levels(tree1, list(tree1.nodes)[0])
    levels2 = assign_levels(tree2, list(tree2.nodes)[0])

    matched_score = 0
    total_possible = sum(level_weights.get(l, 0) for l in levels1.values())

    for n1, l1 in levels1.items():
        for n2, l2 in levels2.items():
            if l1 == l2 and euclidean_distance(n1, n2) < 15:  # match tolerance
                matched_score += level_weights.get(l1, 0)
                break

    return matched_score / total_possible if total_possible else 0.0



tre1= build_mst(example.FeaturesTerminations + example.FeaturesBifurcations)
#minutiae1 = minutiae1 + minutiae_list2

# tree1 = build_mst(minutiae1)
# tree2 = build_mst(minutiae2)

score = compute_matching_score(tre1, tree2)
print(f"Matching Score: {score * 100:.2f}%")
