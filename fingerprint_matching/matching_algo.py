import networkx as nx
import example  
import feature_extractor
import math
import os
from itertools import product

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def build_mst(minutiae_list):
    # Convert minutiae to (x, y) tuples
    points = [(m.locX, m.locY) for m in minutiae_list]

    if not points:
        return []

    # Initialize sets for visited and unvisited nodes
    visited = set()
    mst_edges = []

    # Start with the first point
    current = points[0]
    visited.add(current)
    unvisited = set(points[1:])

    while unvisited:
        min_edge = None
        min_distance = float('inf')

        for u in visited:
            for v in unvisited:
                dist = euclidean_distance(u, v)
                if dist < min_distance:
                    min_distance = dist
                    min_edge = (u, v)

        if min_edge:
            u, v = min_edge
            mst_edges.append((u, v, min_distance))
            visited.add(v)
            unvisited.remove(v)

    return mst_edges


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

term, bif = feature_extractor.extract_and_print_features('enhanced/1.jpg')
combined = term + bif

tree_1 = build_mst(combined)

term, bif = feature_extractor.extract_and_print_features('enhanced/2.jpg')
combined = term + bif

tree_2 = build_mst(combined)
# for edge in tree1:
#     print(f"From {edge[0]} to {edge[1]}, distance = {edge[2]:.2f}")



score = compute_matching_score(tree_1, tree_2)
print(f"Matching Score: {score * 100:.2f}%")
