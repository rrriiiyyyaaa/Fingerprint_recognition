import networkx as nx
import example  
import feature_extractor
import math
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


def mst_to_graph(mst_edges):
    """Convert MST edges list to NetworkX graph"""
    G = nx.Graph()
    for u, v, weight in mst_edges:
        G.add_edge(u, v, weight=weight)
    return G


def assign_levels(tree, root):
    """Assign levels to nodes in the tree starting from root"""
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
    """Compute matching score between two trees"""
    # Check if trees are empty
    if not tree1.nodes or not tree2.nodes:
        return 0.0
    
    # Get levels for both trees
    levels1 = assign_levels(tree1, list(tree1.nodes)[0])
    levels2 = assign_levels(tree2, list(tree2.nodes)[0])

    matched_score = 0
    used_nodes2 = set()  # Track which nodes in tree2 have been matched
    
    # Calculate total possible score based on tree1
    total_possible = sum(level_weights.get(l, 0) for l in levels1.values())

    # For each node in tree1, find best match in tree2
    for n1, l1 in levels1.items():
        best_match = None
        min_distance = float('inf')
        
        # Look for nodes at the same level in tree2
        for n2, l2 in levels2.items():
            if l1 == l2 and n2 not in used_nodes2:
                dist = euclidean_distance(n1, n2)
                if dist < 15 and dist < min_distance:  # match tolerance
                    min_distance = dist
                    best_match = n2
        
        # If we found a match, add to score and mark as used
        if best_match is not None:
            matched_score += level_weights.get(l1, 0)
            used_nodes2.add(best_match)

    return matched_score / total_possible if total_possible else 0.0


# Main execution
try:
    # Extract features from first image
    term, bif = feature_extractor.extract_and_print_features('enhanced/1.jpg')
    combined1 = term + bif
    
    if not combined1:
        print("No minutiae found in image 1")
        exit()
    
    # Build MST for first image
    mst_edges1 = build_mst(combined1)
    tree1 = mst_to_graph(mst_edges1)
    
    print(f"Tree 1: {len(tree1.nodes)} nodes, {len(tree1.edges)} edges")
    
    # Extract features from second image
    term, bif = feature_extractor.extract_and_print_features('enhanced/2.jpg')
    combined2 = term + bif
    
    if not combined2:
        print("No minutiae found in image 2")
        exit()
    
    # Build MST for second image
    mst_edges2 = build_mst(combined2)
    tree2 = mst_to_graph(mst_edges2)
    
    print(f"Tree 2: {len(tree2.nodes)} nodes, {len(tree2.edges)} edges")
    
    # Optional: Print MST edges for debugging
    print("\nMST 1 edges:")
    for edge in mst_edges1[:5]:  # Print first 5 edges
        print(f"From {edge[0]} to {edge[1]}, distance = {edge[2]:.2f}")
    
    print("\nMST 2 edges:")
    for edge in mst_edges2[:5]:  # Print first 5 edges
        print(f"From {edge[0]} to {edge[1]}, distance = {edge[2]:.2f}")
    
    # Compute matching score
    score = compute_matching_score(tree1, tree2)
    print(f"\nMatching Score: {score * 100:.2f}%")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()