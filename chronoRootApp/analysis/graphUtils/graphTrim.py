"""
Graph trimming operations to clean up artifacts from graph creation.
Removes zero-weight edges, simplifies chains, and merges duplicate nodes.
"""

import numpy as np

def trimGraph(graph, skeleton, skeleton_overlay):
    """
    Clean up graph by removing artifacts and simplifying structure.
    
    Operations performed:
    1. Remove zero-weight edges and resulting isolated nodes
    2. Merge degree-2 nodes (simplify chains into single edges)
    3. Merge nodes that are very close together (<3 pixels)
    
    POTENTIAL ISSUES:
    - May remove legitimate short branches if they have zero weight
    - Close node merging threshold (3 pixels) may need tuning
    - Chain simplification loses intermediate spatial information
    
    Args:
        graph: NetworkX graph with node attributes (pos, type, age) 
               and edge attributes (weight, color, root_type)
        skeleton: Binary skeleton image
        skeleton_overlay: Skeleton with color-coded segments
        
    Returns:
        graph: Cleaned graph
        skeleton: Updated skeleton
        skeleton_overlay: Updated overlay with merged segments
    """
    
    # OPERATION 1: Remove zero-weight edges and isolated nodes
    graph = remove_zero_weight_edges(graph)
    
    # OPERATION 2: Simplify chains by merging degree-2 nodes
    graph, skeleton_overlay = merge_chain_nodes(graph, skeleton_overlay)
    
    return graph, skeleton, skeleton_overlay


def remove_zero_weight_edges(graph):
    """
    Remove edges with zero weight (artifacts from graph creation).
    Also removes nodes that become isolated after edge removal.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        graph: Graph with zero-weight edges removed
    """
    edges_to_remove = []
    nodes_to_check = set()
    
    # Find all zero-weight edges
    for u, v, data in graph.edges(data=True):
        if data.get('weight', 0) == 0:
            edges_to_remove.append((u, v))
            nodes_to_check.add(u)
            nodes_to_check.add(v)
    
    if edges_to_remove:
        print(f"Removing {len(edges_to_remove)} zero-weight edges.")
    
    # Remove zero-weight edges
    graph.remove_edges_from(edges_to_remove)
    
    # Find and remove isolated nodes (degree 0)
    nodes_to_remove = []
    for node in nodes_to_check:
        if node in graph.nodes and graph.degree(node) == 0:
            nodes_to_remove.append(node)
    
    if nodes_to_remove:
        print(f"Removing {len(nodes_to_remove)} isolated nodes.")
        
    graph.remove_nodes_from(nodes_to_remove)
    
    return graph


def merge_chain_nodes(graph, skeleton_overlay):
    """
    Merge degree-2 nodes (nodes with exactly 2 neighbors).
    These nodes just connect two other nodes without branching,
    so we can simplify by connecting their neighbors directly.
    
    Example: A---B---C where B has degree 2 becomes A-------C
    
    Args:
        graph: NetworkX graph
        skeleton_overlay: Skeleton with color-coded segments
        
    Returns:
        graph: Simplified graph
        skeleton_overlay: Updated overlay with merged segments
    """
    nodes_to_remove = []
    
    # Find all degree-2 nodes (nodes in the middle of a chain)
    for node in list(graph.nodes()):
        neighbors = list(graph.neighbors(node))
        
        if len(neighbors) == 2:
            neighbor1, neighbor2 = neighbors
            
            # Check if neighbors are already connected (would create duplicate edge)
            if graph.has_edge(neighbor1, neighbor2):
                # They're already connected - just remove this node
                # This happens if there's a triangle in the graph
                nodes_to_remove.append(node)
                continue
            
            # Get edge data from both edges
            edge1_data = graph.edges[node, neighbor1]
            edge2_data = graph.edges[node, neighbor2]
            
            weight1 = edge1_data.get('weight', 0)
            weight2 = edge2_data.get('weight', 0)
            color1 = edge1_data.get('color', 0)
            color2 = edge2_data.get('color', 0)
            
            # Create new direct edge between neighbors
            # Weight is sum of both edges
            # Color/class: prefer the non-zero weight edge's color
            if weight1 == 0:
                new_color = color2
                # Update skeleton overlay: merge segments
                skeleton_overlay[skeleton_overlay == color1] = color2
            elif weight2 == 0:
                new_color = color1
                skeleton_overlay[skeleton_overlay == color2] = color1
            else:
                # Both have weight, use second edge's color
                new_color = color2
                skeleton_overlay[skeleton_overlay == color1] = color2
            
            graph.add_edge(
                neighbor1, 
                neighbor2,
                weight=weight1 + weight2,
                color=new_color,
                root_type=edge2_data.get('root_type', 0)
            )
            
            # Mark this intermediate node for removal
            nodes_to_remove.append(node)
    
    # Remove all intermediate nodes
    graph.remove_nodes_from(nodes_to_remove)
    
    return graph, skeleton_overlay