def trimGraph(graph):
    """
    Clean up graph by removing artifacts and simplifying structure.
    
    Operations performed:
    1. Remove zero-weight edges and resulting isolated nodes
            
    Args:
        graph: NetworkX graph with node attributes (pos, type, age) 
               and edge attributes (weight, color, root_type)

    Returns:
        graph: Cleaned graph
    """
    
    # Remove zero-weight edges 
    graph = remove_zero_weight_edges(graph)
    
    return graph

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
    nodes_to_remove = set()
    
    # Identify zero-weight edges
    for u, v, data in graph.edges(data=True):
        if data.get('weight', 0) == 0:
            edges_to_remove.append((u, v))
            
            # Match old logic: if removing this edge leaves a node isolated
            # (or if it was a tip), mark for removal.
            if graph.degree(u) <= 1: nodes_to_remove.add(u)
            if graph.degree(v) <= 1: nodes_to_remove.add(v)
            
    if edges_to_remove:
        graph.remove_edges_from(edges_to_remove)
        
    # Clean up isolated nodes
    isolated = [n for n in nodes_to_remove if n in graph and graph.degree(n) == 0]
    graph.remove_nodes_from(isolated)
    
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
    
    # Iterate over nodes (snapshot of list to allow modification)
    for node in list(graph.nodes()):
        if node in nodes_to_remove: continue
        
        neighbors = list(graph.neighbors(node))
        
        if len(neighbors) == 2:
            n1, n2 = neighbors
            
            # --- LOOP PROTECTION (Matches 'if edge is None' from old code) ---
            if graph.has_edge(n1, n2):
                # A direct link already exists. Merging 'node' would collapse
                # a parallel path (bubble/loop). Skip this merge.
                continue
            # ---------------------------------------------------------------

            edge1 = graph.edges[node, n1]
            edge2 = graph.edges[node, n2]
            
            w1 = edge1.get('weight', 0)
            w2 = edge2.get('weight', 0)
            c1 = edge1.get('color', 0)
            c2 = edge2.get('color', 0)
            
            # Determine new color (Propagate dominant ID)
            if w1 == 0:
                new_color = c2
                skeleton_overlay[skeleton_overlay == c1] = c2
            elif w2 == 0:
                new_color = c1
                skeleton_overlay[skeleton_overlay == c2] = c1
            else:
                new_color = c2
                skeleton_overlay[skeleton_overlay == c1] = c2

            # Add shortcut edge
            graph.add_edge(n1, n2, 
                           weight=w1 + w2, 
                           color=new_color,
                           root_type=edge2.get('root_type', 0))
            
            nodes_to_remove.append(node)
            
    graph.remove_nodes_from(nodes_to_remove)
    return graph, skeleton_overlay