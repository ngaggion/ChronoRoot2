"""
Graph initialization and temporal tracking functions.
Assigns node types and tracks nodes across frames.
"""

import numpy as np
import networkx as nx


def graphInit(graph):
    """
    Initialize a newly created graph by assigning node types.
    
    Node types:
    - "Ini": Initial node (seed/root base) - topmost node (minimum y)
    - "FTip": Final tip (main root tip) - bottommost node (maximum y)
    - "Bif": Bifurcation (branch point) - degree > 1
    - "LTip": Lateral tip (side branch endpoint) - degree == 1
    
    Also initializes ages to 1 for seed and tip if only 2 nodes exist,
    and marks the main root path.
    
    Args:
        graph: NetworkX graph with node attribute 'pos' (x, y)
        
    Returns:
        graph: Same graph with updated 'type' and 'age' attributes
    """
    node_list = list(graph.nodes())
    
    if len(node_list) == 0:
        raise Exception("Cannot initialize empty graph")
    
    # Get all node positions
    positions = np.array([graph.nodes[node]['pos'] for node in node_list])
    
    # Find seed (topmost = minimum y coordinate)
    seed_idx = np.argmin(positions[:, 1])
    seed_node = node_list[seed_idx]
    seed_position = positions[seed_idx]
    
    # Find main root tip (bottommost = maximum y coordinate)
    tip_idx = np.argmax(positions[:, 1])
    tip_node = node_list[tip_idx]
    tip_position = positions[tip_idx]
    
    # Assign node types
    for i, node in enumerate(node_list):
        node_position = positions[i]
        
        if np.array_equal(node_position, seed_position):
            # This is the seed/root base
            graph.nodes[node]['type'] = "Ini"
        elif np.array_equal(node_position, tip_position):
            # This is the main root tip
            graph.nodes[node]['type'] = "FTip"
        else:
            # Check degree to determine type
            degree = graph.degree(node)
            if degree > 2:
                # Branch point (more than 2 neighbors)
                graph.nodes[node]['type'] = "Bif"
            elif degree == 1:
                # Endpoint (lateral root tip)
                graph.nodes[node]['type'] = "LTip"
            else:
                # Degree == 2: part of a chain 
                graph.nodes[node]['type'] = "null"
    
    # Special case: if graph has only 2 nodes (seed and tip)
    if len(node_list) == 2:
        # Mark the single edge as main root (root_type = 10)
        graph.edges[seed_node, tip_node]['root_type'] = 10
        # Set ages to 1
        graph.nodes[seed_node]['age'] = 1
        graph.nodes[tip_node]['age'] = 1
    else:
        # Find shortest weighted path from seed to tip
        main_root_path = nx.shortest_path(
            graph,
            source=seed_node, 
            target=tip_node, 
            weight='weight'
        )
        # Mark all edges on main root path
        for i in range(len(main_root_path) - 1):
            u = main_root_path[i]
            v = main_root_path[i + 1]
            graph.edges[u, v]['root_type'] = 10  # Main root marker

    return graph


def find_nearest_node(target_position, node_positions_array):
    """
    Find the single nearest node to target position.
    
    Args:
        target_position: (x, y) or [x, y] target
        node_positions_array: Nx2 array of node positions
        
    Returns:
        nearest_idx: Index of nearest node
        distance: Distance to nearest node
    """
    distances = np.linalg.norm(target_position - node_positions_array, axis=1)
    nearest_idx = np.argmin(distances)
    return nearest_idx, distances[nearest_idx]


def find_nearby_nodes(target_position, node_positions_array, distance_threshold=30):
    """
    Find all nodes within distance_threshold of target position.
    
    Args:
        target_position: (x, y) or [x, y] target
        node_positions_array: Nx2 array of node positions
        distance_threshold: Maximum distance to consider "nearby"
        
    Returns:
        nearby_indices: Array of indices of nearby nodes
        distances: Distances to those nodes
    """
    distances = np.linalg.norm(target_position - node_positions_array, axis=1)
    nearby_mask = distances < distance_threshold
    nearby_indices = np.where(nearby_mask)[0]
    
    return nearby_indices, distances[nearby_indices]


def matchGraphs(previous_graph, current_graph, max_movement=70):
    """
    Match nodes between consecutive frames to track growth over time.
    Propagates node ages and types from previous frame to current frame.

    Args:
        previous_graph: NetworkX graph from previous frame
        current_graph: NetworkX graph from current frame
        max_movement: Maximum pixels a node can move between frames (default 70)

    Returns:
        current_graph: Updated with propagated ages and types
    """
    
    # Get node lists and positions
    prev_nodes = list(previous_graph.nodes())
    curr_nodes = list(current_graph.nodes())
    
    if len(curr_nodes) == 0:
        raise Exception("Current graph has no nodes")
    elif len(prev_nodes) == 2:
        # Make a check that everything is ok if previous graph has only 2 nodes
        seed_node = [n for n in prev_nodes if previous_graph.nodes[n]['type'] == "Ini"][0]
        tip_node = [n for n in prev_nodes if previous_graph.nodes[n]['type'] == "FTip"][0]
        # seed should be higher than tip, this is a basic sanity check
        # can happen early on 
        if previous_graph.nodes[seed_node]['pos'][1] >= previous_graph.nodes[tip_node]['pos'][1]:
            current_graph = graphInit(current_graph)
            return current_graph

    curr_positions = np.array([current_graph.nodes[node]['pos'] for node in curr_nodes])
    
    # Track which current nodes have been matched
    matched_current_nodes = set()
    
    # ========================================================================
    # PHASE 1: Match SEED node (Ini)
    # ========================================================================
    
    prev_seed_nodes = [n for n in prev_nodes if previous_graph.nodes[n]['type'] == "Ini"]
    
    if len(prev_seed_nodes) == 0:
        raise Exception("Previous graph has no seed node - cannot track")
    
    prev_seed = prev_seed_nodes[0]
    prev_seed_position = np.array(previous_graph.nodes[prev_seed]['pos'])
    prev_seed_age = previous_graph.nodes[prev_seed]['age']
    
    # Find nearest node to previous seed position
    distances_to_seed = np.linalg.norm(curr_positions - prev_seed_position, axis=1)
    nearest_seed_idx = np.argmin(distances_to_seed)
    seed_distance = distances_to_seed[nearest_seed_idx]
    
    # FIX: Check if seed moved too far - use topology if so
    if seed_distance > max_movement:    
        print(f"Seed moved {seed_distance:.1f}px")
        raise Exception("Seed tracking lost - cannot reliably assign seed node")
    else:
        # Match the seed normally
        seed_node = curr_nodes[nearest_seed_idx]
        current_graph.nodes[seed_node]['age'] = prev_seed_age + 1
    
    current_graph.nodes[seed_node]['type'] = "Ini"
    matched_current_nodes.add(seed_node)
    
    # ========================================================================
    # PHASE 2: Match MAIN TIP (FTip)
    # ========================================================================
    
    prev_tip_nodes = [n for n in prev_nodes if previous_graph.nodes[n]['type'] == "FTip"]
    
    if len(prev_tip_nodes) > 0:
        prev_tip = prev_tip_nodes[0]
        prev_tip_position = np.array(previous_graph.nodes[prev_tip]['pos'])
        prev_tip_age = previous_graph.nodes[prev_tip]['age']
        
        # Calculate distances, excluding already matched seed
        distances_to_tip = np.linalg.norm(curr_positions - prev_tip_position, axis=1)
        
        # Create mask for valid candidates (exclude seed)
        valid_mask = np.ones(len(curr_nodes), dtype=bool)
        valid_mask[curr_nodes.index(seed_node)] = False
        
        # Find nearest among valid nodes
        valid_distances = distances_to_tip.copy()
        valid_distances[~valid_mask] = np.inf
        nearest_tip_idx = np.argmin(valid_distances)
        tip_distance = valid_distances[nearest_tip_idx]
        
        # FIX: Check if tip moved too far
        if tip_distance > max_movement:
            # Check how many nodes are below previous tip position
            below_mask = curr_positions[:, 1] > prev_tip_position[1]
            available_below = np.where(below_mask & valid_mask)[0]
            if len(available_below) == 0:
                # Check the distance between the two nearest nodes
                top2_indices = np.argsort(valid_distances)[:2]
                dist_between = np.abs(curr_positions[top2_indices[0], 1] - curr_positions[top2_indices[1], 1])
                if dist_between < max_movement:
                    # Two nodes are very close - ambiguous
                    raise Exception("Tip tracking ambiguous - two close candidates found")
                else:
                    # Assign the nearest node anyway
                    tip_node = curr_nodes[nearest_tip_idx]
                    current_graph.nodes[tip_node]['age'] = prev_tip_age + 1                
            elif len(available_below) > 1:
                # Multiple candidates below previous tip - ambiguous
                raise Exception("Tip tracking ambiguous - multiple candidates found")
            else:
                # Single candidate below previous tip - assign it
                tip_node = curr_nodes[available_below[0]]
                current_graph.nodes[tip_node]['age'] = prev_tip_age + 1
        else:
            # Match the tip normally
            tip_node = curr_nodes[nearest_tip_idx]
            current_graph.nodes[tip_node]['age'] = prev_tip_age + 1
        
        current_graph.nodes[tip_node]['type'] = "FTip"
        matched_current_nodes.add(tip_node)
    else:
        raise Exception("Previous graph has no tip node - cannot track")
    
    # ========================================================================
    # PHASE 3: Match REMAINING nodes - NEW nodes find OLD matches
    # ========================================================================

    # Get previous node data (excluding seed/tip already matched)
    prev_remaining = [n for n in prev_nodes 
                    if previous_graph.nodes[n]['type'] not in ["Ini", "FTip"]]
    prev_remaining_positions = np.array([previous_graph.nodes[n]['pos'] 
                                        for n in prev_remaining])

    # For each unmatched current node, find best previous match
    for curr_node in curr_nodes:
        # Skip already matched seed and tip
        if curr_node in matched_current_nodes:
            continue
        
        curr_pos = np.array(current_graph.nodes[curr_node]['pos'])
        
        if len(prev_remaining) == 0:
            # No previous nodes to match - this is a new node
            current_graph.nodes[curr_node]['age'] = 1
            degree = current_graph.degree(curr_node)
            if degree > 2:
                current_graph.nodes[curr_node]['type'] = "Bif"
            elif degree == 1:
                current_graph.nodes[curr_node]['type'] = "LTip"
            else:
                current_graph.nodes[curr_node]['type'] = "null"
            continue
        
        # Find nearest previous node
        distances = np.linalg.norm(prev_remaining_positions - curr_pos, axis=1)
        nearest_idx = np.argmin(distances)
        
        if distances[nearest_idx] < max_movement:
            # Match found - propagate age and type
            prev_match = prev_remaining[nearest_idx]
            current_graph.nodes[curr_node]['age'] = previous_graph.nodes[prev_match]['age'] + 1
            
            # Keep topology-based type if already assigned, otherwise use previous type
            if current_graph.nodes[curr_node]['type'] == "null":
                current_graph.nodes[curr_node]['type'] = previous_graph.nodes[prev_match]['type']
        else:
            # No match - new node
            current_graph.nodes[curr_node]['age'] = 1
            degree = current_graph.degree(curr_node)
            if degree > 2:
                current_graph.nodes[curr_node]['type'] = "Bif"
            elif degree == 1:
                current_graph.nodes[curr_node]['type'] = "LTip"
    
    # ========================================================================
    # PHASE 4: Mark MAIN ROOT path from seed to tip
    # ========================================================================
    
    try:
        # Find shortest weighted path from seed to tip
        main_root_path = nx.shortest_path(
            current_graph, 
            source=seed_node, 
            target=tip_node, 
            weight='weight'
        )
        
        # Mark all edges on main root path
        for i in range(len(main_root_path) - 1):
            u = main_root_path[i]
            v = main_root_path[i + 1]
            current_graph.edges[u, v]['root_type'] = 10  # Main root marker
            
    except nx.NetworkXNoPath:
        print("Warning: Graph disconnected - no path from seed to tip")
    
    return current_graph