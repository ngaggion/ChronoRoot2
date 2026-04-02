""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import numpy as np
import networkx as nx

# Global counter for edge coloring/tracking
edge_color_counter = 3

def createGraph(skeleton_image, root_base_position, end_points, branch_points):
    """
    Create a graph from a skeleton image starting from the root base.
    """
    graph = nx.Graph()
    end_points = np.array(end_points)
    branch_points = np.array(branch_points)
    
    global edge_color_counter
    edge_color_counter = 3
    
    # Maintain multiclass and binarize in-place
    multiclass_skeleton = np.clip(skeleton_image, 0, 2).copy()
    skeleton_image[skeleton_image > 0] = 1
    
    actual_root_base, remaining_endpoints, distances = find_nearest(root_base_position, end_points)
    actual_root_base = tuple(actual_root_base)
    
    graph.add_node(
        actual_root_base,
        pos=actual_root_base,
        type='null',
        age=0
    )
    skeleton_image[actual_root_base[1], actual_root_base[0]] = edge_color_counter
    
    neighbor_pixels = find_neighbors(skeleton_image, actual_root_base)
    
    if len(neighbor_pixels) == 0:
        raise Exception("Root base has no neighbors in skeleton")
    
    skeleton_image, first_node_position, first_edge_length = get_next_node(
        skeleton_image, 
        multiclass_skeleton,
        neighbor_pixels[0],
        actual_root_base,
        [],
        0,
        actual_root_base
    )
    first_node_position = tuple(first_node_position)
    
    graph.add_node(
        first_node_position,
        pos=first_node_position,
        type='null',
        age=0
    )
    
    graph.add_edge(
        actual_root_base,
        first_node_position,
        weight=first_edge_length,
        color=edge_color_counter,
        root_type=0
    )
    
    edge_color_counter += 1
        
    remaining_endpoints_list = [tuple(ep) for ep in remaining_endpoints]
    
    if first_node_position not in remaining_endpoints_list:
        graph = continue_graph(
            graph,
            skeleton_image, 
            multiclass_skeleton,
            first_node_position,
            actual_root_base,  
            remaining_endpoints_list,
            branch_points
        )
    
    if graph.number_of_nodes() < 2:
        raise Exception("Graph has only one vertex - no structure detected")
    
    unvisited_pixels = np.sum(skeleton_image == 1)
    if unvisited_pixels > 20:
        raise Exception(f"Skeleton has unvisited pixels ({unvisited_pixels}) - incomplete graph")
    
    return graph, actual_root_base, skeleton_image


def continue_graph(graph, skeleton_image, multiclass_skeleton, current_position, parent_position, end_points_list, branch_points):
    """
    Recursively build graph by exploring branches from current position.
    """
    global edge_color_counter
    
    neighbor_pixels = find_neighbors(skeleton_image, current_position)
    
    for neighbor_start in neighbor_pixels:
        if skeleton_image[neighbor_start[1], neighbor_start[0]] != 1:
            neighbor_tuple = tuple(neighbor_start)
                        
            if neighbor_tuple in graph.nodes:
                if not graph.has_edge(current_position, neighbor_tuple):
                    dist = np.linalg.norm(np.array(current_position) - np.array(neighbor_start))
                    graph.add_edge(
                        current_position,
                        neighbor_tuple,
                        weight=dist,
                        color=edge_color_counter,
                        root_type=0
                    )
                    edge_color_counter += 1
            continue
        
        skeleton_image, next_node_position, edge_length = get_next_node(
            skeleton_image, 
            multiclass_skeleton,
            neighbor_start, 
            current_position,
            neighbor_pixels,
            0,
            current_position
        )
        next_node_position = tuple(next_node_position)
        
        if next_node_position == parent_position:
            continue
        
        if next_node_position not in graph.nodes:
            graph.add_node(
                next_node_position,
                pos=next_node_position,
                type='null',
                age=0
            )
            
        if not graph.has_edge(current_position, next_node_position):
            graph.add_edge(
                current_position,
                next_node_position,
                weight=edge_length,
                color=edge_color_counter,
                root_type=0
            )
            edge_color_counter += 1
        
        if skeleton_image[next_node_position[1], next_node_position[0]] == 1:
            skeleton_image[next_node_position[1], next_node_position[0]] = edge_color_counter
            
        if next_node_position not in end_points_list:
            graph = continue_graph(
                graph,
                skeleton_image, 
                multiclass_skeleton,
                next_node_position,
                current_position,  
                end_points_list,
                branch_points
            )
    
    return graph


def get_next_node(skeleton_image, multiclass_skeleton, current_pixel, parent_pixel, sibling_pixels, accumulated_distance, initial_position):
    """
    Trace along skeleton from current pixel until reaching a node (branch/endpoint).
    """
    global edge_color_counter
    
    if accumulated_distance == 0:
        distance_to_current = np.linalg.norm(np.array(current_pixel) - np.array(initial_position))
        accumulated_distance = distance_to_current
    
    while True:
        neighbor_pixels = find_neighbors(skeleton_image, current_pixel)
        
        valid_children = []
        for neighbor in neighbor_pixels:
            is_parent = np.array_equal(neighbor, parent_pixel)
            is_sibling = neighbor in sibling_pixels
            if not is_parent and not is_sibling:
                valid_children.append(neighbor)
        
        # --- LOOK-AHEAD LOGIC START ---
        color_changed = False
        if not np.array_equal(current_pixel, initial_position):
            if multiclass_skeleton[current_pixel[1], current_pixel[0]] != multiclass_skeleton[parent_pixel[1], parent_pixel[0]]:
                color_changed = True
                
                # If there's exactly 1 valid child, we are on a straight line. 
                # Let's peek at the next pixel to see if it's a structural node.
                if len(valid_children) == 1:
                    next_pixel_candidate = valid_children[0]
                    next_neighbors = find_neighbors(skeleton_image, next_pixel_candidate)
                    
                    # How many forward paths does the NEXT pixel have?
                    next_valid_children = [n for n in next_neighbors if not np.array_equal(n, current_pixel)]
                    
                    # If the next pixel is an endpoint (0 forward paths) or branch (>1 forward paths),
                    # we ignore the color change here. We let the loop run one more time so it 
                    # naturally cuts at the structural node.
                    if len(next_valid_children) != 1:
                        color_changed = False
        # --- LOOK-AHEAD LOGIC END ---

        if not np.array_equal(current_pixel, initial_position):
            skeleton_image[current_pixel[1], current_pixel[0]] = edge_color_counter
        
        # Stop if structural node OR confirmed color boundary
        if len(valid_children) != 1 or color_changed:
            return skeleton_image, current_pixel, accumulated_distance
        
        skeleton_image[current_pixel[1], current_pixel[0]] = edge_color_counter
        
        next_pixel = valid_children[0]
        distance_increment = np.linalg.norm(np.array(current_pixel) - np.array(next_pixel))
        accumulated_distance += distance_increment
        
        parent_pixel = current_pixel
        current_pixel = next_pixel
        sibling_pixels = []


def find_neighbors(skeleton_image, pixel_position, search_value=1):
    """
    Find 8-connected neighbors of a pixel with a specific value.
    """
    neighbors = []
    x, y = pixel_position[0], pixel_position[1]
    height, width = skeleton_image.shape
    
    for i in range(y + 1, y - 2, -1):
        for j in range(x - 1, x + 2):
            if 0 <= i < height and 0 <= j < width:
                if skeleton_image[i, j] == search_value:
                    if not (x == j and y == i):
                        neighbors.append([j, i])
    
    return neighbors


def find_nearest(target_position, point_list):
    """
    Find the point in point_list nearest to target_position.
    """
    if len(point_list) == 0:
        return None, point_list, None
    
    distances = np.linalg.norm(target_position - point_list, axis=1)
    nearest_index = np.argmin(distances)
    nearest_point = point_list[nearest_index, :]
    remaining_points = np.delete(point_list, nearest_index, axis=0)
    
    return nearest_point, remaining_points, distances