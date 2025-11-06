""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import networkx as nx

# Global counter for edge coloring/tracking
edge_color_counter = 3

def createGraph(skeleton_image, root_base_position, end_points, branch_points):
    """
    Create a graph from a skeleton image starting from the root base.
        
    Args:
        skeleton_image: Binary skeleton (will be modified - values written for tracking)
        root_base_position: (x, y) where root starts
        end_points: List of (x, y) endpoint coordinates
        branch_points: List of (x, y) branch point coordinates
        
    Returns:
        graph: NetworkX Graph with node attributes (pos, type, age) and edge attributes (weight, color, root_type)
        actual_root_base: The endpoint closest to root_base_position (snapped to skeleton)
        marked_skeleton: skeleton_image with values marked during traversal
    """
    # Initialize graph structure
    graph = nx.Graph()
    end_points = np.array(end_points)
    branch_points = np.array(branch_points)
    
    global edge_color_counter
    edge_color_counter = 3
    
    actual_root_base, remaining_endpoints, distances = find_nearest(root_base_position, end_points)
    actual_root_base = tuple(actual_root_base)  # Convert to tuple for use as node ID
    
    # Create root node at the actual skeleton position
    graph.add_node(
        actual_root_base,
        pos=actual_root_base,
        type='null',
        age=0
    )
    skeleton_image[actual_root_base[1], actual_root_base[0]] = edge_color_counter
    
    neighbor_pixels = find_neighbors(skeleton_image, actual_root_base)
    
    if len(neighbor_pixels) == 0:
        # Root base has no neighbors - isolated point
        raise Exception("Root base has no neighbors in skeleton")
    
    skeleton_image, first_node_position, first_edge_length = get_next_node(
        skeleton_image, 
        neighbor_pixels[0],  # Start from first neighbor
        actual_root_base,    # Coming from root
        [],                   # No siblings yet
        0,                    # Distance accumulator
        actual_root_base      # Initial start position for distance calculation
    )
    first_node_position = tuple(first_node_position)
    
    # Create second vertex and first edge
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
        root_type=0  # 0=unknown, will be set later
    )
    
    edge_color_counter += 1
        
    # Convert remaining endpoints to list of tuples for checking
    remaining_endpoints_list = [tuple(ep) for ep in remaining_endpoints]
    
    if first_node_position not in remaining_endpoints_list:
        # This is a branch point, not an endpoint - continue building graph
        graph = continue_graph(
            graph,
            skeleton_image, 
            first_node_position,
            actual_root_base,  
            remaining_endpoints_list,
            branch_points
        )
    
    if graph.number_of_nodes() < 2:
        raise Exception("Graph has only one vertex - no structure detected")
    
    return graph, actual_root_base, skeleton_image

def continue_graph(graph, skeleton_image, current_position, parent_position, end_points_list, branch_points):
    """
    Recursively build graph by exploring branches from current position.
    
    Args:
        graph: NetworkX Graph object
        skeleton_image: The skeleton being traversed (modified)
        current_position: (x, y) tuple - current location in skeleton
        parent_position: (x, y) tuple - node we came from (to avoid backtracking)
        end_points_list: List of endpoint coordinate tuples
        branch_points: Array of branch point coordinates
    """
    global edge_color_counter
    
    # Find all neighboring skeleton pixels from current position
    neighbor_pixels = find_neighbors(skeleton_image, current_position)
    
    # Explore each branch from this point
    for neighbor_start in neighbor_pixels:
        
        # Trace from this neighbor to the next node (branch or endpoint)
        skeleton_image, next_node_position, edge_length = get_next_node(
            skeleton_image, 
            neighbor_start, 
            current_position,
            neighbor_pixels,  # Siblings to avoid backtracking
            0,
            current_position  # Initial start position for distance calculation
        )
        next_node_position = tuple(next_node_position)
        
        # Skip if this would create an edge back to parent
        if next_node_position == parent_position:
            continue
        
        # Check if this node already exists in graph (handles cycles/reconnections)
        if next_node_position not in graph.nodes:
            # New node - add it
            graph.add_node(
                next_node_position,
                pos=next_node_position,
                type='null',
                age=0
            )
            
        # Create edge to this node (avoid duplicate edges)
        if not graph.has_edge(current_position, next_node_position):
            graph.add_edge(
                current_position,
                next_node_position,
                weight=edge_length,
                color=edge_color_counter,
                root_type=0
            )
            edge_color_counter += 1
        
        # Mark this pixel as visited
        if skeleton_image[next_node_position[1], next_node_position[0]] == 1:
            skeleton_image[next_node_position[1], next_node_position[0]] = edge_color_counter
            
        # If not an endpoint, continue recursively
        if next_node_position not in end_points_list:
            graph = continue_graph(
                graph,
                skeleton_image, 
                next_node_position,
                current_position,  
                end_points_list,
                branch_points
            )
    
    return graph

'''
def get_next_node(skeleton_image, current_pixel, parent_pixel, sibling_pixels, accumulated_distance, initial_position):
    """
    Trace along skeleton from current pixel until reaching a node (branch/endpoint).
    Recursively follows the skeleton path.
    
    Args:
        skeleton_image: Skeleton being traversed
        current_pixel: Current [x, y] position
        parent_pixel: Where we came from (to avoid backtracking)
        sibling_pixels: Other branches from parent (to avoid)
        accumulated_distance: Total distance traveled so far
        initial_position: Starting position for distance calculation
        
    Returns:
        skeleton_image: Modified with markings
        node_position: [x, y] of the node we reached
        total_distance: Length of path traveled
    """
    global edge_color_counter
    
    # Find neighbors of current pixel
    neighbor_pixels = find_neighbors(skeleton_image, current_pixel)
    
    # Filter out parent and siblings (avoid backtracking)
    valid_children = []
    for neighbor in neighbor_pixels:
        is_parent = np.array_equal(neighbor, parent_pixel)
        is_sibling = neighbor in sibling_pixels
        if not is_parent and not is_sibling:
            valid_children.append(neighbor)
    
    # Calculate distance from initial position to current pixel if this is first call
    if accumulated_distance == 0:
        # First step - calculate distance from initial position
        distance_to_current = np.linalg.norm(np.array(current_pixel) - np.array(initial_position))
        accumulated_distance = distance_to_current
    
    # Mark pixel if we've moved from initial position
    if not np.array_equal(current_pixel, initial_position):
        skeleton_image[current_pixel[1], current_pixel[0]] = edge_color_counter
    
    # Stop condition: reached a node (0 or multiple children)
    if len(valid_children) != 1:
        # This is either: an endpoint (0 children) or a branch point (>1 children)
        return skeleton_image, current_pixel, accumulated_distance
    
    # Continue along the path
    skeleton_image[current_pixel[1], current_pixel[0]] = edge_color_counter
    
    next_pixel = valid_children[0]
    distance_increment = np.linalg.norm(np.array(current_pixel) - np.array(next_pixel))
    new_distance = accumulated_distance + distance_increment
    
    # RECURSION: Continue to next pixel
    return get_next_node(skeleton_image, next_pixel, current_pixel, [], new_distance, initial_position)
'''

def get_next_node(skeleton_image, current_pixel, parent_pixel, sibling_pixels, accumulated_distance, initial_position):
    """
    Trace along skeleton from current pixel until reaching a node (branch/endpoint).
    Uses iteration instead of recursion to avoid stack overflow.
    
    Args:
        skeleton_image: Skeleton being traversed
        current_pixel: Current [x, y] position
        parent_pixel: Where we came from (to avoid backtracking)
        sibling_pixels: Other branches from parent (to avoid)
        accumulated_distance: Total distance traveled so far
        initial_position: Starting position for distance calculation
        
    Returns:
        skeleton_image: Modified with markings
        node_position: [x, y] of the node we reached
        total_distance: Length of path traveled
    """
    global edge_color_counter
    
    # Calculate initial distance if starting
    if accumulated_distance == 0:
        distance_to_current = np.linalg.norm(np.array(current_pixel) - np.array(initial_position))
        accumulated_distance = distance_to_current
    
    # Iteratively follow the path
    while True:
        neighbor_pixels = find_neighbors(skeleton_image, current_pixel)
        
        # Filter out parent and siblings (avoid backtracking)
        valid_children = []
        for neighbor in neighbor_pixels:
            is_parent = np.array_equal(neighbor, parent_pixel)
            is_sibling = neighbor in sibling_pixels
            if not is_parent and not is_sibling:
                valid_children.append(neighbor)
        
        # Mark pixel if not initial position
        if not np.array_equal(current_pixel, initial_position):
            skeleton_image[current_pixel[1], current_pixel[0]] = edge_color_counter
        
        # Stop condition: reached a node (0 or multiple children)
        if len(valid_children) != 1:
            return skeleton_image, current_pixel, accumulated_distance
        
        # Mark pixel before continuing (matches original behavior)
        skeleton_image[current_pixel[1], current_pixel[0]] = edge_color_counter
        
        # Move to next pixel
        next_pixel = valid_children[0]
        distance_increment = np.linalg.norm(np.array(current_pixel) - np.array(next_pixel))
        accumulated_distance += distance_increment
        
        # Update for next iteration
        parent_pixel = current_pixel
        current_pixel = next_pixel
        sibling_pixels = []

def find_neighbors(skeleton_image, pixel_position, search_value=1):
    """
    Find 8-connected neighbors of a pixel with a specific value.

    Args:
        skeleton_image: Image to search
        pixel_position: [x, y] or (x, y) position
        search_value: Value to search for (default 1 = skeleton)
        
    Returns:
        neighbors: List of [x, y] positions of matching neighbors
    """
    neighbors = []
    x, y = pixel_position[0], pixel_position[1]
    height, width = skeleton_image.shape
    
    # Search 3x3 neighborhood (8-connected)
    for i in range(y + 1, y - 2, -1):  # y+1, y, y-1
        for j in range(x - 1, x + 2):   # x-1, x, x+1
            # Check bounds
            if 0 <= i < height and 0 <= j < width:
                if skeleton_image[i, j] == search_value:
                    if not (x == j and y == i):
                        neighbors.append([j, i])
    
    return neighbors


def find_nearest(target_position, point_list):
    """
    Find the point in point_list nearest to target_position.
    
    Args:
        target_position: (x, y) or [x, y] target
        point_list: Nx2 array of points
        
    Returns:
        nearest_point: Closest point to target (as array)
        remaining_points: point_list with nearest_point removed
        distances: Array of all distances (for debugging)
    """
    if len(point_list) == 0:
        return None, point_list, None
    
    distances = np.linalg.norm(target_position - point_list, axis=1)
    nearest_index = np.argmin(distances)
    nearest_point = point_list[nearest_index, :]
    remaining_points = np.delete(point_list, nearest_index, axis=0)
    
    return nearest_point, remaining_points, distances