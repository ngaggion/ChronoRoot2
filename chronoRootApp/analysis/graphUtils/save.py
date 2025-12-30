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

import os
import gzip
import networkx as nx

def saveGraph(graph, conf, image_name):
    """
    Save graph structure to disk as compressed GraphML (XML format).
    
    Args:
        graph: NetworkX graph with node attributes (pos, type, age) 
               and edge attributes (weight, color, root_type)
               Can also be False/None for failed frames
        conf: Configuration dictionary with output folders
        image_name: Name of the image file (will replace .png with .xml.gz)
    """
    save_filename = image_name.replace('.png', '.xml.gz')
    save_path = os.path.join(conf['folders']['graphs'], save_filename)
    
    if graph is not False and graph is not None:
        # Convert tuple positions to strings for GraphML compatibility
        # GraphML doesn't natively support tuple attributes
        graph_copy = graph.copy()
        
        for node in graph_copy.nodes():
            pos = graph_copy.nodes[node]['pos']
            # Store position as "x,y" string
            graph_copy.nodes[node]['pos_x'] = float(pos[0])
            graph_copy.nodes[node]['pos_y'] = float(pos[1])
            # Remove tuple attribute (GraphML can't serialize tuples)
            del graph_copy.nodes[node]['pos']
        
        # Write compressed GraphML
        with gzip.open(save_path, 'wb') as f:
            # Convert to bytes for gzip
            graphml_string = '\n'.join(nx.generate_graphml(graph_copy))
            f.write(graphml_string.encode('utf-8'))
    else:
        print(f'Not valid graph for {image_name}')
    
    return


def saveProps(image_name, frame_number, graph, csv_writer, number_lateral_roots, hypocotyl_length):
    """
    Extract and save measurements from the graph to CSV.
    
    Args:
        image_name: Name of the image file
        frame_number: Frame index in sequence
        graph: NetworkX graph (or False/None for failed frames)
        csv_writer: CSV writer object
        number_lateral_roots: Count of lateral roots from RSML creation
    """
    if graph is not False and graph is not None:
        # Calculate lengths by summing edge weights
        main_root_length = 0
        lateral_roots_length = 0
        total_length = 0
        
        for u, v, data in graph.edges(data=True):
            edge_weight = data.get('weight', 0)
            edge_type = data.get('root_type', 0)
            
            total_length += edge_weight
            
            if edge_type == 10:
                # Main root edge
                main_root_length += edge_weight
            else:
                # Lateral root edge
                lateral_roots_length += edge_weight
        
        row = [
            image_name, 
            frame_number, 
            main_root_length, 
            lateral_roots_length, 
            number_lateral_roots, 
            total_length,
            hypocotyl_length
        ]
    else:
        # No valid graph - write zeros
        row = [image_name, frame_number, 0, 0, 0, 0, 0]
    
    csv_writer.writerow(row)
    return

def loadGraph(filepath):
    """
    Load a previously saved graph from disk.
    
    Reconstructs the graph from compressed GraphML format and
    restores the position tuples from separate x,y attributes.
    
    Args:
        filepath: Path to the .xml.gz file
        
    Returns:
        graph: NetworkX graph with all attributes restored
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    
    # Read compressed GraphML
    with gzip.open(filepath, 'rb') as f:
        graph = nx.read_graphml(f)
    
    # Reconstruct position tuples from separate x,y attributes
    for node in graph.nodes():
        pos_x = float(graph.nodes[node].get('pos_x', 0))
        pos_y = float(graph.nodes[node].get('pos_y', 0))
        graph.nodes[node]['pos'] = (pos_x, pos_y)
    
    return graph
