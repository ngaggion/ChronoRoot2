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

import cv2 
import os
import numpy as np


def getImgName(image, conf):
    """Extract just the image filename from full path."""
    return image.replace(conf['ImagePath'], '').replace('/', '')


def plot_segmentation_overlay(graph, skeleton_overlay, hypocotyl_skeleton):
    """
    Create a color-coded visualization of the root system.
    
    Main root is shown in green, lateral roots in blue.
    
    Args:
        graph: NetworkX graph with edge attribute 'root_type' and 'color'
        skeleton_overlay: Skeleton image with color-coded segments
        
    Returns:
        colored_overlay: 3-channel image (BGR) with main root (green) and laterals (blue)
    """
    # Create separate masks for main root and lateral roots
    main_root_mask = np.zeros_like(skeleton_overlay).astype('uint8')
    lateral_root_mask = np.zeros_like(skeleton_overlay).astype('uint8')
    hypocotyl_mask = np.zeros_like(hypocotyl_skeleton).astype('uint8')
    
    # Iterate through all edges and mark pixels by root type
    for u, v, data in graph.edges(data=True):
        edge_color = data.get('color', 0)
        edge_type = data.get('root_type', 0)
        
        # Find pixels in skeleton that belong to this edge
        pixel_positions = np.where(skeleton_overlay == edge_color)
        
        if edge_type == 10:
            # Main root edge
            main_root_mask[pixel_positions] = 255
        else:
            # Lateral root edge
            lateral_root_mask[pixel_positions] = 255
    
    # Add hypocotyl skeleton to main root mask
    hypocotyl_mask[np.where(hypocotyl_skeleton > 0)] = 255
    
    # Dilate masks to make roots more visible
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    main_root_mask = cv2.dilate(main_root_mask, kernel)
    lateral_root_mask = cv2.dilate(lateral_root_mask, kernel)
    hypocotyl_mask = cv2.dilate(hypocotyl_mask, kernel)
    
    # Create 3-channel color image (BGR format for OpenCV)
    colored_overlay = np.zeros(list(skeleton_overlay.shape) + [3], dtype='uint8')
    colored_overlay[:, :, 1] = main_root_mask      # Green channel = main root
    colored_overlay[:, :, 0] = lateral_root_mask   # Blue channel = lateral roots
    
    # Hypocotyl should be yellow (Red + Green)
    colored_overlay[:, :, 1] = np.maximum(colored_overlay[:, :, 1], hypocotyl_mask)  # Green channel
    colored_overlay[:, :, 2] = np.maximum(colored_overlay[:, :, 2], hypocotyl_mask)  # Red channel
    
    # Draw Ini and FTip nodes in green and blue dots
    nodes = list(graph.nodes())
    for i in range(len(nodes)):
        node = graph.nodes[nodes[i]]
        node_type = node["type"]
        x, y = np.array(node["pos"])
        if node_type == 'Ini':
            cv2.circle(colored_overlay, (x, y), 8, (0, 0, 255), -1)  # Red dot
        elif node_type == 'FTip':
            cv2.circle(colored_overlay, (x, y), 8, (255, 255, 0), -1)  # Yellow dot

    return colored_overlay


def saveImages(conf, images, frame_idx, segmentation_mask, graph=None, skeleton_overlay=None, hypocotyl_skeleton=None):
    """
    Save visualization images for a frame.
    
    Saves three types of images:
    1. Input: Original cropped image
    2. Seg: Binary segmentation mask
    3. SegMulti: Color-coded root visualization (main root in green, laterals in blue)
    
    Args:
        conf: Configuration dictionary with folders and settings
        images: List of image paths
        frame_idx: Index of current frame
        segmentation_mask: Binary segmentation mask
        graph: NetworkX graph (None if frame failed)
        skeleton_overlay: Skeleton with color-coded segments (None if frame failed)
    """
    output_folder = conf['folders']['images']
    image_name = getImgName(images[frame_idx], conf)
    roi_bounds = conf['bounding box']
    
    # Only save if enabled in config
    if conf['saveImages']:
        # ----------------------------------------------------------------
        # Save 1: Original cropped image
        # ----------------------------------------------------------------
        original_image = cv2.imread(images[frame_idx])[
            roi_bounds[0]:roi_bounds[1], 
            roi_bounds[2]:roi_bounds[3]
        ]
        input_folder = os.path.join(output_folder, "Input")
        input_path = os.path.join(input_folder, image_name)
        cv2.imwrite(input_path, original_image)
    
    # ----------------------------------------------------------------
    # Save 2: Binary segmentation mask
    # ----------------------------------------------------------------
    seg_folder = os.path.join(output_folder, "Seg")
    seg_path = os.path.join(seg_folder, image_name)
    cv2.imwrite(seg_path, segmentation_mask)
    
    # ----------------------------------------------------------------
    # Save 3: Color-coded root visualization
    # ----------------------------------------------------------------
    if graph is None or graph is False or skeleton_overlay is None:
        # No valid graph - save black image
        colored_image = np.zeros_like(segmentation_mask).astype('uint8')
        # Convert to 3-channel for consistency
        colored_image = np.stack([colored_image] * 3, axis=-1)
    else:
        # Create color visualization from graph
        colored_image = plot_segmentation_overlay(graph, skeleton_overlay, hypocotyl_skeleton)
    
    segmulti_folder = os.path.join(output_folder, "SegMulti")
    segmulti_path = os.path.join(segmulti_folder, image_name)
    cv2.imwrite(segmulti_path, colored_image)
    
    return