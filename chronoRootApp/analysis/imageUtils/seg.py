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

from skimage.morphology import skeletonize
import cv2
import numpy as np
import cv2

def extract_root_segmentation(segmentation_path, roi_bounds, current_root_base, fixed_seed_position):
    # Load segmentation and crop to region of interest
    multi_class_mask = cv2.imread(segmentation_path, 0)[roi_bounds[0]:roi_bounds[1], roi_bounds[2]:roi_bounds[3]]
    
    # Remove anything above the original seed point (not part of root)
    multi_class_mask[0:fixed_seed_position[1], :] = 0

    # Combine segmentation classes (1, 2 are root classes)
    binary_mask = (multi_class_mask == 1) + (multi_class_mask == 2)
    binary_mask = np.array(binary_mask, dtype='uint8') * 255
    
    # Clean up mask with morphological operations 
    morph_kernel_size = 5
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.dilate(binary_mask, morph_kernel)
    binary_mask = cv2.erode(binary_mask, morph_kernel)
    
    morph_kernel_size = 3
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.erode(binary_mask, morph_kernel)
    binary_mask = cv2.dilate(binary_mask, morph_kernel)
    
    # Find all connected components
    connected_components, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components_by_area = [(cv2.contourArea(component), component) for component in connected_components]
    
    if len(components_by_area) == 0:
        return binary_mask, False
    
    # Sort components by area (largest first)
    components_by_area.sort(key=lambda x: x[0], reverse=True)
    
    # Find the component that contains or is near the root base
    for area, component in components_by_area:
        if area < 40:  # Skip tiny components
            break
        
        # Check if this component contains the current root base position
        distance_to_root_base = cv2.pointPolygonTest(component, (int(current_root_base[0]), int(current_root_base[1])), True)
        distance_to_root_base = np.abs(distance_to_root_base)
        contains_root_base = cv2.pointPolygonTest(component, (int(current_root_base[0]), int(current_root_base[1])), False) > 0
        
        if distance_to_root_base < 50 or contains_root_base:
            # This is the root - extract only this component
            component_mask = np.zeros(binary_mask.shape, np.uint8)
            cv2.drawContours(component_mask, [component], -1, 255, -1)
            filtered_mask = cv2.bitwise_and(component_mask, binary_mask.copy())
            return filtered_mask, True
    
    # No component found near root base
    return binary_mask, False


def extract_skeleton(binary_mask):
    # Convert binary mask to 1-pixel-wide skeleton
    skeleton = np.array(skeletonize(binary_mask // 255), dtype='uint8')

    # First cleanup pass
    skeleton = prune(skeleton, 5)  # Remove branches shorter than 7 pixels
    skeleton = trim(skeleton)
    
    # Second cleanup pass: remove shorter branches emerged after trimming
    skeleton = prune(skeleton, 3)  # Remove branches shorter than 5 pixels
    skeleton = trim(skeleton)

    # Third cleanup pass: remove shorter branches emerged after trimming
    skeleton = prune(skeleton, 3)  # Remove branches shorter than 3 pixels
    skeleton = trim(skeleton)
    
    # Identify branch points (where root splits) and end points (root tips)
    branch_points, end_points = skeleton_nodes(skeleton)
        
    # Valid skeleton must have at least 2 endpoints (seed + at least one tip)
    is_valid = len(end_points) >= 2
    
    return skeleton, branch_points, end_points, is_valid

def trim(ske): ## Removes unwanted pixels from the skeleton
    
    T=[]
    T0=np.array([[-1, 1, -1], 
                 [1, 1, 1], 
                 [0, 0, 0]]) # T0 contains X0
    T2=np.array([[-1, 1, 0], 
                 [1, 1, 0], 
                 [-1, 1, 0]])
    T4=np.array([[0, 0, 0], 
                 [1, 1, 1], 
                 [-1, 1, -1]])
    T6=np.array([[0, 1, -1], 
                 [0, 1, 1], 
                 [0, 1, -1]])
    S1=np.array([[1, -1, -1], 
                 [1, 1, -1], 
                 [-1, 1, -1]])
    S2=np.array([[-1, 1, -1], 
                 [1, 1, -1], 
                 [1, -1, -1]])
    S3=np.array([[-1, -1, -1], 
                 [1, 1, -1], 
                 [-1, 1, 1]])
    S4=np.array([[-1, -1, -1], 
                 [-1, 1, 1], 
                 [1, 1, -1]])
    S5=np.array([[-1, 1, 1], 
                 [1, 1, -1], 
                 [-1, -1, -1]])
    S6=np.array([[1, 1, -1], 
                 [-1, 1, 1], 
                 [-1, -1, -1]])
    S7=np.array([[-1, -1, 1], 
                 [-1, 1, 1], 
                 [-1, 1, -1]])
    S8=np.array([[-1, 1, -1], 
                 [-1, 1, 1], 
                 [-1, -1, 1]])
    C1=np.array([[-1, 1, -1], 
                 [-1, 1, 1], 
                 [-1, -1, -1]])
    C2=np.array([[-1, -1, -1], 
                 [-1, 1, 1], 
                 [-1, 1, -1]])
    C3=np.array([[-1, 1, -1], 
                 [1, 1, -1], 
                 [-1, -1, -1]])
    C4=np.array([[-1, -1, -1], 
                 [1, 1, -1], 
                 [-1, 1, -1]])
    
    T.append(T0)
    T.append(T2)
    T.append(T4)
    T.append(T6)
    T.append(S1)
    T.append(S2)
    T.append(S3)
    T.append(S4)
    T.append(S5)
    T.append(S6)
    T.append(S7)
    T.append(S8)    
    T.append(C1)
    T.append(C2)
    T.append(C3)
    T.append(C4)
    
    bp = np.zeros_like(ske)
    for t in T:
        bp = cv2.morphologyEx(ske, cv2.MORPH_HITMISS, t)
        ske = cv2.subtract(ske, bp)
    
    # ske = cv2.subtract(ske, bp)
    
    return ske


def prune(skel, num_it): ## Removes branches with length lower than num_it
    orig = skel
    
    endpoint1 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [0, 1, 0]])
    
    endpoint2 = np.array([[0, 1, 0],
                          [-1, 1, -1],
                          [-1, -1, -1]])
    
    endpoint4 = np.array([[0, -1, -1],
                          [1, 1, -1],
                          [0, -1, -1]])
    
    endpoint5 = np.array([[-1, -1, 0],
                          [-1, 1, 1],
                          [-1, -1, 0]])
    
    endpoint3 = np.array([[-1, -1, 1],
                          [-1, 1, -1],
                          [-1, -1, -1]])
    
    endpoint6 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [1, -1, -1]])
    
    endpoint7 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [-1, 1, -1]])
    
    endpoint8 = np.array([[1, -1, -1],
                          [-1, 1, -1],
                          [-1, -1, -1]])
    
    
    for i in range(0, num_it):
        ep1 = skel - cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint1)
        ep2 = ep1 - cv2.morphologyEx(ep1, cv2.MORPH_HITMISS, endpoint2)
        ep3 = ep2 - cv2.morphologyEx(ep2, cv2.MORPH_HITMISS, endpoint3)
        ep4 = ep3 - cv2.morphologyEx(ep3, cv2.MORPH_HITMISS, endpoint4)
        ep5 = ep4 - cv2.morphologyEx(ep4, cv2.MORPH_HITMISS, endpoint5)
        ep6 = ep5 - cv2.morphologyEx(ep5, cv2.MORPH_HITMISS, endpoint6)
        ep7 = ep6 - cv2.morphologyEx(ep6, cv2.MORPH_HITMISS, endpoint7)
        ep8 = ep7 - cv2.morphologyEx(ep7, cv2.MORPH_HITMISS, endpoint8)
        skel = ep8
        
    end = endPoints(skel)
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    
    for i in range(0, num_it):
        end = cv2.dilate(end, kernel)
        end = cv2.bitwise_and(end, orig)
        
    return cv2.bitwise_or(end, skel)


def endPoints(skel):
    endpoint1=np.array([[1, -1, -1],
                        [-1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint2=np.array([[-1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint3=np.array([[-1, -1, 1],
                        [-1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint4=np.array([[-1, -1, -1],
                        [1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint5=np.array([[-1, -1, -1],
                        [-1, 1, 1],
                        [-1, -1, -1]])
    
    endpoint6=np.array([[-1, -1, -1],
                        [-1, 1, -1],
                        [1, -1, -1]])
    
    endpoint7=np.array([[-1, -1, -1],
                        [-1, 1, -1],
                        [-1, 1, -1]])
    
    endpoint8=np.array([[-1, -1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1]])
    
    ep1 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint1)
    ep2 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint2)
    ep3 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint3)
    ep4 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint4)
    ep5 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint5)
    ep6 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint6)
    ep7 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint7)
    ep8 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint8)
    
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    return ep


def skeleton_nodes(ske):
    branch = branchedPoints(ske)
    end = endPoints(ske)
    
    bp = np.where(branch == 1)
    bnodes = []
    for i in range(len(bp[0])):
        bnodes.append([bp[1][i],bp[0][i]])
    
    ep = np.where(end == 1)
    enodes = []
    for i in range(len(ep[0])):
        enodes.append([ep[1][i],ep[0][i]])
    
    return np.array(bnodes), np.array(enodes)


def branchedPoints(skel):
    X=[]
    #cross X
    X0 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    X1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    X.append(X0)
    X.append(X1)
    
    #T like
    T=[]
    T0=np.array([[2, 1, 2], 
                 [1, 1, 1], 
                 [2, 2, 2]]) # T0 contains X0
    T1=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]]) # T1 contains X1
    T2=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    T3=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    T4=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    T5=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    T6=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    T7=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    
    #Y like
    Y=[]
    Y0=np.array([[1, 0, 1], [0, 1, 0], [2, 1, 2]])
    Y1=np.array([[0, 1, 0], [1, 1, 2], [0, 2, 1]])
    Y2=np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]])
    Y3=np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]])
    Y4=np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])
    Y5 = np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)
    
    bp = np.zeros(skel.shape, dtype=int)
    for x in X:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, x)
    for y in Y:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, y)
    for t in T:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, t)
        
    return bp