""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

Optimized for efficiency by processing expensive operations in ROIs 
and pre-allocating morphological kernels.
"""

from skimage.morphology import skeletonize
import cv2
import numpy as np

# --- PRE-ALLOCATED KERNELS (Global Scope) ---
# Moving these out of the functions prevents re-allocation on every loop/call.

# Kernels for trim()
T_KERNELS = [
    np.array([[-1, 1, -1], [1, 1, 1], [0, 0, 0]]),
    np.array([[-1, 1, 0], [1, 1, 0], [-1, 1, 0]]),
    np.array([[0, 0, 0], [1, 1, 1], [-1, 1, -1]]),
    np.array([[0, 1, -1], [0, 1, 1], [0, 1, -1]]),
    np.array([[1, -1, -1], [1, 1, -1], [-1, 1, -1]]),
    np.array([[-1, 1, -1], [1, 1, -1], [1, -1, -1]]),
    np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, 1]]),
    np.array([[-1, -1, -1], [-1, 1, 1], [1, 1, -1]]),
    np.array([[-1, 1, 1], [1, 1, -1], [-1, -1, -1]]),
    np.array([[1, 1, -1], [-1, 1, 1], [-1, -1, -1]]),
    np.array([[-1, -1, 1], [-1, 1, 1], [-1, 1, -1]]),
    np.array([[-1, 1, -1], [-1, 1, 1], [-1, -1, 1]]),
    np.array([[-1, 1, -1], [-1, 1, 1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, -1]]),
    np.array([[-1, 1, -1], [1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, -1]])
]

# Kernels for prune() and endPoints()
EP_KERNELS = [
    np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, 1, -1], [-1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, 1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, -1], [-1, 1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
]

# Kernels for branchedPoints()
X_KERNELS = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
]

T_BRANCH_KERNELS = [
    np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]]),
    np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]]),
    np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]]),
    np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]]),
    np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]]),
    np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]]),
    np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]]),
    np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
]

Y_KERNELS = [
    np.array([[1, 0, 1], [0, 1, 0], [2, 1, 2]]),
    np.array([[0, 1, 0], [1, 1, 2], [0, 2, 1]]),
    np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]]),
    np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]]),
    np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])
]
# Add rotated versions to Y_KERNELS
_Y3 = np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]])
_Y4 = np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])
Y_KERNELS.append(np.rot90(_Y3))
Y_KERNELS.append(np.rot90(_Y4))
Y_KERNELS.append(np.rot90(np.rot90(_Y3)))


def get_roi_bounding_box(mask, padding=5):
    """
    Helper to find the bounding box of non-zero pixels.
    Returns slices for cropping and offsets for coordinate restoration.
    """
    points = cv2.findNonZero(mask)
    if points is None:
        return None, None
    
    x, y, w, h = cv2.boundingRect(points)
    
    # Apply padding while staying within image bounds
    h_img, w_img = mask.shape
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(w_img, x + w + padding)
    y_end = min(h_img, y + h + padding)
    
    roi_slice = (slice(y_start, y_end), slice(x_start, x_end))
    offset = (x_start, y_start)
    
    return roi_slice, offset


def extract_root_segmentation(segmentation_path, roi_bounds, current_root_base, fixed_seed_position):
    # Load segmentation and crop to region of interest
    multi_class_mask = cv2.imread(segmentation_path, 0)[roi_bounds[0]:roi_bounds[1], roi_bounds[2]:roi_bounds[3]]
    
    # Extract hypocotyl length
    hypocotyl_skeleton, hypocotyl_length = extract_hypocotyl_length(multi_class_mask)
    
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
        return binary_mask, hypocotyl_skeleton, hypocotyl_length, False
    
    # Sort components by area (largest first)
    components_by_area.sort(key=lambda x: x[0], reverse=True)
    
    # Find the component that contains or is near the root base
    for area, component in components_by_area:
        if area < 30:  # Skip tiny components
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
            
            return filtered_mask, hypocotyl_skeleton, hypocotyl_length, True
    
    # No component found near root base
    return binary_mask, hypocotyl_skeleton, hypocotyl_length, False

def extract_hypocotyl_length(multi_class_mask):
    # Filter Mask for Hypocotyls (Class 4)
    binary_mask = (multi_class_mask == 4) 
    binary_mask = np.array(binary_mask, dtype='uint8') * 255
    
    morph_kernel_size = 5
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.dilate(binary_mask, morph_kernel)
    binary_mask = cv2.erode(binary_mask, morph_kernel)
    
    morph_kernel_size = 3
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.erode(binary_mask, morph_kernel)
    binary_mask = cv2.dilate(binary_mask, morph_kernel)
    
    connected_components, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Pre-calculate Area and BoundingRect for valid components only
    components_data = []
    for comp in connected_components:
        area = cv2.contourArea(comp)
        if area > 30: 
            rect = cv2.boundingRect(comp)
            components_data.append((area, comp, rect))
    
    if not components_data:
        return np.zeros_like(binary_mask), 0
    
    # Sort: Largest area first
    components_data.sort(key=lambda x: x[0], reverse=True)
    
    component_mask = np.zeros(binary_mask.shape, np.uint8)
    accepted_rects = [] 
    GAP_THRESHOLD = 100 
    
    for i, (_, component, rect) in enumerate(components_data):
        # Always keep the largest component (Index 0)
        if i == 0:
            cv2.drawContours(component_mask, [component], -1, 255, -1)
            accepted_rects.append(rect)
            continue
        
        # Efficient Proximity Check 
        # We check this new component against all previously accepted ones.
        x1, y1, w1, h1 = rect
        is_close = False
        
        for (rx, ry, rw, rh) in accepted_rects:
            # Calculate gap distance using simple arithmetic (no square roots)
            # Horizontal gap
            if x1 + w1 < rx: h_gap = rx - (x1 + w1)
            elif rx + rw < x1: h_gap = x1 - (rx + rw)
            else: h_gap = 0
            
            # Vertical gap
            if y1 + h1 < ry: v_gap = ry - (y1 + h1)
            elif ry + rh < y1: v_gap = y1 - (ry + rh)
            else: v_gap = 0
            
            # If gap is small in both dimensions, it's a match
            if max(h_gap, v_gap) < GAP_THRESHOLD:
                is_close = True
                break
        
        if is_close:
            cv2.drawContours(component_mask, [component], -1, 255, -1)
            accepted_rects.append(rect)

    filtered_mask = cv2.bitwise_and(component_mask, binary_mask.copy())

    roi_slice, offset = get_roi_bounding_box(filtered_mask)
    
    if roi_slice is None:
        return np.zeros_like(binary_mask), 0
    
    # Crop -> Skeletonize -> Prune -> Reconstruct
    cropped_mask = filtered_mask[roi_slice]
    skeleton_crop = np.array(skeletonize(cropped_mask // 255), dtype='uint8')
    
    # Apply your cleanup pipeline
    skeleton_crop = trim(prune(skeleton_crop, 5))
    skeleton_crop = trim(prune(skeleton_crop, 3))
    skeleton_crop = trim(prune(skeleton_crop, 3))
    
    # Place back into full frame
    full_skeleton = np.zeros_like(binary_mask)
    full_skeleton[roi_slice] = skeleton_crop
    
    return full_skeleton, np.sum(skeleton_crop)

def extract_skeleton(binary_mask):
    roi_slice, offset = get_roi_bounding_box(binary_mask)
    
    if roi_slice is None:
        # Empty mask
        return np.zeros_like(binary_mask), np.array([]), np.array([]), False

    # Perform operations on the smaller crop
    cropped_mask = binary_mask[roi_slice]
    
    # Convert binary mask to 1-pixel-wide skeleton
    skeleton_crop = np.array(skeletonize(cropped_mask // 255), dtype='uint8')

    # First cleanup pass
    skeleton_crop = prune(skeleton_crop, 5)  
    skeleton_crop = trim(skeleton_crop)
    
    # Second cleanup pass
    skeleton_crop = prune(skeleton_crop, 3)  
    skeleton_crop = trim(skeleton_crop)

    # Third cleanup pass
    skeleton_crop = prune(skeleton_crop, 3)  
    skeleton_crop = trim(skeleton_crop)
    
    # Identify branch points and end points on the crop
    branch_points, end_points = skeleton_nodes(skeleton_crop)
    
    # Adjust coordinates back to full image space
    if len(branch_points) > 0:
        branch_points[:, 0] += offset[0] # x
        branch_points[:, 1] += offset[1] # y
        
    if len(end_points) > 0:
        end_points[:, 0] += offset[0] # x
        end_points[:, 1] += offset[1] # y

    # Reconstruct full size skeleton image
    full_skeleton = np.zeros_like(binary_mask)
    full_skeleton[roi_slice] = skeleton_crop
    
    # Valid skeleton must have at least 2 endpoints (seed + at least one tip)
    is_valid = len(end_points) >= 2
    
    return full_skeleton, branch_points, end_points, is_valid

def trim(ske): ## Removes unwanted pixels from the skeleton
    # Using pre-allocated global kernels
    bp = np.zeros_like(ske)
    for t in T_KERNELS:
        bp = cv2.morphologyEx(ske, cv2.MORPH_HITMISS, t)
        ske = cv2.subtract(ske, bp)
    return ske


def prune(skel, num_it): ## Removes branches with length lower than num_it
    orig = skel
    
    # 1. Pruning loop using global kernels
    for i in range(0, num_it):
        # We sequentially subtract hit-miss results
        # To avoid multiple allocations, we can chain the operations or keep it simple
        # Keeping logic identical but using pre-allocated kernels:
        current_skel = skel
        for kernel in EP_KERNELS:
            hit = cv2.morphologyEx(current_skel, cv2.MORPH_HITMISS, kernel)
            current_skel = cv2.subtract(current_skel, hit)
        skel = current_skel
        
    # 2. Re-grow endpoints
    end = endPoints(skel)
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    for i in range(0, num_it):
        end = cv2.dilate(end, kernel)
        end = cv2.bitwise_and(end, orig)
        
    return cv2.bitwise_or(end, skel)


def endPoints(skel):
    ep = np.zeros_like(skel)
    # Use global EP_KERNELS
    for kernel in EP_KERNELS:
        ep = cv2.add(ep, cv2.morphologyEx(skel, cv2.MORPH_HITMISS, kernel))
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
    bp = np.zeros(skel.shape, dtype=int)
    
    for x in X_KERNELS:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, x)
    for y in Y_KERNELS:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, y)
    for t in T_BRANCH_KERNELS:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, t)
        
    return bp