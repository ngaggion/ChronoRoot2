import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import cv2
import networkx as nx
import json
import scipy.stats as stats
import logging
from typing import Tuple, List

# Import shared utilities
from analysis.utils import report_utils as utils

# Suppress minor logging and warnings for cleaner output
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)
plt.switch_backend('agg')

# Constants
PADDING_X = 200 # Pixel padding used for alignment
PIXEL_SIZE_MM = 0.04 # Conversion factor

def calculate_atlas_geometry(experiment_paths: List[str]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Scans all experiments to determine the maximum biological bounding box relative to the seed.
    Returns:
        canvas_shape: (height, width)
        seed_center: (y, x) coordinates for the seed on the new canvas
    """
    max_left = 0
    max_right = 0
    max_depth = 0
    BUFFER = 50 

    print("Scanning experiments to determine minimal Atlas size...")

    for exp_path in experiment_paths:
        # Load all results folders
        result_paths = utils.load_paths(exp_path, '*/*/*/Results*')
        
        for r_path in result_paths:
            # 1. Load Metadata
            meta_path = os.path.join(r_path, 'metadata.json')
            if not os.path.exists(meta_path): continue
            
            with open(meta_path) as f: 
                metadata = json.load(f)
            
            # 2. Load Last Segmentation Image
            seg_folder = os.path.join(r_path, 'Images/Seg/')
            seg_files = utils.load_paths(seg_folder, "*.png")
            if not seg_files: continue
            
            # Read image
            img = cv2.imread(seg_files[-1], 0)
            
            # Apply standard padding for alignment
            img = np.pad(img, ((0,0), (PADDING_X, 0)))
            seed = np.array(metadata['seed'])
            seed[0] += PADDING_X 
            
            # 3. Load Graph to find tips for rotation
            graph_folder = os.path.join(r_path, 'Graphs/')
            # Try to find corresponding graph file
            graph_filename = os.path.basename(seg_files[-1]).replace('png', 'xml.gz')
            graph_path = os.path.join(graph_folder, graph_filename)
            
            try: 
                g = nx.read_graphml(graph_path)
            except: 
                continue

            # Find root endpoints (Start and Tip)
            end1 = end2 = None
            for n in g.nodes():
                if g.nodes[n]['type'] == 'FTip': 
                    end1 = np.array([g.nodes[n]["pos_x"], g.nodes[n]["pos_y"]])
                if g.nodes[n]['type'] == 'Ini': 
                    end2 = np.array([g.nodes[n]["pos_x"], g.nodes[n]["pos_y"]])
            
            if end1 is None or end2 is None: continue
            
            # Determine which is the bottom tip (highest Y value)
            root_tip = end1 if end1[1] > end2[1] else end2
            root_tip[0] += PADDING_X

            # Calculate Angle to vertical
            v_vertical = np.array([0, 1])
            v_root = utils.unit_vector(root_tip - seed)   
            angle_rad = utils.angle_between(v_vertical, v_root)
            
            # Determine direction of rotation
            angle_deg = -np.rad2deg(angle_rad) if v_root[0] > 0 else np.rad2deg(angle_rad)
            
            # Rotate Image to straighten root
            rot_center = (float(seed[0]), float(seed[1]))
            rot_mat = cv2.getRotationMatrix2D(rot_center, angle_deg, 1.0)
            
            result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            _, result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)
            
            # 4. Measure Size Relative to Seed
            points = cv2.findNonZero(result)
            if points is None: continue
            
            x, y, w, h = cv2.boundingRect(points)
            
            # Calculate extent relative to seed
            dist_left = seed[0] - x
            dist_right = (x + w) - seed[0]
            dist_down = (y + h) - seed[1]
            
            max_left = max(max_left, dist_left)
            max_right = max(max_right, dist_right)
            max_depth = max(max_depth, dist_down)

    # Calculate final geometry
    center_x = int(max_left + BUFFER)
    center_y = 100 # Fixed top margin
    
    canvas_width = int(center_x + max_right + BUFFER)
    canvas_height = int(center_y + max_depth + BUFFER)
    
    print(f"Optimal Geometry: Size[{canvas_height}, {canvas_width}], Center[{center_y}, {center_x}]")
    return (canvas_height, canvas_width), (center_y, center_x)


def generate_root_atlases(save_path, days, timestep, canvas_shape, center_coords, rotate_root=True):
    """
    Generates the accumulated heatmaps (atlases) and calculates convex hull metrics.
    """
    dest_seed_y, dest_seed_x = center_coords 
    
    # Initialize Accumulators
    atlas_hull_mask = np.zeros(canvas_shape, dtype='uint8')
    temp_mask = np.zeros(canvas_shape, dtype='uint8')
    atlas_contours_rgb = np.zeros([canvas_shape[0], canvas_shape[1], 3], dtype='uint8')
    atlas_root_density = np.zeros(canvas_shape, dtype='float64')

    result_paths = utils.load_paths(save_path, '*/*/*/Results*')
    imgs_per_day = int(24 * (60 / timestep))

    all_frames_list = []
    set_of_atlases = []

    for day in days:
        day = int(day)
        
        # Reset Atlases for the new day
        atlas_contours_rgb.fill(255) # White background
        atlas_hull_mask.fill(0)
        temp_mask.fill(0)
        atlas_root_density.fill(0)
        
        # Metrics storage for this day
        metrics = {
            'area_bbox': [], 'area_chull': [], 'aspect_ratio': [],
            'density_lat': [], 'density_tot': [],
            'density_lat_bbox': [], 'density_tot_bbox': [],
            'width': [], 'height': []
        }
        
        for r_path in result_paths:
            # --- Load Data ---
            meta_path = os.path.join(r_path, 'metadata.json')
            if not os.path.exists(meta_path): continue
            with open(meta_path) as f: metadata = json.load(f)

            csv_files = utils.load_paths(r_path, 'PostProcess_Hour.csv')
            if not csv_files: continue
            df_temporal = pd.read_csv(csv_files[0])

            seg_path = os.path.join(r_path, 'Images/Seg/')
            seg_files = utils.load_paths(seg_path, "*.png")
            
            # Select specific image index based on day
            img_idx = int(day * imgs_per_day)
            
            # Handle case where experiment ended early
            if img_idx >= len(seg_files):
                current_seg_file = seg_files[-1]
            else:
                current_seg_file = seg_files[img_idx]
            
            img = cv2.imread(current_seg_file, 0)
            
            # Load Graph
            graph_folder = os.path.join(r_path, 'Graphs/')
            graph_file = current_seg_file.replace(seg_path, graph_folder).replace('png', 'xml.gz')
            
            try: 
                g = nx.read_graphml(graph_file)
            except: 
                # Fallback to last available graph
                fallback_graphs = utils.load_paths(graph_folder, "*.xml.gz")
                if not fallback_graphs: continue
                g = nx.read_graphml(fallback_graphs[-1])

            # --- Pre-processing (Padding) ---
            img = np.pad(img, ((0,0), (PADDING_X, 0)))
            seed = metadata['seed']
            seed[0] += PADDING_X
            
            # Find endpoints
            end1 = end2 = None
            for n in g.nodes():
                if g.nodes[n]['type'] == 'FTip': end1 = np.array([g.nodes[n]["pos_x"], g.nodes[n]["pos_y"]])
                if g.nodes[n]['type'] == 'Ini': end2 = np.array([g.nodes[n]["pos_x"], g.nodes[n]["pos_y"]])
            
            if end1 is None or end2 is None: continue
            
            # Determine tip
            root_tip = end1 if end1[1] > end2[1] else end2
            root_tip[0] += PADDING_X

            # --- Rotation Logic ---
            angle_deg = 0.0
            if rotate_root:
                v_vertical = np.array([0, 1])
                v_root = utils.unit_vector(root_tip - seed)
                angle_rad = utils.angle_between(v_vertical, v_root)
                angle_deg = -np.rad2deg(angle_rad) if v_root[0] > 0 else np.rad2deg(angle_rad)

            # --- Spatial Transformation ---
            # 1. Rotation Matrix around original seed
            rot_center = (float(seed[0]), float(seed[1]))
            rot_mat = cv2.getRotationMatrix2D(rot_center, angle_deg, 1.0)

            # 2. Translate to align seed with destination center
            rot_mat[0, 2] += (dest_seed_x - seed[0])
            rot_mat[1, 2] += (dest_seed_y - seed[1])

            # 3. Warp to final canvas
            warped_img = cv2.warpAffine(img, rot_mat, (canvas_shape[1], canvas_shape[0]), flags=cv2.INTER_LINEAR)
            _, warped_img = cv2.threshold(warped_img, 10, 255, cv2.THRESH_BINARY)
            
            # --- Hull & Atlas Calculation ---
            contours, _ = cv2.findContours(warped_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            # Calculate Convex Hulls for all fragments (usually just one main root)
            hulls = [cv2.convexHull(c, False) for c in contours]
            
            temp_mask.fill(0)
            for j in range(len(contours)):
                cv2.drawContours(temp_mask, hulls, j, 1, -1)
                # Draw outline on the RGB map
                cv2.drawContours(atlas_contours_rgb, hulls, j, (255, 0, 0), 3)
            
            atlas_hull_mask += temp_mask
            atlas_root_density += warped_img / 255.0
            
            # --- Metrics Calculation ---
            # Find largest contour for analysis
            contour_sizes = [(cv2.contourArea(c), c) for c in contours]
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            
            # Check if root is actually connected to the seed area
            # We check the destination seed coordinates because the image is already warped
            dist_to_seed = cv2.pointPolygonTest(biggest_contour, (dest_seed_x, dest_seed_y), True)
            is_inside = cv2.pointPolygonTest(biggest_contour, (dest_seed_x, dest_seed_y), False) > 0
            
            if is_inside or dist_to_seed < 30:
                hull_big = cv2.convexHull(biggest_contour, False)
                _,_,w,h = cv2.boundingRect(hull_big)
                
                area_bbox = w * h * (PIXEL_SIZE_MM**2)
                area_chull = cv2.contourArea(hull_big) * (PIXEL_SIZE_MM**2)
                
                # Match temporal data index
                csv_idx = int(img_idx * timestep / 60)
                if csv_idx >= len(df_temporal['TotalLength (mm)']) or img_idx == -1:
                    csv_idx = len(df_temporal['TotalLength (mm)']) - 1
                
                try:
                    total_len = df_temporal['TotalLength (mm)'][csv_idx]
                    lateral_len = df_temporal['LateralRootsLength (mm)'][csv_idx]
                    
                    # Store Metrics
                    metrics['area_bbox'].append(area_bbox)
                    metrics['area_chull'].append(area_chull)
                    metrics['aspect_ratio'].append(h/w if w > 0 else 0)
                    metrics['density_lat'].append(lateral_len/area_chull if area_chull > 0 else 0)
                    metrics['density_tot'].append(total_len/area_chull if area_chull > 0 else 0)
                    metrics['density_lat_bbox'].append(lateral_len/area_bbox if area_bbox > 0 else 0)
                    metrics['density_tot_bbox'].append(total_len/area_bbox if area_bbox > 0 else 0)
                    metrics['width'].append(w * PIXEL_SIZE_MM)
                    metrics['height'].append(h * PIXEL_SIZE_MM)
                except Exception as e:
                    pass

        # Construct DataFrame for the day
        day_df = pd.DataFrame({
            'Convex Hull Aspect Ratio': metrics['aspect_ratio'],
            'Lateral Root Area Density': metrics['density_lat'],
            'Lateral Root Area Density BBOX': metrics['density_lat_bbox'],
            'Total Root Area Density': metrics['density_tot'],
            'Total Root Area Density BBOX': metrics['density_tot_bbox'],
            'Convex Hull Area': metrics['area_chull'],
            'Bounding Box Area': metrics['area_bbox'],
            'Convex Hull Width': metrics['width'],
            'Convex Hull Height': metrics['height']
        })
        day_df['Day'] = day
        
        all_frames_list.append(day_df)
        set_of_atlases.append([atlas_hull_mask.copy(), atlas_contours_rgb.copy(), atlas_root_density.copy()])
    
    final_df = pd.concat(all_frames_list, ignore_index=True) if all_frames_list else pd.DataFrame()

    return set_of_atlases, final_df


def visualize_single_atlas(atlas_hull, atlas_contours, atlas_roots, save_path, name, day=None):
    """Plots the three-panel atlas visualization."""
    plt.ioff()
    plt.figure(figsize=(9, 4))

    plt.subplot(1,3,1)
    plt.imshow(atlas_roots, cmap='jet', vmin=0, vmax=25)
    plt.title("Accumulated Roots")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(atlas_contours, cmap='jet', vmin=0, vmax=25)
    plt.title("Convex Hull Contours")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(atlas_hull, alpha=0.6)
    plt.title("Accumulated Convex Hulls")
    plt.axis('off')

    title_suffix = f" - Day: {day}" if day is not None else " - Last Day"
    full_title = f"{name}{title_suffix}"
    plt.suptitle(full_title)
    
    save_dir = os.path.join(save_path, "Per Experiment")
    utils.ensure_directory(save_dir)
    
    plt.savefig(os.path.join(save_dir, full_title), dpi=300, bbox_inches='tight')
    plt.close('all')


def visualize_combined_atlases(folder):
    """
    Stacks atlas images vertically for qualitative comparison across days.
    """
    images_per_day = {}
    source_dir = os.path.join(folder, 'Per Experiment')

    if not os.path.exists(source_dir):
        return

    # Group images by day
    for filename in os.listdir(source_dir):
        if not filename.endswith(('png', 'jpg', 'jpeg')): continue
        
        name = os.path.splitext(filename)[0]
        
        # Parse day
        if "Last Day" in name:
            day = "Last"
        elif "Day:" in name:
            try: day = int(name.split('Day:')[1].strip())
            except: continue
        else:
            try: day = int(name.split('-')[-1].strip())
            except: continue

        if day not in images_per_day:
            images_per_day[day] = []
        images_per_day[day].append(filename)
    
    # Sort keys: Numbers first, then "Last"
    sorted_days = sorted([d for d in images_per_day.keys() if d != "Last"]) 
    if "Last" in images_per_day:
        sorted_days.append("Last")
    
    for day in sorted_days:
        images = images_per_day[day]
        images.sort(key=utils.natural_key)

        # Create subplots stack
        fig, axs = plt.subplots(len(images), 1, figsize=(9, 4 * len(images)))
        
        # Ensure axs is iterable even if 1 image
        if len(images) == 1:
            axs = [axs]

        for i, image_name in enumerate(images):
            image_path = os.path.join(source_dir, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i].imshow(img)
            axs[i].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        
        save_name = f'Qualitative - Day {day}'
        plt.savefig(os.path.join(folder, save_name), dpi=300, bbox_inches='tight')
        plt.close('all')


def _plot_single_metric(save_path, data, x_col, y_col, hue, title, ylabel, filter_zeros=False):
    """
    Internal helper to generate violin plots for a specific metric.
    """
    plt.figure()
    fig, ax = plt.subplots()
    
    plot_data = data.copy()
    if filter_zeros:
        plot_data = plot_data[plot_data[y_col] > 0].reset_index(drop=True)
    
    hue_order = plot_data[hue].unique()
    n_types = len(hue_order)

    sns.violinplot(x=x_col, y=y_col, data=plot_data, hue=hue, inner=None, 
                   zorder=2, legend=False, hue_order=hue_order)

    # Manual legend handling
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[0:n_types], labels[0:n_types], loc=2)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    
    # Save as both PNG and SVG
    clean_title = title.replace('/', ' per ')
    plt.savefig(os.path.join(save_path, f"{clean_title}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f"{clean_title}.svg"), dpi=300, bbox_inches='tight')
    plt.close('all')


def plot_hull_metrics_summary(save_path, frame):
    """
    Generates all statistical plots (violins) and the summary CSV table.
    """
    plt.ioff()
    utils.ensure_directory(save_path)

    # 1. Generate Plots
    metrics_config = [
        # (Y Column, Title, Y Label, Filter Zeros?)
        ('Convex Hull Area', 'Convex Hull Area', 'Area (mm²)', False),
        ('Lateral Root Area Density', 'Lateral Roots Area Density', 'LR / convex hull area (mm/mm²)', True),
        ('Convex Hull Aspect Ratio', 'Aspect Ratio', 'Aspect Ratio (height/width)', False),
        ('Total Root Area Density', 'Total Root Area Density', 'TR / convex hull area (mm/mm²)', True),
        ('Convex Hull Width', 'Convex Hull Width', 'Width (mm)', False),
        ('Convex Hull Height', 'Convex Hull Height', 'Height (mm)', False)
    ]

    for y_col, title, y_label, filter_zeros in metrics_config:
        _plot_single_metric(save_path, frame, 'Day', y_col, 'Experiment', title, y_label, filter_zeros)

    # 2. Generate Summary Table
    # Group by Day/Experiment, calculate Mean/Std
    agg_dict = {
        'Convex Hull Area': ['count', 'mean', 'std'],
        'Lateral Root Area Density': ['mean', 'std'],
        'Convex Hull Aspect Ratio': ['mean', 'std'],
        'Total Root Area Density': ['mean', 'std'],
        'Convex Hull Width': ['mean', 'std'],
        'Convex Hull Height': ['mean', 'std']
    }
    
    summary_data = frame.groupby(['Day', 'Experiment']).agg(agg_dict)
    
    # Flatten MultiIndex columns (e.g., ('Convex Hull Area', 'mean') -> 'Convex Hull Area mean')
    summary_data.columns = [' '.join(col).strip() for col in summary_data.columns.values]
    summary_data = summary_data.reset_index()
    summary_data = summary_data.round(3)
    
    summary_data.to_csv(os.path.join(save_path, "Summary Table.csv"), index=False)


def analyze_hull_statistics(conf, data, metric):
    """
    Performs Mann-Whitney U tests between experiments for a specific metric on specific days.
    """
    data['Experiment'] = data['Experiment'].astype(str)
    data['Day'] = data['Day'].astype(str)
    
    unique_experiments = data['Experiment'].unique()
    n_exp = len(unique_experiments)
    
    # Days to analyze are defined in the config string (e.g. "1,3,5")
    days_to_analyze = conf['daysConvexHull'].split(',')
    
    # Setup report file
    report_folder = os.path.join(conf['MainFolder'], 'Report', 'Convex Hull and Area Analysis')
    utils.ensure_directory(report_folder)
    
    file_metric_name = metric.replace('/', ' over ')
    report_file = os.path.join(report_folder, f'{file_metric_name} Stats.txt')
        
    with open(report_file, 'w') as f:
        f.write('Using Mann Whitney U test to compare different experiments\n')
        
        if metric == 'Lateral Root Area Density':
            f.write('For Lateral Root Area Density, plants without lateral roots are excluded.\n\n')
         
        for day in days_to_analyze:                       
            f.write(f'Day: {day}\n')
            subdata = data[data['Day'] == day]
            
            # Filter zeros if necessary
            if metric == 'Lateral Root Area Density':
                subdata = subdata[subdata['Lateral Root Area Density'] > 0].reset_index(drop=True)
            
            # Pairwise comparison
            for i in range(n_exp - 1):
                for j in range(i + 1, n_exp):
                    exp_name_1 = unique_experiments[i]
                    exp_name_2 = unique_experiments[j]
                    
                    vals1 = subdata[subdata['Experiment'] == exp_name_1][metric]
                    vals2 = subdata[subdata['Experiment'] == exp_name_2][metric]
                    
                    try:
                        if len(vals1) == 0 or len(vals2) == 0:
                            raise ValueError("Empty data for one experiment")
                            
                        U, p = stats.mannwhitneyu(vals1, vals2)
                        p_val = round(p, 6)
                        
                        f.write(f'Mean {exp_name_1}: {round(vals1.mean(), 2)}\n')
                        f.write(f'Mean {exp_name_2}: {round(vals2.mean(), 2)}\n')
                        
                        sig_text = "are significantly different" if p_val < 0.05 else "are not significantly different"
                        f.write(f'Experiments {exp_name_1} and {exp_name_2} {sig_text}. P-value: {p_val}\n')
                        
                    except Exception:
                        f.write(f'Experiments {exp_name_1} and {exp_name_2} could not be compared (Insufficient data)\n')
                        
            f.write('\n')