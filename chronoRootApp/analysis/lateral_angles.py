"""
Lateral Root Angle Analysis Module

This module provides functions for analyzing lateral root angles from RSML 
(Root System Markup Language) files. It includes functionality for:
- Parsing RSML files and extracting root geometry
- Calculating tip angles and emergence angles
- Tracking lateral roots across time series
- Statistical analysis and visualization of angle data

Dependencies:
    - xml.etree.ElementTree: XML parsing for RSML files
    - matplotlib/seaborn: Visualization
    - numpy/pandas: Data manipulation
    - scipy.stats: Statistical tests
    - cv2: Image processing
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import csv
import pandas as pd
from .report import load_path as loadPath
import scipy.stats as stats
import cv2
import logging

# Suppress matplotlib category warnings and configure backend for non-interactive use
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)
plt.switch_backend('agg')
pd.options.mode.chained_assignment = None


def traverse_root(root_element, is_lateral_root=False):
    """
    Extract coordinate points from a root element in the RSML structure.
    
    Traverses the XML element representing a root and extracts all x,y coordinates
    of the polyline defining the root shape. Points are ordered from base to tip
    (ascending y-coordinate).
    
    Args:
        root_element: XML element containing the root geometry data
        is_lateral_root: If True, access the first child element first 
                        (lateral roots have an extra nesting level)
    
    Returns:
        list: List of [x, y] coordinate pairs ordered from base to tip
    """
    if is_lateral_root:
        root_element = root_element[0]

    # Extract all coordinate points from the polyline
    points = [[int(point.attrib['x']), int(point.attrib['y'])] 
              for point in root_element[0]]
  
    start_point = points[0]
    end_point = points[-1]

    # Ensure points are ordered from base (lower y) to tip (higher y)
    # If start has higher y than end, reverse the list
    if start_point[1] > end_point[1]:
        return points[::-1]
    else:
        return points


def tipAngle(points):
    """
    Calculate the tip angle of a root from base to tip.
    
    The tip angle is measured as the angle between the vertical axis and
    the line connecting the first point (base) to the last point (tip).
    An angle of 0 degrees means the root grows straight down (vertical).
    
    Args:
        points: List of [x, y] coordinate pairs defining the root polyline
    
    Returns:
        float: Tip angle in degrees (0 = vertical, 90 = horizontal)
    """
    start = points[0]
    end = points[-1]
    
    # Calculate hypotenuse (Euclidean distance between start and end)
    hypotenuse = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    
    # Calculate vertical distance (adjacent side for angle calculation)
    vertical_distance = end[1] - start[1]

    # Calculate angle using arccos (angle from vertical)
    angle_rad = np.arccos(vertical_distance / hypotenuse)
    angle_deg = angle_rad * 180 / np.pi

    return angle_deg


def emergenceAngle(points, distance_cm=1, pixel_size=0.04):
    """
    Calculate the emergence angle of a root over a specified initial distance.
    
    The emergence angle captures the initial growth direction of the root,
    measured from the base to a point at a specified physical distance.
    This is more representative of the root's initial trajectory than the tip angle.
    
    Args:
        points: List of [x, y] coordinate pairs defining the root polyline
        distance_cm: Physical distance in cm to measure the angle over
        pixel_size: Size of each pixel in cm (for converting pixels to physical units)
    
    Returns:
        float: Emergence angle in degrees (0 = vertical, 90 = horizontal)
    """
    # Convert physical distance to pixel distance
    distance_pixels = int(distance_cm / pixel_size)
    num_points = len(points)
    
    start = points[0]
    # Use the point at the specified distance, or the last point if root is shorter
    end = points[min(distance_pixels, num_points) - 1]

    # Calculate hypotenuse (Euclidean distance)
    hypotenuse = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    
    # Calculate vertical distance
    vertical_distance = end[1] - start[1]

    # Calculate angle using arccos
    angle_rad = np.arccos(vertical_distance / hypotenuse)
    angle_deg = angle_rad * 180 / np.pi

    return angle_deg


def plotAngles(ax, points, angle, tip=True, distance_cm=1, pixel_size=0.04):
    """
    Plot angle visualization lines on a matplotlib axes.
    
    Draws the reference lines used for angle measurement: the line between
    start and end points, and a vertical reference line. Also annotates
    the calculated angle value.
    
    Args:
        ax: Matplotlib axes object to draw on
        points: List of [x, y] coordinate pairs defining the root
        angle: Pre-calculated angle value to display
        tip: If True, measure to the tip; if False, measure to emergence distance
        distance_cm: Physical distance for emergence angle measurement
        pixel_size: Size of each pixel in cm
    
    Returns:
        None (modifies ax in place)
    """
    if tip:
        start = points[0]
        end = points[-1]
    else:
        distance_pixels = int(distance_cm / pixel_size)
        num_points = len(points)
        start = points[0]
        end = points[min(distance_pixels, num_points) - 1]

    # Draw line from start to end point
    xs = start[0], end[0]
    ys = start[1], end[1]
    ax.plot(xs, ys, linewidth=0.5)

    # Draw vertical reference line
    ax.plot([start[0], start[0]], ys, linewidth=0.5)

    # Annotate with angle value
    ax.text(x=start[0], y=start[1], s=str(round(angle, 2)), fontsize='xx-small')

    return


def find_nearest(point, point_list):
    """
    Find the nearest point in a list to a given reference point.
    
    Uses Euclidean distance to find the closest match. Useful for
    tracking roots across frames by matching starting points.
    
    Args:
        point: Reference point as numpy array [x, y]
        point_list: Array of points to search, shape (N, 2)
    
    Returns:
        tuple: (index of nearest point, distance to nearest point)
    """
    distances = np.linalg.norm(point - point_list, axis=1)
    nearest_index = np.argmin(distances)

    return nearest_index, distances[nearest_index]


def matching(new_starts, all_new_points, old_starts, all_old_points, old_names, num_roots=0):
    """
    Match lateral roots between consecutive frames for tracking.
    
    Implements a simple tracking algorithm that matches roots based on the
    proximity of their starting points. Roots within 20 pixels of a previous
    root's start point are considered the same root. New roots are assigned
    new names, and disappeared roots are preserved in the list.
    
    Args:
        new_starts: List of starting points for roots in current frame
        all_new_points: List of full polyline points for each root in current frame
        old_starts: List of starting points from previous frame
        all_old_points: List of full polyline points from previous frame
        old_names: List of root names from previous frame
        num_roots: Counter for total number of unique roots seen
    
    Returns:
        tuple: (matched_starts, matched_points, matched_names, updated_num_roots)
    """
    matched_names = []
    matched_starts = []
    matched_points = []
    seen_indices = []

    # If no previous data, assign new names to all roots
    if old_starts == [] or old_names == []:
        for j in range(len(new_starts)):
            matched_names.append('LR%s' % num_roots)
            matched_starts.append(new_starts[j])
            matched_points.append(all_new_points[j])
            num_roots += 1
    else:
        old_points_array = np.array(old_starts)
        new_points_array = np.array(new_starts)

        # Match each new root to the nearest old root
        for j in range(new_points_array.shape[0]):
            nearest_idx, distance = find_nearest(new_points_array[j, :], old_points_array)

            # Distance threshold of 20 pixels for matching
            if distance < 20:
                # Match found - use existing name
                matched_names.append(old_names[nearest_idx])
                matched_starts.append(new_starts[j])
                matched_points.append(all_new_points[j])
                seen_indices.append(nearest_idx)
            else:
                # No match - assign new name
                matched_names.append('LR%s' % num_roots)
                matched_starts.append(new_starts[j])
                matched_points.append(all_new_points[j])
                num_roots += 1
        
        # Preserve old roots that weren't matched (may have temporarily disappeared)
        num_old = old_points_array.shape[0]
        seen_indices.sort()

        for j in range(num_old):
            if j not in seen_indices:
                matched_names.append(old_names[j])
                matched_starts.append(old_starts[j])
                matched_points.append(all_old_points[j])

        # Sort all results by name for consistent ordering
        sort_indices = np.argsort(np.array(matched_names))
        matched_names = np.array(matched_names, dtype=object)[sort_indices].tolist()
        matched_starts = np.array(matched_starts, dtype=object)[sort_indices].tolist()
        matched_points = np.array(matched_points, dtype=object)[sort_indices].tolist()

    return matched_starts, matched_points, matched_names, num_roots


def lenRoot(points, pixel_size=0.04):
    """
    Calculate the total length of a root polyline in physical units.
    
    Sums the Euclidean distances between consecutive points and converts
    from pixels to centimeters using the pixel size.
    
    Args:
        points: List of [x, y] coordinate pairs defining the root
        pixel_size: Size of each pixel in cm
    
    Returns:
        float: Total root length in cm
    """
    # Convert to numpy array for vectorized operations
    points_array = np.array(points)
    
    # Calculate differences between consecutive points
    diffs = np.diff(points_array, axis=0)
    
    # Calculate segment lengths and sum
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segment_lengths) * pixel_size
    
    return total_length


def getAngles(conf, path):
    """
    Extract and save angle measurements for all lateral roots in a time series.
    
    Processes all RSML files in the specified path, tracks lateral roots across
    frames, and calculates tip and emergence angles. Results are saved to a CSV file.
    
    Args:
        conf: Configuration dictionary containing 'emergenceDistance' parameter
        path: Path to the analysis results folder containing RSML subfolder
    
    Returns:
        None (writes results to LateralRootsData.csv)
    """
    # Load list of images that have been postprocessed
    images = pd.read_csv(os.path.join(path, "FilesAfterPostprocessing.csv"))
    images.dropna(inplace=True)
    images = images['FileName'].tolist()

    # Get list of available RSML files
    rsml_files = os.listdir(os.path.join(path, "RSML"))
    
    # Filter to only images with corresponding RSML files
    paths = [image for image in images 
             if image.split('/')[-1].replace('.png', '.rsml') in rsml_files]
    paths = [os.path.join(path, 'RSML', image.split('/')[-1].replace('.png', '.rsml')) 
             for image in paths]

    # Initialize tracking variables
    lateral_roots = []
    lateral_root_starts = []
    lateral_root_names = []
    num_roots = 0
    
    filepath = os.path.join(path, 'LateralRootsData.csv')
    
    with open(filepath, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(["Img", "Number of lateral roots", "Mean tip angle", 
                        "Mean emergence angle", "First LR tip", "First LR emergence"])

        for step in paths:
            # Parse RSML file
            tree = ET.parse(step).getroot()

            # Load metadata for pixel size
            with open(os.path.join(path, 'metadata.json')) as f:
                metadata = json.load(f)

            y1, y2, x1, x2 = metadata['bounding box']
            h = y2 - y1
            w = x2 - x1

            plant = tree[1][0][0]
            
            # Check if lateral roots exist (more than just primary root)
            if len(plant) > 1:
                lateral_root_elements = plant[1:]
                
                points_list = []
                start_points = []

                # First: perform tracking of lateral roots
                for lr_element in lateral_root_elements:
                    points = traverse_root(lr_element, is_lateral_root=True)
                    points_list.append(points)
                    start_points.append(points[0])

                lateral_root_starts, lateral_roots, lateral_root_names, num_roots = matching(
                    start_points, points_list, 
                    lateral_root_starts, lateral_roots, lateral_root_names, 
                    num_roots
                )
                
                # Second: estimate angles for all tracked roots
                tip_angles = []
                emergence_angles = []

                for root in lateral_roots:
                    tip_angle = tipAngle(root)
                    emergence_angle = emergenceAngle(
                        root, 
                        float(conf['emergenceDistance']), 
                        metadata['pixel_size']
                    )
                    tip_angles.append(tip_angle)
                    emergence_angles.append(emergence_angle)

            # Extract image name for output
            imgname = step.replace(os.path.join(path, 'RSML') + '/', '').replace('.rsml', '.png')
            
            # Write measurements to CSV
            if num_roots == 0:
                measures = [imgname, 0, 0, 0, 0, 0]
            else:
                measures = [
                    imgname, 
                    num_roots, 
                    round(np.mean(tip_angles), 3), 
                    round(np.mean(emergence_angles), 3), 
                    round(tip_angles[0], 3), 
                    round(emergence_angles[0], 3)
                ]
                    
            writer.writerow(measures)
    
    return


def dataWork(df, first_day, last_day):
    """
    Process and normalize time series data for angle measurements.
    
    Extracts datetime from image filenames, filters to the specified date range,
    fills gaps in the time series, and aggregates by day and hour.
    
    Args:
        df: DataFrame with angle measurements and 'Img' column containing timestamps
        first_day: Start of the analysis period (datetime)
        last_day: End of the analysis period (datetime)
    
    Returns:
        DataFrame: Processed data with 'Day', 'Hour', and angle measurements,
                   or empty DataFrame if no valid data
    """
    # Extract datetime components from image filename
    datetime_strings = df['Img'].str.extract(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})')
    datetime_strings.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute']
    datetime_strings = datetime_strings.astype(int)

    df = df.drop(['Img'], axis=1)
    df['Date'] = pd.to_datetime(datetime_strings[['Year', 'Month', 'Day', 'Hour', 'Minute']])

    # Remove timezone info if present for consistent comparison
    if first_day.tzinfo is not None:
        first_day = first_day.tz_localize(None)
    if last_day.tzinfo is not None:
        last_day = last_day.tz_localize(None)

    beginning = df['Date'].min()

    # Handle data that starts before first_day
    if beginning < first_day:
        df = df[df['Date'] >= first_day]
        if df.empty:
            return pd.DataFrame()
    # Handle data that starts after first_day (fill gaps with zeros)
    elif beginning > first_day:
        rows = pd.DataFrame()
        rows['Date'] = pd.date_range(start=first_day, end=beginning, freq='15T')
        for col in ['Number of lateral roots', 'Mean tip angle', 'Mean emergence angle', 
                    'First LR tip', 'First LR emergence']:
            rows[col] = 0
        df = pd.concat([rows, df], ignore_index=True)

    end = df['Date'].max()

    # Handle data that extends beyond last_day
    if end > last_day:
        df = df[df['Date'] <= last_day]
    # Handle data that ends before last_day (extend with last values)
    elif end < last_day:
        rows = pd.DataFrame()
        rows['Date'] = pd.date_range(start=end, end=last_day, freq='15T')
        for col in ['Number of lateral roots', 'Mean tip angle', 'Mean emergence angle', 
                    'First LR tip', 'First LR emergence']:
            rows[col] = df[col].iloc[-1]
        df = pd.concat([df, rows], ignore_index=True)

    if df.empty:
        return pd.DataFrame()

    # Calculate days and hours elapsed from first_day
    time_elapsed = df['Date'] - first_day
    df['Day'] = time_elapsed.dt.days
    df['Hour'] = df['Date'].dt.hour
    
    df = df.drop(['Date'], axis=1)
    
    # Aggregate by day and hour
    df = df.groupby(['Day', 'Hour']).mean(numeric_only=True).reset_index()
    df = df.astype(float)
    
    return df


def avoidIncreasingValues(data, metric, tolerance=0.3):
    """
    Apply smoothing to prevent unrealistic jumps in time series data.
    
    First applies a median filter to remove spikes, then enforces that values
    cannot increase by more than the specified tolerance (30% by default).
    This corrects for tracking errors that might cause angle values to jump.
    
    Args:
        data: DataFrame containing the time series
        metric: Column name to process
        tolerance: Maximum allowed relative increase between consecutive values
    
    Returns:
        DataFrame: Data with smoothed metric column
    """
    # Apply median filter to remove spikes
    data[metric] = data[metric].rolling(window=5, min_periods=1).median()
    
    # Enforce maximum increase constraint (skip initial transient period)
    series = data[metric]
    for j in range(12, len(series)):
        if series.iloc[j] < series.iloc[j-1] * (1 + tolerance):
            continue
        else:
            series.iloc[j] = series.iloc[j-1]
    data[metric] = series
    
    return data


def getFirstLateralRoots(conf, df):
    """
    Extract and synchronize first lateral root data across all plants.
    
    For each plant, finds when the first lateral root appeared and tracks
    its tip angle over a 72-hour window. Data is synchronized so that
    time=0 corresponds to first LR emergence for each plant.
    
    Args:
        conf: Configuration dictionary with 'MainFolder' path
        df: DataFrame with angle measurements for all plants
    
    Returns:
        DataFrame: Synchronized first LR data with columns 
                   ['Time', 'First LR tip', 'Experiment', 'Plant_id', 'Real']
    """
    report_path = os.path.join(conf['MainFolder'], 'Report')
    
    plants = df['Plant_id'].unique()
    
    # Mark original data points
    df['Real'] = 1

    # Maximum tracking window: 72 hours
    max_hours = 72
    synchronized_data = pd.DataFrame(columns=['Time', 'First LR tip', 'Experiment', 'Plant_id', 'Real'])

    for plant in plants:
        plant_data = df[df['Plant_id'] == str(plant)]
        
        # Filter to only rows where first LR exists (tip angle > 0)
        has_lr_idx = plant_data['First LR tip'] > 0
        plant_data = plant_data.loc[has_lr_idx]
        num_rows = plant_data.shape[0]
        
        if num_rows > 0:
            # Apply smoothing to remove artifacts
            plant_data = avoidIncreasingValues(plant_data, 'First LR tip')
            plant_data = plant_data.iloc[:(max_hours + 1), :]

            # Extend data if shorter than 72 hours
            while num_rows <= max_hours:
                last_row = plant_data.iloc[-1, :]
                last_row = last_row.to_frame().T
                if last_row.shape[0] != 1 or last_row.shape[1] != 10:
                    raise ValueError('Extending 1st LR Error')
                last_row['Real'] = np.nan  # Mark extended rows as not real
                plant_data = pd.concat([plant_data, last_row], ignore_index=True)
                num_rows = num_rows + 1
                
            # Reset index to create time column
            plant_data = plant_data.reset_index()
            plant_data['Time'] = plant_data.index
            plant_data = plant_data.drop(['index'], axis=1)
            plant_data = plant_data.loc[:, ['Time', 'First LR tip', 'Experiment', 'Plant_id', 'Real']]
            
            synchronized_data = pd.concat([synchronized_data, plant_data], ignore_index=True)

    synchronized_data.to_csv(os.path.join(report_path, 'SyncronizedFirstLR.csv'), index=False)
    
    return synchronized_data


def makeLateralAnglesPlots(conf):
    """
    Generate all lateral root angle analysis plots and statistics.
    
    Main entry point for angle analysis. Loads data from all experiments,
    generates violin plots for emergence angles, tracks first lateral roots,
    and performs statistical comparisons between experiments.
    
    Args:
        conf: Configuration dictionary containing:
              - 'MainFolder': Root path for analysis
              - 'daysAngles': Comma-separated days to analyze
              - 'everyXhourFieldAngles': Interval for statistical analysis
              - 'averagePerPlantStats': Whether to average per plant before stats
    
    Returns:
        None (generates plots and CSV files in Report folder)
    """
    parent_folder = conf['MainFolder']
    analysis = os.path.join(conf['MainFolder'], "Analysis")
    experiments = loadPath(analysis, '*')
    
    report_path = os.path.join(parent_folder, 'Report')
    report_path_angle = os.path.join(report_path, 'Angles Analysis')
    os.makedirs(report_path_angle, exist_ok=True)
    
    all_data = pd.DataFrame()
        
    # Check for cached data (currently disabled with 'and False')
    if os.path.exists(os.path.join(report_path, 'LateralRootsData.csv')) and False:
        all_data = pd.read_csv(os.path.join(report_path, 'LateralRootsData.csv'))
        all_data['Experiment'] = all_data['Experiment'].astype('str')
        all_data['Plant_id'] = all_data['Plant_id'].astype('str')
    else:
        # Load data from all experiments and plants
        for exp in experiments:
            plants = loadPath(exp, '*/*/*')
            exp_name = exp.replace(analysis, '').replace('/', '')
            print('Experiment:', exp_name, '- Total plants', len(plants))

            for plant in plants:
                results = loadPath(plant, '*')
                if len(results) == 0:
                    continue
                else:
                    results = results[-1]
                plant_name = plant.replace(exp, '').replace('/', '_')

                file = os.path.join(results, 'LateralRootsData.csv')
                file2 = os.path.join(results, 'PostProcess_Original.csv')

                data2 = pd.read_csv(file2)
                data2.dropna(inplace=True)
                
                # Get date range from postprocessing data
                date1 = pd.to_datetime(data2.loc[0, "Date"], dayfirst=False)
                date2 = pd.to_datetime(data2.iloc[-1]["Date"], dayfirst=False)
                            
                data = pd.read_csv(file)
                data = dataWork(data, date1, date2)

                if data.empty:
                    continue

                data['Plant_id'] = plant_name
                data['Experiment'] = exp_name

                all_data = pd.concat([all_data, data], ignore_index=True)

        all_data.to_csv(os.path.join(report_path, 'LateralRootsData.csv'), index=False)

    # Filter data for specified analysis days
    frame = []
    if not all_data.empty:
        for day in conf['daysAngles'].split(','):
            aux = all_data[all_data['Day'] == int(day)]
            aux = aux[aux['Hour'] == 0]
            aux = aux[aux['Mean emergence angle'] > 0]
            frame.append(aux)
    
    # Generate emergence angle plots
    if len(frame) > 0:
        frame = pd.concat(frame)
        if not frame.empty:
            n_exp = len(frame['Experiment'].unique())

            plt.figure(figsize=(8, 9), dpi=200)
            sns.color_palette("tab10")
                            
            ax = plt.subplots()

            # Create violin plot with overlaid swarm plot
            sns.violinplot(x='Day', y='Mean emergence angle', data=frame, 
                          hue='Experiment', inner=None, zorder=2, legend=False)
            ax = sns.swarmplot(x='Day', y='Mean emergence angle', data=frame, 
                              hue='Experiment', dodge=True, size=4, 
                              palette='muted', edgecolor='black', 
                              linewidth=0.5, zorder=1, s=2)
            ax.set_ylim(-20, 120)
            
            # Fix legend to only show experiments once
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[0:n_exp], labels[0:n_exp], loc=4)

            ax.set_title('Mean emergence angle')

            plt.savefig(os.path.join(report_path_angle, 'Mean Emergence Angle.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(report_path_angle, 'Mean Emergence Angle.svg'), 
                       dpi=300, bbox_inches='tight')
            
            # Perform statistical analysis
            performStatisticalAnalysisAngles(conf, frame, 'Mean emergence angle')

            # Generate summary table
            summary_data = frame.groupby(['Day', 'Experiment']).agg(
                {'Mean emergence angle': ['count', 'mean', 'std']}
            )
            summary_data.columns = [' '.join(col).strip() for col in summary_data.columns.values]
            summary_data = summary_data.reset_index()
            summary_data.columns = ['Day', 'Experiment', 'N Samples', 
                                   'Mean Emergence Angle (Mean)', 'Mean Emergence Angle (std)']
            summary_data = summary_data.round(3)
            summary_data['Day'] = summary_data['Day'].astype('int')
            summary_data = summary_data.sort_values(by='Day', ascending=True)
            summary_data.to_csv(os.path.join(report_path_angle, "Mean Emergence Angle Table.csv"), 
                               index=False)
    
    # Generate first lateral root analysis
    synchronized_data = getFirstLateralRoots(conf, all_data)
    
    if synchronized_data.empty:
        print("Warning: No data available for First Lateral Root plots. Skipping.")
    else:
        # Plot number of plants with first LR over time
        plt.figure(figsize=(6, 4), dpi=200)
        sns.lineplot(y="Real", x='Time', hue="Experiment", data=synchronized_data, 
                    errorbar=None, estimator=np.count_nonzero)
        plt.title('Number of plant with first LR roots per experiment')
        plt.ylabel('Number of first LR')
        plt.xlabel('Time elapsed since emergence (h)')
        
        plt.savefig(os.path.join(report_path_angle, 'First LR Number.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(report_path_angle, 'First LR Number.svg'), 
                   dpi=300, bbox_inches='tight')
        
        # Remove extended (non-real) data points for angle analysis
        synchronized_data.dropna(inplace=True)

        if not synchronized_data.empty:
            # Plot tip angle decay over time
            plt.figure(figsize=(6, 4), dpi=200)
            sns.lineplot(y="First LR tip", x='Time', hue="Experiment", 
                        data=synchronized_data, errorbar='se')
            plt.title('Decay of the tip angle (1st LR)')
            plt.xlabel('Time (h)')
            plt.ylabel('Angle')

            plt.savefig(os.path.join(report_path_angle, 'First LR Tip Angle Decay.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(report_path_angle, 'First LR Tip Angle Decay.svg'), 
                       dpi=300, bbox_inches='tight')

            performStatisticalAnalysisFirstLR(conf, synchronized_data.copy(), 'First LR tip')

            # Generate summary table for first LR tip angles
            summary_df_list = []
            
            data = synchronized_data.copy()
            data['Experiment'] = data['Experiment'].astype(str)
            report_path = os.path.join(conf['MainFolder'], 'Report', 'Angles Analysis')
            dt = int(conf['everyXhourFieldAngles'])

            for hour in range(0, 73 - dt, dt):
                end = min(hour + dt, 72)
                hours = np.arange(hour, end)
                subdata = data[data['Time'].isin(hours)]

                if not subdata.empty:
                    subdata["First LR tip"] = subdata["First LR tip"].astype(float)
                    subdata = subdata.groupby(['Experiment', 'Plant_id'])["First LR tip"].mean(
                        numeric_only=True
                    ).reset_index()
                    subdata = subdata.groupby(['Experiment']).agg(
                        {'First LR tip': ['count', 'mean', 'std']}
                    )

                    subdata.columns = [' '.join(col).strip() for col in subdata.columns.values]
                    subdata = subdata.reset_index()
                    subdata['Interval'] = str(hour) + '-' + str(end)
                    
                    subdata.columns = ['Experiment', 'N Samples', 'First LR tip (Mean)', 
                                      'First LR tip (std)', 'Hours interval']
                    subdata = subdata.round(3)
                    summary_df_list.append(subdata)
            
            if len(summary_df_list) > 0:
                summary_df = pd.concat(summary_df_list)
                col = summary_df.pop("Hours interval")
                summary_df.insert(0, col.name, col)
                summary_df.to_csv(os.path.join(report_path_angle, "First LR Tip Angle Table.csv"), 
                                 index=False)
    return


def performStatisticalAnalysisAngles(conf, data, metric):
    """
    Perform pairwise Mann-Whitney U tests between experiments for angle data.
    
    Compares all pairs of experiments at each specified day using the
    non-parametric Mann-Whitney U test. Results are written to a text file.
    
    Args:
        conf: Configuration dictionary with 'MainFolder' and 'daysAngles'
        data: DataFrame with angle measurements
        metric: Column name to analyze (e.g., 'Mean emergence angle')
    
    Returns:
        None (writes results to Stats.txt file)
    """
    data['Experiment'] = data['Experiment'].astype(str)
    data['Day'] = data['Day'].astype(int).astype(str)
        
    unique_experiments = data['Experiment'].unique()
    n_exp = int(len(unique_experiments))
    
    days = conf['daysAngles'].split(',')
    
    # Create output file path (handle metrics with '/' in name)
    report_path = os.path.join(conf['MainFolder'], 'Report', 'Angles Analysis')
    
    if "/" in metric:
        report_path_stats = os.path.join(report_path, '%s Stats.txt' % metric.replace('/', ' over '))
    else:
        report_path_stats = os.path.join(report_path, '%s Stats.txt' % metric)
        
    with open(report_path_stats, 'w') as f:
        f.write('Using Mann Whitney U test to compare different experiments\n')
        
        for day in days:
            f.write('Day: ' + str(day) + '\n')
            subdata = data[data['Day'] == str(day)]
            
            # Compare every pair of experiments
            for i in range(0, n_exp - 1):
                for j in range(i + 1, n_exp):
                    exp1 = subdata[subdata['Experiment'] == unique_experiments[i]][metric]
                    exp2 = subdata[subdata['Experiment'] == unique_experiments[j]][metric]
                    
                    try:
                        if len(exp1) == 0 or len(exp2) == 0:
                            raise Exception()
                            
                        U, p = stats.mannwhitneyu(exp1, exp2)
                        p = round(p, 6)
                        
                        # Report sample sizes
                        f.write('Number of samples ' + unique_experiments[i] + ': ' + 
                               str(len(exp1)) + ' - ')
                        f.write('Number of samples ' + unique_experiments[j] + ': ' + 
                               str(len(exp2)) + '\n')
                        
                        # Report means
                        f.write('Mean ' + unique_experiments[i] + ': ' + 
                               str(round(exp1.mean(), 2)) + ' - ')
                        f.write('Mean ' + unique_experiments[j] + ': ' + 
                               str(round(exp2.mean(), 2)) + '\n')
                        
                        # Report significance
                        if p < 0.05:
                            f.write('Experiments ' + unique_experiments[i] + ' and ' + 
                                   unique_experiments[j] + 
                                   ' are significantly different. P-value: ' + str(p) + '\n')
                        else:
                            f.write('Experiments ' + unique_experiments[i] + ' and ' + 
                                   unique_experiments[j] + 
                                   ' are not significantly different. P-value: ' + str(p) + '\n')
                    except:
                        f.write('Experiments ' + unique_experiments[i] + ' and ' + 
                               unique_experiments[j] + ' could not be compared\n')
                        
            f.write('\n')
    return


def performStatisticalAnalysisFirstLR(conf, data, metric):
    """
    Perform pairwise Mann-Whitney U tests for first lateral root data.
    
    Similar to performStatisticalAnalysisAngles but operates on time intervals
    (every N hours) rather than specific days. Optionally averages per plant
    before statistical comparison.
    
    Args:
        conf: Configuration dictionary with analysis parameters
        data: DataFrame with first LR tracking data
        metric: Column name to analyze (e.g., 'First LR tip')
    
    Returns:
        None (writes results to Stats.txt file)
    """
    data['Experiment'] = data['Experiment'].astype(str)
    unique_experiments = data['Experiment'].unique()
    n_exp = int(len(unique_experiments))
        
    # Create output file path
    report_path = os.path.join(conf['MainFolder'], 'Report', 'Angles Analysis')
    
    if "/" in metric:
        report_path_stats = os.path.join(report_path, '%s Stats.txt' % metric.replace('/', ' over '))
    else:
        report_path_stats = os.path.join(report_path, '%s Stats.txt' % metric)
    
    dt = int(conf['everyXhourFieldAngles'])

    with open(report_path_stats, 'w') as f:
        f.write('Using Mann Whitney U test to compare different experiments\n')
        f.write('Statistical analysis is performed every 6 hours\n')
        
        for hour in range(0, 73 - dt, dt):
            end = min(hour + dt, 72)
            f.write('Hour: ' + str(hour) + '-' + str(end) + '\n')
            
            hours = np.arange(hour, end)
            subdata = data[data['Time'].isin(hours)]

            # Optionally average per plant before comparison
            if conf['averagePerPlantStats']:
                subdata[metric] = subdata[metric].astype(float)
                subdata = subdata.groupby(['Experiment', 'Plant_id']).mean(
                    numeric_only=True
                ).reset_index()

            # Compare every pair of experiments
            for i in range(0, n_exp - 1):
                for j in range(i + 1, n_exp):
                    exp1 = subdata[subdata['Experiment'] == unique_experiments[i]][metric]
                    exp2 = subdata[subdata['Experiment'] == unique_experiments[j]][metric]

                    try:
                        if len(exp1) == 0 or len(exp2) == 0:
                            raise Exception()
                            
                        U, p = stats.mannwhitneyu(exp1, exp2)
                        p = round(p, 6)

                        # Report sample sizes
                        f.write('Number of samples ' + unique_experiments[i] + ': ' + 
                               str(len(exp1)) + ' - ')
                        f.write('Number of samples ' + unique_experiments[j] + ': ' + 
                               str(len(exp2)) + '\n')
                        
                        # Report means
                        f.write('Mean ' + unique_experiments[i] + ': ' + 
                               str(round(exp1.mean(), 2)) + ' - ')
                        f.write('Mean ' + unique_experiments[j] + ': ' + 
                               str(round(exp2.mean(), 2)) + '\n')
                        
                        # Report significance
                        if p < 0.05:
                            f.write('Experiments ' + unique_experiments[i] + ' and ' + 
                                   unique_experiments[j] + 
                                   ' are significantly different. P-value: ' + str(p) + '\n')
                        else:
                            f.write('Experiments ' + unique_experiments[i] + ' and ' + 
                                   unique_experiments[j] + 
                                   ' are not significantly different. P-value: ' + str(p) + '\n')
                    except:
                        f.write('Experiments ' + unique_experiments[i] + ' and ' + 
                               unique_experiments[j] + ' could not be compared\n')
    
            f.write('\n')
    return


def estimateAngles(path, ax, img, i=-1, tip=False):
    """
    Estimate and visualize lateral root angles on an image.
    
    Parses the RSML file, calculates angles for all lateral roots,
    and plots the angle measurement lines on the provided axes.
    
    Args:
        path: Path to the analysis results folder
        ax: Matplotlib axes to draw on
        img: Image array to display as background
        i: Index of the RSML file to use (-1 = last file)
        tip: If True, show tip angles; if False, show emergence angles
    
    Returns:
        ax: Modified matplotlib axes with angle visualizations
    """
    plt.ioff()
    paths = loadPath(os.path.join(path, 'RSML'))

    # Initialize tracking variables
    lateral_roots = []
    lateral_root_starts = []
    lateral_root_names = []
    num_roots = 0

    step = paths[i]
    tree = ET.parse(step).getroot()

    with open(os.path.join(path, 'metadata.json')) as f:
        metadata = json.load(f)

    y1, y2, x1, x2 = metadata['bounding box']
    h = y2 - y1
    w = x2 - x1

    plant = tree[1][0][0]
    
    # Check if lateral roots exist
    if len(plant) > 1:
        lateral_root_elements = plant[1:]
        
        points_list = []
        start_points = []

        # First: perform tracking
        for lr_element in lateral_root_elements:
            points = traverse_root(lr_element, is_lateral_root=True)
            points_list.append(points)
            start_points.append(points[0])

        lateral_root_starts, lateral_roots, lateral_root_names, num_roots = matching(
            start_points, points_list, 
            lateral_root_starts, lateral_roots, lateral_root_names, 
            num_roots
        )
        
        # Second: estimate angles
        tip_angles = []
        emergence_angles = []

        for root in lateral_roots:
            tip_angle = tipAngle(root)
            emergence_angle = emergenceAngle(root, 2, metadata['pixel_size'])
            tip_angles.append(tip_angle)
            emergence_angles.append(emergence_angle)

        # Display image and overlay angle measurements
        ax.imshow(img)
        
        for i in range(len(points_list)):
            if not tip:
                plotAngles(ax, points_list[i], emergence_angles[i], tip=tip)
            else:
                plotAngles(ax, points_list[i], tip_angles[i], tip=tip)
        ax.axis('off')
    
    return ax


def plotLateralAnglesOnTop(conf):
    """
    Generate angle visualization overlays for all plants.
    
    Creates images showing the emergence and tip angle measurements
    overlaid on the original root images for visual verification.
    
    Args:
        conf: Configuration dictionary with 'MainFolder' path
    
    Returns:
        None (saves PNG and SVG images to Report folder)
    """
    exp_path = os.path.join(conf['MainFolder'], "Analysis")
    save_path = os.path.join(conf['MainFolder'], "Report", "Angles Analysis", "Images")
    os.makedirs(save_path, exist_ok=True)

    experiments = os.listdir(exp_path)

    for exp in experiments:
        robot_path = os.path.join(exp_path, exp)

        for robot in os.listdir(robot_path):
            cam_path = os.path.join(robot_path, robot)

            for cam in os.listdir(cam_path):
                plant_path = os.path.join(cam_path, cam)

                for plant in os.listdir(plant_path):
                    results_path = os.listdir(os.path.join(plant_path, plant))

                    if len(results_path) > 0:
                        results_path = os.path.join(plant_path, plant, results_path[-1])
                    else:
                        continue
                    
                    if os.path.exists(results_path):
                        metadata_path = results_path + "/metadata.json"
                        metadata = json.load(open(metadata_path))
                        
                        i = -1

                        # Load list of postprocessed images
                        images = pd.read_csv(os.path.join(results_path, "FilesAfterPostprocessing.csv"))
                        images.dropna(inplace=True)
                        images = images["FileName"].tolist()

                        rsml_files = os.listdir(os.path.join(results_path, "RSML"))

                        # Filter to images with corresponding RSML
                        images = [image for image in images 
                                 if image.split('/')[-1].replace('.png', '.rsml') in rsml_files]
                        
                        try:
                            images = [os.path.join(metadata["ImagePath"], image) for image in images]
                        except:
                            # Fallback for retrocompatibility
                            images = [os.path.join(metadata["folder"], image) for image in images]

                        # Load and crop the image
                        bbox = metadata["bounding box"]
                        crop = cv2.imread(images[i])[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                        
                        # Save cropped image
                        save = exp + "_" + robot + "_" + cam + "_" + plant + "_crop.png"
                        cv2.imwrite(os.path.join(save_path, save), crop)

                        # Generate emergence angles visualization
                        fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
                        estimateAngles(results_path, ax, crop.copy(), i)
                        plt.title("Emergence Angles")

                        save = exp + "_" + robot + "_" + cam + "_" + plant + "_emergence_angles.svg"
                        plt.savefig(os.path.join(save_path, save), bbox_inches="tight", dpi=200)

                        plt.close()
                        plt.clf()
                        plt.cla()

                        # Generate tip angles visualization
                        fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
                        estimateAngles(results_path, ax, crop.copy(), i, True)
                        plt.title("Tip Angles")

                        save = exp + "_" + robot + "_" + cam + "_" + plant + "_tip_angles.svg"
                        plt.savefig(os.path.join(save_path, save), bbox_inches="tight", dpi=200)

                        plt.close()
                        plt.clf()
                        plt.cla()