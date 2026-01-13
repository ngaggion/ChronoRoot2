""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion
"""

import shutil
import pandas as pd
import os
import json
import argparse

from analysis.utils import report_utils as utils
from analysis import convex_hull
from analysis.report import (
    plot_individual_plant, 
    plot_info_all, 
    performStatisticalAnalysis, 
    generateTableTemporal
)

from analysis.fourier_analysis import makeFourierPlots
from analysis.lateral_angles import makeLateralAnglesPlots, plotLateralAnglesOnTop
from analysis.fpca_analysis import performFPCA
from analysis.utils.fileUtilities import convertToPathSafe, convertFromPathSafe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChronoRoot: Report Generation')
    parser.add_argument('--config', type=str, help='Path to the configuration file (default: config.json)')
       
    args = parser.parse_args()
    conf = json.load(open(args.config, 'r'))
    
    analysis_folder = os.path.join(conf['MainFolder'], 'Analysis')
    experiments = utils.load_paths(analysis_folder, '*')

    reportPath = os.path.join(conf['MainFolder'], 'Report')
    utils.ensure_directory(reportPath)
    
    all_data = pd.DataFrame()
    convex_hull_df = pd.DataFrame()
    reportPath_convex = os.path.join(reportPath, 'Convex Hull and Area Analysis')

    print("Report generation began. This may take a while.")
    
    FORCE_REPORT = True 
    
    # Check if we can skip hull generation
    if not FORCE_REPORT and conf['doConvex'] and not os.path.exists(os.path.join(reportPath, 'Convex_Hull_Data.csv')):
        if not os.path.exists(reportPath_convex):
            utils.ensure_directory(reportPath_convex)
        print("Convex hull analysis not found, forcing report generation")
        FORCE_REPORT = True
    
    individual_plots_folder = os.path.join(reportPath, 'Individual plant plots')
    utils.ensure_directory(individual_plots_folder)
    
    temporal_parameters = [
        'MainRootLength (mm)', 'LateralRootsLength (mm)', 'TotalLength (mm)', 
        'NumberOfLateralRoots', 'DiscreteLateralDensity (LR/cm)', 'MainOverTotal (%)',
        'HypocotylLength (mm)'
    ]
    
    # --- 1. Global Configuration for Atlases ---
    if conf['doConvex']:
        global_shape, global_center = convex_hull.calculate_atlas_geometry(experiments)
        
    # --- 2. Main Data Loading Loop ---
    temporal_data_path = os.path.join(reportPath, 'Temporal_Data.csv')
    
    if not os.path.exists(temporal_data_path) or FORCE_REPORT:
        for exp_dir in experiments:
            # Determine Names: Check metadata for 'Experiment', fallback to path-safe folder name
            exp_dir_name = os.path.basename(exp_dir)
            real_exp_name = convertFromPathSafe(exp_dir_name) # Default fallback
            
            first_meta = utils.load_paths(exp_dir, '*/*/*/metadata.json')
            if first_meta:
                try:
                    with open(first_meta[0], 'r') as f:
                        meta_data = json.load(f)
                        real_exp_name = meta_data.get('Experiment', real_exp_name)
                except:
                    pass

            print(f'Loading experiment: {real_exp_name}')

            iplots_exp_folder = os.path.join(individual_plots_folder, exp_dir_name)
            utils.ensure_directory(iplots_exp_folder)

            rpi_paths = utils.load_paths(exp_dir, '*')
            for rpi in rpi_paths:
                rpi_name = os.path.basename(rpi)
                cam_paths = utils.load_paths(rpi, '*')
                for cam in cam_paths:
                    cam_name = os.path.basename(cam)
                    plant_paths = utils.load_paths(cam, '*')
                    for plant in plant_paths:
                        plant_name = os.path.basename(plant)
                        results = utils.load_paths(plant, '*')
                        
                        if len(results) == 0:
                            continue
                        res_folder = results[-1]
                        
                        plant_id = f"{rpi_name}_{cam_name}_{plant_name}"
                        file_csv = os.path.join(res_folder, 'PostProcess_Hour.csv')
                        
                        if not os.path.exists(file_csv): 
                            continue
                        
                        data = pd.read_csv(file_csv)
                        data['Plant_id'] = plant_id
                        data['Experiment'] = real_exp_name 

                        all_data = pd.concat([all_data, data], ignore_index=True)
                        
                        # Handle individual plots
                        plot_filename = f"{exp_dir_name}_{plant_id}.png"
                        iplot_cache = os.path.join(res_folder, plot_filename)
                        report_dest = os.path.join(iplots_exp_folder, plot_filename)

                        if not os.path.exists(iplot_cache):
                            plot_individual_plant(iplots_exp_folder, data, plot_filename)
                            if os.path.exists(report_dest):
                                shutil.copy(report_dest, iplot_cache)
                        else:
                            shutil.copy(iplot_cache, report_dest)

            # --- 3. Convex Hull Analysis per Experiment ---
            if conf['doConvex']:
                print(f"Performing convex hull analysis for experiment: {real_exp_name}")
                utils.ensure_directory(reportPath_convex)
                days = conf['daysConvexHull'].split(',')
                
                atlases, current_convex_df = convex_hull.generate_root_atlases(
                    exp_dir, 
                    days=days, 
                    timestep=conf['timeStep'], 
                    canvas_shape=global_shape,  
                    center_coords=global_center,       
                    rotate_root=True
                )
                
                if not current_convex_df.empty:
                    current_convex_df['Experiment'] = real_exp_name
                    convex_hull_df = pd.concat([convex_hull_df, current_convex_df], ignore_index=True)

                if conf['saveImagesConvex'] and atlases:
                    for i in range(len(days)):
                        at_hull, at_cont, at_root = atlases[i]
                        convex_hull.visualize_single_atlas(
                            at_hull, at_cont, at_root, 
                            reportPath_convex, exp_dir_name, days[i]
                        )
                elif atlases:
                    at_hull, at_cont, at_root = atlases[-1]
                    convex_hull.visualize_single_atlas(
                        at_hull, at_cont, at_root, 
                        reportPath_convex, exp_dir_name
                    )

        all_data.to_csv(temporal_data_path, index=False)
    else:
        all_data = pd.read_csv(temporal_data_path)
        all_data['Experiment'] = all_data['Experiment'].astype(str)

    # --- 4. Final Processing & Stats ---
    temp_plots_dir = os.path.join(reportPath, 'Temporal Parameters')
    utils.ensure_directory(temp_plots_dir)
    
    for parameter in temporal_parameters:
        performStatisticalAnalysis(conf, all_data, parameter)
    
    plot_info_all(temp_plots_dir, all_data)
    generateTableTemporal(conf, all_data)
    
    if conf['doFPCA']:
        performFPCA(args.config)
    
    if conf['doConvex'] and not convex_hull_df.empty:
        convex_hull_df.to_csv(os.path.join(reportPath, 'Convex_Hull_Data.csv'), index=False)
        convex_hull.plot_hull_metrics_summary(reportPath_convex, convex_hull_df)
        convex_hull.visualize_combined_atlases(reportPath_convex)
        
        convex_params = ['Convex Hull Area', 'Lateral Root Area Density', 'Total Root Area Density', 'Convex Hull Aspect Ratio', 'Convex Hull Height', 'Convex Hull Width']
        for param in convex_params:
            convex_hull.analyze_hull_statistics(conf, convex_hull_df, param)

    if conf['doFourier']:
        makeFourierPlots(conf)
    
    if conf['doLateralAngles']:
        makeLateralAnglesPlots(conf)
        plotLateralAnglesOnTop(conf)

    print("Report generation finished.")