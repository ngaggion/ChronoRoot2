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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture')
    parser.add_argument('--config', type=str, help='Path to the configuration file (default: config.json)')
       
    conf = json.load(open(parser.parse_args().config, 'r'))
    
    # Use utils.load_paths instead of load_path
    analysis_folder = os.path.join(conf['MainFolder'], 'Analysis')
    experiments = utils.load_paths(analysis_folder, '*')

    reportPath = os.path.join(conf['MainFolder'], 'Report')
    utils.ensure_directory(reportPath)
    
    all_data = pd.DataFrame()
    convex_hull_df = pd.DataFrame()
    reportPath_convex = os.path.join(reportPath, 'Convex Hull and Area Analysis')

    print("Report generation began. This may take a while.")
    
    FORCE_REPORT = True
    
    # Logic to check if we can skip hull generation
    if not FORCE_REPORT and conf['doConvex'] and not os.path.exists(os.path.join(reportPath, 'Convex_Hull_Data.csv')):
        if not os.path.exists(reportPath_convex):
            utils.ensure_directory(reportPath_convex)
        print("Convex hull analysis not found, forcing report generation")
        FORCE_REPORT = True
    
    individual_plots_folder = os.path.join(reportPath, 'Individual plant plots')
    utils.ensure_directory(individual_plots_folder)
    
    temporal_parameters = [
        'MainRootLength (mm)', 'LateralRootsLength (mm)', 'TotalLength (mm)', 
        'NumberOfLateralRoots', 'DiscreteLateralDensity (LR/cm)', 'MainOverTotal (%)'
    ]
    temp_folder = os.path.join(reportPath, 'Temporal Parameters')
    utils.ensure_directory(temp_folder)
    
    # --- 1. Global Configuration for Atlases ---
    if conf['doConvex']:
        days = conf['daysConvexHull'].split(',')
        # Updated Function Call: calculate_atlas_geometry
        global_shape, global_center = convex_hull.calculate_atlas_geometry(experiments)
        
    # --- 2. Main Data Loading Loop ---
    if not os.path.exists(os.path.join(reportPath, 'Temporal_Data.csv')) or FORCE_REPORT:
        for exp in experiments:
            exp_name = os.path.basename(exp)
            print('Loading experiment:', exp_name)

            iplots = os.path.join(individual_plots_folder, exp_name)
            utils.ensure_directory(iplots)

            # Load hierarchy using shared utility
            rpi_paths = utils.load_paths(exp, '*')
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
                        else:
                            results = results[-1]
                            
                        name = f"{rpi_name}_{cam_name}_{plant_name}"

                        file_csv = os.path.join(results, 'PostProcess_Hour.csv')
                        if not os.path.exists(file_csv): continue
                        
                        data = pd.read_csv(file_csv)
                        data['Plant_id'] = name
                        data['Experiment'] = exp_name

                        all_data = pd.concat([all_data, data], ignore_index=True)
                        
                        # Handle individual plots
                        iplot_dest = os.path.join(results, f"{exp_name}_{name}.png")
                        final_dest = os.path.join(iplots, f"{exp_name}_{name}.png")

                        if not os.path.exists(iplot_dest):
                            # Generate plot if missing
                            plot_individual_plant(iplots, data, f"{exp_name}_{name}.png")
                            # Copy back to results folder for cache
                            if os.path.exists(final_dest):
                                shutil.copy(final_dest, iplot_dest)
                        else:
                            # Copy from cache to report
                            shutil.copy(iplot_dest, final_dest)

            # --- 3. Convex Hull Analysis per Experiment ---
            if conf['doConvex']:
                print("Performing convex hull analysis for experiment:", exp_name)
                utils.ensure_directory(reportPath_convex)

                days = conf['daysConvexHull'].split(',')
                
                # Updated Function Call: generate_root_atlases
                atlases, current_convex_df = convex_hull.generate_root_atlases(
                    exp, 
                    days=days, 
                    timestep=conf['timeStep'], 
                    canvas_shape=global_shape,  
                    center_coords=global_center,       
                    rotate_root=True
                )
                
                if not current_convex_df.empty:
                    current_convex_df['Experiment'] = exp_name
                    convex_hull_df = pd.concat([convex_hull_df, current_convex_df], ignore_index=True)

                # Plot Atlases (Heatmaps)
                # Updated Function Call: visualize_single_atlas
                if conf['saveImagesConvex']:
                    for i in range(len(days)):
                        at_hull, at_cont, at_root = atlases[i]
                        convex_hull.visualize_single_atlas(
                            at_hull, at_cont, at_root, 
                            reportPath_convex, exp_name, days[i]
                        )
                elif atlases:
                    # Plot only the last one
                    at_hull, at_cont, at_root = atlases[-1]
                    convex_hull.visualize_single_atlas(
                        at_hull, at_cont, at_root, 
                        reportPath_convex, exp_name
                    )

        # Save temporal data
        all_data.to_csv(os.path.join(reportPath, 'Temporal_Data.csv'), index=False)
    else:
        all_data = pd.read_csv(os.path.join(reportPath, 'Temporal_Data.csv'))
        all_data['Experiment'] = all_data['Experiment'].astype(str)

    # --- 4. Temporal Stats & Plots ---
    print("Generating temporal parameter plots.")
    for parameter in temporal_parameters:
        performStatisticalAnalysis(conf, all_data, parameter)
    
    plot_info_all(os.path.join(reportPath, 'Temporal Parameters'), all_data)
    generateTableTemporal(conf, all_data)
    
    if conf['doFPCA']:
        performFPCA(parser.parse_args().config)
    
    # --- 5. Convex Hull Stats & Plots ---
    if conf['doConvex'] and not convex_hull_df.empty:
        print("Generating convex hull and area analysis plots.")
        convex_hull_df.to_csv(os.path.join(reportPath, 'Convex_Hull_Data.csv'), index=False)
        
        # Updated Function Call: plot_hull_metrics_summary (Handles violins and summary table)
        convex_hull.plot_hull_metrics_summary(reportPath_convex, convex_hull_df)
        
        # Updated Function Call: visualize_combined_atlases (Qualitative comparison)
        convex_hull.visualize_combined_atlases(reportPath_convex)

        convex_hull_parameters = [
            'Convex Hull Area', 'Lateral Root Area Density', 
            'Total Root Area Density', 'Convex Hull Aspect Ratio', 
            'Convex Hull Height', 'Convex Hull Width'
        ]
        
        # Updated Function Call: analyze_hull_statistics (Mann-Whitney U)
        for parameter in convex_hull_parameters:
            convex_hull.analyze_hull_statistics(conf, convex_hull_df, parameter)

    # --- 6. Fourier & Angle Analysis ---
    if conf['doFourier']:
        print("Generating Fourier analysis plots.")      
        makeFourierPlots(conf)
    
    if conf['doLateralAngles']:
        print("Generating lateral angles analysis plots.")
        makeLateralAnglesPlots(conf)
        plotLateralAnglesOnTop(conf)

    print("Report generation finished.")