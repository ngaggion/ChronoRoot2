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
    generateTableTemporal,
)

from analysis.fourier_analysis import makeFourierPlots
from analysis.lateral_angles import makeLateralAnglesPlots, plotLateralAnglesOnTop
from analysis.fpca_analysis import performFPCA
from analysis.utils.fileUtilities import (
    convertFromPathSafe,
    get_latest_result_dir,
    load_result_metadata,
    attach_plant_metadata_columns,
    build_plant_id,
    normalize_factor_value,
)
from analysis.utils.report_paths import (
    data_file,
    individual_plots_dir,
    module_dir,
    MODULE_CONVEX,
    reset_report_index,
    write_report_index,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChronoRoot: Report Generation')
    parser.add_argument('--config', type=str, help='Path to the configuration file (default: config.json)')

    args = parser.parse_args()
    conf = json.load(open(args.config, 'r'))

    analysis_folder = os.path.join(conf['MainFolder'], 'Analysis')
    experiments = utils.load_paths(analysis_folder, '*')

    utils.ensure_directory(os.path.join(conf['MainFolder'], 'Report'))
    reset_report_index()

    all_data = pd.DataFrame()
    convex_hull_df = pd.DataFrame()

    print("Report generation began. This may take a while.")

    FORCE_REPORT = True

    temporal_parameters = [
        'MainRootLength (mm)', 'LateralRootsLength (mm)', 'TotalLength (mm)',
        'NumberOfLateralRoots', 'DiscreteLateralDensity (LR/cm)', 'MainOverTotal (%)',
        'HypocotylLength (mm)',
    ]

    if conf['doConvex']:
        global_shape, global_center = convex_hull.calculate_atlas_geometry(experiments)

    temporal_data_path = data_file(conf, 'Temporal_Data.csv')

    if not os.path.exists(temporal_data_path) or FORCE_REPORT:
        for exp_dir in experiments:
            exp_dir_name = os.path.basename(exp_dir)
            real_exp_name = convertFromPathSafe(exp_dir_name)

            first_meta = utils.load_paths(exp_dir, '*/*/*/metadata.json')
            if first_meta:
                try:
                    with open(first_meta[0], 'r') as f:
                        meta_data = json.load(f)
                        real_exp_name = meta_data.get('Experiment', real_exp_name)
                except Exception:
                    pass

            print(f'Loading experiment: {real_exp_name}')

            iplots_exp_folder = individual_plots_dir(conf, exp_dir_name)

            rpi_paths = utils.load_paths(exp_dir, '*')
            for rpi in rpi_paths:
                rpi_name = os.path.basename(rpi)
                cam_paths = utils.load_paths(rpi, '*')
                for cam in cam_paths:
                    cam_name = os.path.basename(cam)
                    plant_paths = utils.load_paths(cam, '*')
                    for plant in plant_paths:
                        plant_name = os.path.basename(plant)
                        res_folder = get_latest_result_dir(plant)
                        if res_folder is None:
                            continue

                        meta = load_result_metadata(res_folder)
                        plant_id = build_plant_id(rpi_name, cam_name, plant_name)
                        file_csv = os.path.join(res_folder, 'PostProcess_Hour.csv')

                        if not os.path.exists(file_csv):
                            continue

                        data = pd.read_csv(file_csv)
                        data['Plant_id'] = plant_id
                        data = attach_plant_metadata_columns(
                            data, meta,
                            rpi_name=rpi_name,
                            cam_name=cam_name,
                            plant_name=plant_name,
                            experiment_fallback=real_exp_name,
                        )

                        all_data = pd.concat([all_data, data], ignore_index=True)

                        plot_filename = f"{exp_dir_name}_{plant_id}.png"
                        iplot_cache = os.path.join(res_folder, plot_filename)
                        report_dest = os.path.join(iplots_exp_folder, plot_filename)

                        if not os.path.exists(iplot_cache):
                            plot_individual_plant(iplots_exp_folder, data, plot_filename)
                            if os.path.exists(report_dest):
                                shutil.copy(report_dest, iplot_cache)
                        else:
                            shutil.copy(iplot_cache, report_dest)

            if conf['doConvex']:
                print(f"Performing convex hull analysis for experiment: {real_exp_name}")
                days = conf['daysConvexHull'].split(',')
                convex_overview = module_dir(conf, MODULE_CONVEX)

                atlases, current_convex_df = convex_hull.generate_root_atlases(
                    exp_dir,
                    days=days,
                    timestep=conf['timeStep'],
                    canvas_shape=global_shape,
                    center_coords=global_center,
                    rotate_root=True,
                )

                if not current_convex_df.empty:
                    current_convex_df['Experiment'] = real_exp_name
                    convex_hull_df = pd.concat([convex_hull_df, current_convex_df], ignore_index=True)

                if conf['saveImagesConvex'] and atlases:
                    for i in range(len(days)):
                        at_hull, at_cont, at_root = atlases[i]
                        convex_hull.visualize_single_atlas(
                            at_hull, at_cont, at_root,
                            convex_overview, exp_dir_name, days[i],
                        )
                elif atlases:
                    at_hull, at_cont, at_root = atlases[-1]
                    convex_hull.visualize_single_atlas(
                        at_hull, at_cont, at_root,
                        convex_overview, exp_dir_name,
                    )

        all_data.to_csv(temporal_data_path, index=False)
    else:
        all_data = pd.read_csv(temporal_data_path)
        all_data['Experiment'] = all_data['Experiment'].astype(str)
        if 'PlateCondition' in all_data.columns:
            all_data['PlateCondition'] = all_data['PlateCondition'].apply(normalize_factor_value)
        if 'ExtraVariable' in all_data.columns:
            all_data['ExtraVariable'] = all_data['ExtraVariable'].apply(normalize_factor_value)

    for parameter in temporal_parameters:
        performStatisticalAnalysis(conf, all_data, parameter)

    plot_info_all(conf, all_data)
    generateTableTemporal(conf, all_data)

    if conf['doFPCA']:
        performFPCA(args.config)

    if conf['doConvex'] and not convex_hull_df.empty:
        convex_hull_df.to_csv(data_file(conf, 'Convex_Hull_Data.csv'), index=False)
        convex_hull.plot_hull_metrics_summary(conf, convex_hull_df)
        convex_hull.visualize_combined_atlases(conf)

        convex_params = [
            'Convex Hull Area', 'Lateral Root Area Density', 'Total Root Area Density',
            'Convex Hull Aspect Ratio', 'Convex Hull Height', 'Convex Hull Width',
        ]
        for param in convex_params:
            convex_hull.analyze_hull_statistics(conf, convex_hull_df, param)

    if conf['doFourier']:
        makeFourierPlots(conf)

    if conf['doLateralAngles']:
        makeLateralAnglesPlots(conf)
        plotLateralAnglesOnTop(conf)

    index_path = write_report_index(conf)
    print(f"Report index written to {index_path}")
    print("Report generation finished.")
