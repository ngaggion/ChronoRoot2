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

import shutil
import pandas as pd
import os
import json
from analysis.report import load_path, plot_individual_plant, mkdir, plot_info_all, get_atlases, plot_atlases
from analysis.report import plot_combined_atlases, plot_convex_hull, performStatisticalAnalysis
from analysis.report import performStatisticalAnalysisConvexHull, generateTableTemporal
from analysis.fourier_analysis import makeFourierPlots
from analysis.lateral_angles import makeLateralAnglesPlots, plotLateralAnglesOnTop
import subprocess

if __name__ == "__main__":
    conf = json.load(open('config.json'))
    analysis = os.path.join(conf['MainFolder'],'Analysis')
    experiments = load_path(analysis, '*')

    reportPath = os.path.join(conf['MainFolder'],'Report')
    if not os.path.exists(reportPath):
        os.makedirs(reportPath)
    
    all_data = pd.DataFrame()
    convex_hull = pd.DataFrame()
    reportPath_convex = os.path.join(reportPath, 'Convex Hull and Area Analysis')

    print("Report generation began. This may take a while.")
    
    FORCE_REPORT = True
    
    if not FORCE_REPORT and conf['doConvex'] and not os.path.exists(os.path.join(reportPath, 'Convex_Hull_Data.csv')):
        if not os.path.exists(reportPath_convex):
            os.makedirs(reportPath_convex)
        print("Convex hull analysis not found, forcing report generation")
        FORCE_REPORT = True
    
    individual_plots_folder = os.path.join(reportPath, 'Individual plant plots')
    mkdir(individual_plots_folder)
    
    temporal_parameters = ['MainRootLength (mm)', 'LateralRootsLength (mm)', 'TotalLength (mm)', 'NumberOfLateralRoots', 'DiscreteLateralDensity (LR/cm)', 'MainOverTotal (%)']
    temp_folder = os.path.join(reportPath, 'Temporal Parameters')
    mkdir(temp_folder)
    
    # Check if all_data.csv exists
    if not os.path.exists(os.path.join(reportPath, 'Temporal_Data.csv')) or FORCE_REPORT:
        for exp in experiments:
            exp_name = exp.replace(analysis, '').replace('/','')
            print('Loading experiment:', exp_name)

            iplots = os.path.join(individual_plots_folder, exp_name)
            mkdir(iplots)

            rpi = load_path(exp, '*')
            for rpi in rpi:
                rpi_name = rpi.replace(exp, '').replace('/','')

                cam = load_path(rpi, '*')
                for cam in cam:
                    cam_name = cam.replace(rpi, '').replace('/','')

                    plants = load_path(cam, '*')
                    for plant in plants:
                        plant_name = plant.replace(cam, '').replace('/','')

                        results = load_path(plant, '*')
                        
                        if len(results) == 0:
                            continue
                        else:
                            results = results[-1]
                            
                        name = rpi_name + '_' + cam_name + '_' + plant_name

                        file = os.path.join(results,'PostProcess_Hour.csv')
                        data = pd.read_csv(file)

                        data['Plant_id'] = name
                        data['Experiment'] = exp_name

                        all_data = pd.concat([all_data, data], ignore_index=True)
                        
                        iplot = os.path.join(results, exp_name + "_" + name + ".png")
                        shutil.copy(iplot, os.path.join(iplots, exp_name + "_" + name+".png"))

            if conf['doConvex']:
                print("Performing convex hull analysis for experiment:", exp_name)
                # Convex hull experiment
                if not os.path.exists(reportPath_convex):
                    os.makedirs(reportPath_convex)

                days = conf['daysConvexHull'].split(',')
                atlases, convexDF = get_atlases(exp, days = days, rotateRoot = True)
                convexDF['Experiment'] = exp_name

                convex_hull = pd.concat([convex_hull, convexDF], ignore_index=True)

                if conf['saveImagesConvex']:
                    for i in range(0, len(days)):
                        at1, at2, at3 = atlases[i]
                        plot_atlases(at1, at2, at3, reportPath_convex, exp_name, days[i])
                else:
                    at1, at2, at3 = atlases[-1]
                    plot_atlases(at1, at2, at3, reportPath_convex, exp_name)

        # save dataframes to files
        all_data.to_csv(os.path.join(reportPath, 'Temporal_Data.csv'), index=False)
    else:
        all_data = pd.read_csv(os.path.join(reportPath, 'Temporal_Data.csv'))
        all_data['Experiment'] = all_data['Experiment'].astype(str)

    print("Generating temporal parameter plots.")
    for parameter in temporal_parameters:
        performStatisticalAnalysis(conf, all_data, parameter)
    plot_info_all(os.path.join(reportPath, 'Temporal Parameters'), all_data)
    generateTableTemporal(conf, all_data)
    
    if conf['doFPCA']:
        command = [
            "conda", "run", "-n", "FDA", "python", "analysis/fpca_analysis.py", "config.json"
        ]
        subprocess.run(command, check=True)
    
    if conf['doConvex']:
        print("Generating convex hull and area analysis plots.")
        convex_hull.to_csv(os.path.join(reportPath, 'Convex_Hull_Data.csv'), index=False)
        plot_convex_hull(reportPath_convex, convex_hull)
        plot_combined_atlases(reportPath_convex)

        convex_hull_parameters = ['Convex Hull Area', 'Lateral Root Area Density', 
                                  'Total Root Area Density', 'Convex Hull Aspect Ratio', 
                                  'Convex Hull Height', 'Convex Hull Width']
        
        for parameter in convex_hull_parameters:
            performStatisticalAnalysisConvexHull(conf, convex_hull, parameter)

    if conf['doFourier']:
        print("Generating Fourier analysis plots.")      
        makeFourierPlots(conf)
    
    if conf['doLateralAngles']:
        print("Generating lateral angles analysis plots.")
        makeLateralAnglesPlots(conf)
        plotLateralAnglesOnTop(conf)

    print("Report generation finished.")
    