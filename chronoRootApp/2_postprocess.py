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

from analysis.dataWork import dataWork
from analysis.qr import qr_detect, get_pixel_size, load_path
from analysis.report import plot_individual_plant
from analysis.lateral_angles import getAngles
from analysis.utils.fileUtilities import convertFromPathSafe
import json
import os 
import pandas as pd
import argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='ChronoRoot Post-processing')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
       
    conf = json.load(open(parser.parse_args().config, 'r'))
    analysis = os.path.join(conf['MainFolder'], 'Analysis')

    # 'varieties' are just directory paths now
    experiment_dirs = load_path(analysis, '*')

    print('Post processing started.')
    
    # --- Cleanup Phase ---
    for exp_dir in experiment_dirs:
        rpis = load_path(exp_dir, '*')
        for rpi in rpis:
            cams = load_path(rpi, '*')
            for cam in cams:
                plants = load_path(cam, '*')
                for plant in plants:
                    results = load_path(plant, '*')
                    if len(results) == 0:
                        os.rmdir(plant)
    
    # --- Processing Phase ---
    for exp_dir in experiment_dirs:
        print(f'Processing Experiment: {convertFromPathSafe(exp_dir)}')
                
        rpis = load_path(exp_dir, '*')
        for rpi in rpis:
            cams = load_path(rpi, '*')
            for cam in cams:
                plants = load_path(cam, '*')
                
                if len(plants) == 0:
                    continue
                
                # Use first plant to establish calibration/metadata for the group
                sample_results = load_path(plants[0], '*')
                if len(sample_results) == 0:
                    continue
                
                res_path = sample_results[-1]
                with open(os.path.join(res_path, 'metadata.json'), 'r') as f:
                    metadata = json.load(f)

                # 1. Determine Experiment Name from Metadata
                # Fallback to folder name if 'Experiment' key is missing
                raw_exp_name = metadata.get('Experiment', os.path.basename(exp_dir))
                readable_name = convertFromPathSafe(raw_exp_name)
                
                # 2. Pixel Size Calibration Logic
                try:
                    pixel_size = metadata['pixel_size']
                except KeyError:
                    # [Calibration Logic simplified for brevity, keeping your existing logic]
                    if not conf.get('videoHasQRbutton', True):
                        pixel_size = float(conf['knownDistance']) / float(conf['pixelDistance'])
                    else:
                        image_path = metadata['ImagePath']
                        images = load_path(image_path, '*.png')
                        pixel_size = 0.04 # Default
                        for i, image in enumerate(images[:20]):
                            qr = qr_detect(image)
                            if qr is not None:
                                pixel_size = 10 / get_pixel_size(qr[0])
                                break
                
                # 3. Process Individual Plants
                for plant in plants:
                    plant_results = load_path(plant, '*')
                    if len(plant_results) == 0:
                        continue
                    
                    target_res = plant_results[-1]
                    meta_file = os.path.join(target_res, 'metadata.json')
                    
                    with open(meta_file, 'r') as f:
                        plant_metadata = json.load(f)

                    # Update metadata with calculated pixel size
                    plant_metadata['pixel_size'] = pixel_size
                    with open(meta_file, 'w') as f:
                        json.dump(plant_metadata, f)

                    # Run analysis
                    n_limit = conf['Limit'] if conf['Limit'] != 0 else None
                    pfile = os.path.join(target_res, 'Results_raw.csv')
                    
                    try:
                        dataWork(conf, pfile, target_res, N_exp=n_limit)
                    except Exception as e:
                        print(f"Error processing {pfile}, experiment may have not finished yet. Error: {e}")
                        continue

                    # 4. Generate Plot Name with path components
                    plot_label = f"{os.path.basename(exp_dir)}_{os.path.basename(rpi)}_{os.path.basename(cam)}_{os.path.basename(plant)}"
                    
                    processed_csv = os.path.join(target_res, 'PostProcess_Hour.csv')
                    if os.path.exists(processed_csv):
                        data = pd.read_csv(processed_csv)
                        plot_individual_plant(target_res, data, plot_label)

                    getAngles(conf, target_res)
                    
    print('Post processing finished.')