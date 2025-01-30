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

import json
import os 
import pandas as pd

if __name__ == "__main__":    
    conf = json.load(open('config.json'))
    analysis = os.path.join(conf['MainFolder'],'Analysis')

    varieties = load_path(analysis, '*')

    print('Post processing started.')
    
    for variety in varieties:
        print('Processing ' + variety)
        rpi = load_path(variety, '*')
        for rpi in rpi:
            cam = load_path(rpi, '*')
            for cam in cam:
                plants = load_path(cam, '*')
                
                # Reads QR only once per cam
                if len(plants) == 0:
                    continue
                
                results = load_path(plants[0], '*')

                if len(results) == 0:
                    continue
                else:
                    results = results[-1]
                    
                metadata = json.load(open(results + '/metadata.json', 'r'))

                try:
                    pixel_size = metadata['pixel_size']
                except:
                    image_path = metadata['ImagePath']
                    images = load_path(image_path, '*.png')
                    
                    k = 0
                    for image in images:
                        qr = qr_detect(image)
                        if qr is not None:
                            pixel_size = 10 / get_pixel_size(qr[0])
                            break
                        k+=1
                        if k > 20:
                            pixel_size = 0.04
                            break
                        
                for plant in plants:
                    results = load_path(plant, '*')
                    
                    if len(results) == 0:
                        continue
                    else:
                        results = results[-1]

                    # Saves QR in each metadata
                    metadata = json.load(open(results + '/metadata.json', 'r'))
                    metadata['pixel_size'] = pixel_size
                    json.dump(metadata, open(results + '/metadata.json', 'w'))

                    if conf['Limit'] != 0:
                        N_exp = conf['Limit']
                    else:
                        N_exp = None
                    
                    pfile = results + '/Results_raw.csv'

                    dataWork(conf, pfile, results, N_exp = N_exp)

                    file = os.path.join(results,'PostProcess_Hour.csv')
                    data = pd.read_csv(file)

                    name = variety.split('/')[-1] + '_' + rpi.split('/')[-1] + '_' + cam.split('/')[-1] + '_' + plant.split('/')[-1]
                    plot_individual_plant(results, data, name)

                    getAngles(conf, results)

    print('Post processing finished.')