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

from analysis.plantAnalysis import plantAnalysis
import argparse
import json 
from analysis.utils.fileUtilities import loadPath
import os
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file (default: config.json)')

    args = parser.parse_args()
    
    # Old main folder
    mainFolder = "/home/ngaggion/DATA/Raices/AndanaBamboo"
    analysis = os.path.join(mainFolder, 'Analysis')
    experiments = loadPath(analysis, '*/*/*/*/*/metadata.json')

    for exp in experiments:
        conf = json.load(open(exp))

        # New main folder for rerun analysis
        conf['MainFolder'] = "/home/ngaggion/DATA/Raices/Bamboo_3"

        if "rpi" not in str(conf['rpi']):
            rpi = "rpi" + str(conf['rpi'])
        else:
            rpi = str(conf['rpi'])

        conf['fileKey'] = conf['identifier']
        conf['sequenceLabel'] = conf['identifier'] + '/' + conf['rpi'] + '/' + str(conf['cam']) + '/' + str(conf['plant'])
        conf['Plant'] = 'Arabidopsis thaliana'
        
        conf["processingLimit"] = 10
        conf['timeStep'] = 15
        conf['Limit'] = int(conf['processingLimit'] * 24 * 60 / conf['timeStep'])

        plantAnalysis(conf, True, False)
