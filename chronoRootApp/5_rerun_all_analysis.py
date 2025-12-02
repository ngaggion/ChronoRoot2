"""
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from analysis.plantAnalysis import plantAnalysis
import argparse
import json
from analysis.utils.fileUtilities import loadPath
import os
from multiprocessing import Pool


def analyze_experiment(exp):
    # Load the configuration for each experiment
    conf = json.load(open(exp))

    # New main folder for rerun analysis
    conf['MainFolder'] = "/DATA/tomatest_2"
    os.makedirs(conf['MainFolder'], exist_ok=True)
    conf['fileKey'] = conf['identifier']
    conf['sequenceLabel'] = str(conf['identifier']) + '/' + str(conf['rpi']) + '/' + str(conf['cam']) + '/' + str(conf['plant'])
    conf['Plant'] = 'Tomato'

    conf["processingLimit"] = 6
    conf['timeStep'] = 15
    conf['Limit'] = int(conf['processingLimit'] * 24 * 60 / conf['timeStep'])

    # Perform the analysis
    plantAnalysis(conf, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file (default: config.json)')

    args = parser.parse_args()

    # Old main folder
    mainFolder = "/DATA/tomatest"
    analysis = os.path.join(mainFolder, 'Analysis')
    experiments = loadPath(analysis, '*/*/*/*/*/metadata.json')

    # Create a Pool with 4 processes
    with Pool(4) as pool:
        pool.map(analyze_experiment, experiments)
