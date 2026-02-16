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

import pathlib
import re
import os 

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def loadPath(search_path, ext = '*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key = natural_key)
    
    return all_files

def convertToPathSafe(name):
    name = name.replace('.', '_dot_')
    name = name.replace('/', '_slash_')
    name = name.replace('\\', '_backslash_')
    return name
def convertFromPathSafe(name):
    name = name.replace('_dot_', '.')
    name = name.replace('_slash_', '/')
    name = name.replace('_backslash_', '\\')
    return name

def createSaveFolder(conf):
    # Create the folder for the general results
    analysis = os.path.join(conf['MainFolder'], 'Analysis')
    os.makedirs(analysis, exist_ok=True)
    
    # Create the folder for the identifier
    identifier = convertToPathSafe(conf['Experiment'])
    
    id_path = os.path.join(analysis, identifier)
    os.makedirs(id_path, exist_ok=True)

    # Create the folder for the rpi
    rpi = str(conf['rpi'])
    rpi_path = os.path.join(id_path, rpi)
    os.makedirs(rpi_path, exist_ok=True)
    
    # Create the folder for the cam
    cam = "cam_" + str(str(conf['cam']))
    cam_path = os.path.join(rpi_path, cam)
    os.makedirs(cam_path, exist_ok=True)
    
    # Create the folder for the plant
    plant = "plant_" + str(str(conf['plant']))
    plant_path = os.path.join(cam_path, plant)
    os.makedirs(plant_path, exist_ok=True)
    
    # Create the folder for the results
    for j in range(0, 50):
        result_path = os.path.join(plant_path, 'Results_%s'%j)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            break
    
    # create folders for outputs
    graphsPath = os.path.join(result_path, 'Graphs')
    os.makedirs(graphsPath, exist_ok=True)
    
    imagePath = os.path.join(result_path, 'Images')
    os.makedirs(imagePath, exist_ok=True)
    
    outSegPath = os.path.join(imagePath, 'Seg')
    os.makedirs(outSegPath, exist_ok=True)
        
    multiPath = os.path.join(imagePath, 'SegMulti')
    os.makedirs(multiPath, exist_ok=True)
    
    if conf['saveImages']:
        inPath = os.path.join(imagePath, 'Input')
        os.makedirs(inPath, exist_ok=True)

    rsmlPath = os.path.join(result_path, 'RSML')
    os.makedirs(rsmlPath, exist_ok=True)
    
    # creates a dictionary with all the paths
    paths = {'analysis': analysis, 'result': result_path, 'graphs': graphsPath, 'images': imagePath, 'rsml': rsmlPath}

    return paths

def getImages(conf):
    # Get the list of images    
    images = loadPath(conf['Images'], ext = "*.png") 

    conf['ImagePath'] = conf['Images']
                
    # Check if there is no images, then look for a file called "segmentation_metadata.json"
    if len(images) == 0:
        metadata_path = os.path.join(conf['Images'], 'Segmentation', 'segmentation_metadata.json')
        #print("No images found in the specified folder. Looking for segmentation metadata file at: ", metadata_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            images_path = metadata.get('input_path', None)
            if images_path and os.path.exists(images_path):
                images = loadPath(images_path, ext="*.png")
                conf['ImagePath'] = images_path
    
    # Get the list of segmentation images
    SegPath = os.path.join(conf['Images'], 'Segmentation', 'Ensemble')
    if not os.path.exists(SegPath):
        SegPath = os.path.join(conf['Images'], 'Seg')
    
    segFiles = loadPath(SegPath, ext = "*.png") 

    # Save configuration
    
    conf['SegPath'] = SegPath
        
    return images, segFiles

import json

def saveMetadata(bbox, seed, conf):
    metadata = {}
    metadata['bounding box'] = bbox
    metadata['seed'] = seed

    # combine metadata and conf
    metadata.update(conf)

    metapath = os.path.join(metadata['folders']['result'], 'metadata.json')

    with open(metapath, 'w') as fp:
        json.dump(metadata, fp)

    metapath = os.path.join(metadata['MainFolder'], 'lastAnalysis.json')
    with open(metapath, 'w') as fp:
        json.dump(metadata, fp)

    return metadata