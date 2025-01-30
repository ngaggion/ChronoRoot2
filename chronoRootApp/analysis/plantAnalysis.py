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

from .utils.fileUtilities import createSaveFolder, getImages, saveMetadata
from .utils.getROIandSeed import getROIandSeed
from .imageUtils.seg import getCleanSeg, getCleanSke
from .imageUtils.plot import saveImages
from .graphUtils.save import saveGraph, saveProps
from .graphUtils.graph import createGraph
from .graphUtils.graphTrim import trimGraph
from .graphUtils.graphTrack import graphInit, matchGraphs
from .rsmlUtils.rsml import createTree, saveRSML

import os 
import csv
import datetime

def getImgName(image, conf):
    return image.replace(conf['ImagePath'],'').replace('/','')

def plantAnalysis(conf, replicate = False, get_bbox = False):
    # Get the images
    images, segFiles = getImages(conf)
    
    if not replicate:
        # Get the bounding box and the seed
        bbox, seed = getROIandSeed(conf, images, segFiles)
        if seed is None:
            return
        originalSeed = seed.copy()

    else:
        bbox = conf['bounding box']
        seed = conf['seed']
        originalSeed = seed.copy()

    if get_bbox:
        return
    
    # Create the folder for the results
    folders = createSaveFolder(conf)

    # Save the metadata into a json, for replicability
    conf['folders'] = folders
    conf = saveMetadata(bbox, seed, conf)

    N = len(images)

    print('Number of frames:', N)

    pfile = os.path.join(folders['result'], "Results_raw.csv") # For CSV Saver

    # Creates a placeholder for logs
    logs = []
    # Error log per time frame, will be 0 if no error, 1 if error
    error_log = []

    with open(pfile, 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        head = ['FileName', 'Frame', 'MainRootLength', 'LateralRootsLength', 'NumberOfLateralRoots', 'TotalLength']
        csv_writer.writerow(head)

        for i in range(0, N):
            print('Frame', i+1, 'of', N)
            seg, segFound = getCleanSeg(segFiles[i], bbox, originalSeed, originalSeed)
            
            if segFound:
                ske, bnodes, enodes, flag = getCleanSke(seg)

                if flag:
                    try: 
                        graph, seed, ske2 = createGraph(ske.copy(), seed, enodes, bnodes)
                        graph, ske, ske2 = trimGraph(graph, ske, ske2)
                        graph = graphInit(graph)
                        rsmlTree, numberLR = createTree(conf, i, images, graph, ske, ske2)
                        break
                    except:
                        pass
            
            image_name = getImgName(images[i], conf)
            saveProps(image_name, i, False, csv_writer, 0)
            saveImages(conf, images, i, seg, None, None)
            error_log.append(0)
        
        if i == N-1:
            print('No segmentation found')
            # save log 
            with open(os.path.join(folders['result'], "log.txt"), 'w+') as log_file:
                log_file.write('No segmentation found \n')
            return
        
        start = i 

        print('Growth Begin')
        # saves growth begin in log, number of frame
        logs.append(['Frame %s Growth Begin \n'%start])

        image_name = getImgName(images[i], conf)
        saveImages(conf, images, i, seg, graph, ske2)
        saveGraph(graph, conf, image_name)
        saveRSML(rsmlTree, conf, image_name)       
        saveProps(image_name, i, graph, csv_writer, numberLR)

        error_count = 0
        for i in range(start + 1, N):
            print('TimeStep', i+1, 'of', N)

            errorSeg = False
            errorGraph = False
            errorRSML = False

            newSeg, segFound = getCleanSeg(segFiles[i], bbox, seed, originalSeed)

            if segFound:
                newSke, bnodes, enodes, flag = getCleanSke(newSeg)
                                
                if flag:
                    errorSeg = False

                    try:
                        newGraph, seed, newSke2 = createGraph(newSke.copy(), seed, enodes, bnodes)
                        newGraph, newSke_, newSke2 = trimGraph(newGraph, newSke.copy(), newSke2)

                    except:
                        print("Error in graph creation")
                        logs.append(['Frame %s Error in graph creation \n'%i]) 
                        error_count += 1
                        errorGraph = True
                        errorRSML = True
                    
                    if not errorGraph:
                        try:
                            newGraph = matchGraphs(graph, newGraph)
                        except:
                            if i - start < 50:
                                print("Reinitializing Graph")
                                newGraph = graphInit(graph)
                            else:
                                print("Error in tracking")
                                logs.append(['Frame %s Error in tracking \n'%i])   
                                error_count += 1
                                errorGraph = True
                                errorRSML = True
                    
                    if not errorGraph:
                        try:
                            rsmlTreeNew, numberLR = createTree(conf, i, images, newGraph, newSke.copy(), newSke2.copy())
                        except:
                            print("Error in RSML")
                            # saves error in log, number of frame
                            logs.append(['Frame %s Error in RSML \n'%i])
                            error_count += 1
                            errorRSML = True
                else:
                    if i - start > 50:
                        error_count += 1
                    print("Error in skeletonization")
                    logs.append(['Frame %s Error in skeletonization \n'%i])
                    errorSeg = True 
                    errorGraph = True
                    errorRSML = True

            else:
                print("Error in segmentation")
                # saves error in log, number of frame
                logs.append(['Frame %s Error in segmentation \n'%i])
                if i - start > 50:
                    error_count += 1
                errorSeg = True
                errorGraph = True
                errorRSML = True
            
            if not errorSeg:
                seg=newSeg
                ske=newSke
                ske2=newSke2
            
            if not errorGraph:
                graph=newGraph
            
            if not errorRSML:
                rsmlTree = rsmlTreeNew

            if not errorSeg or not errorGraph or not errorRSML:
                error_log.append(0)
            else:
                error_log.append(1)

            image_name = getImgName(images[i], conf)
            saveImages(conf, images, i, seg, graph, ske2)
            saveGraph(graph, conf, image_name)
            saveRSML(rsmlTree, conf, image_name)       
            saveProps(image_name, i, graph, csv_writer, numberLR)

        # Saves a log file with the number of errors
        # If the number of errors is too high, the analysis is not reliable
        # and the user should check the results
        with open(os.path.join(folders['result'], "log.txt"), 'w+') as log_file:
            log_file.write("Finish time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

            # counts errors over total of growth steps
            error_rate = round(error_count / (N - start), 3)
            log_file.write("Error rate: " + str(error_rate)+ "\n")
            # number of total errors
            log_file.write("Total errors: " + str(error_count)+ "\n")
            # number of total growth steps
            log_file.write("Total growth steps: " + str(N - start)+ "\n")
            # leaves empty line
            log_file.write("\n")
            # saves the log of errors
            for log in logs:
                log_file.write(log[0])

        # Save error log as text array
        with open(os.path.join(folders['result'], "error_log.txt"), 'w+') as log_file:
            for error in error_log:
                log_file.write(str(error) + "\n")

        print('Growth End')
            
    return