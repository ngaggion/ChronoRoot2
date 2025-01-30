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
import os
import cv2
import numpy as np

from analysis.utils.fileUtilities import getImages

def preview(conf):
    """
    Function to preview the sequence of images
    Uses a openCV window to show the sequence of images
    The sequence of images is a set of png files, obtained by getImages function
    Includes a scrollbar to advance in the sequence of images
    Ensure that the window is in focus to use the scrollbar
    Ensure that the image is not too big to fit in the screen
    Allows to close the window with the 'c' key
    Allows to overlay the segmentation in red, over the image, with the 's' key 
    Allows to continue scrolling with segmentation overlayed
    """

    images, segFiles = getImages(conf)
    N = len(images)

    n = len(images)
    n2 = len(segFiles)
    
    n = min(n, n2)
    images = images[:n]
    segFiles = segFiles[:n]

    # Timestep in minutes
    timeStep = conf['timeStep'] # minutes per frame
    # Create a vector to know at which time the frame was taken
    time = np.arange(0, n*timeStep, timeStep) # in minutes
    
    minutes = (time % 60).astype('int') # in minutes
    hours = ((time/60) % 24).astype('int') # in hours
    days = (time // 1440).astype('int') # in days

    cv2.namedWindow('Preview Image (show segmentation with "s" key)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Preview Image (show segmentation with "s" key)', 800,800)
    cv2.createTrackbar('Preview Image', 'Preview Image (show segmentation with "s" key)', 0, N-1, lambda x: None)

    useSeg = False

    while True:
        i = cv2.getTrackbarPos('Preview Image', 'Preview Image (show segmentation with "s" key)')
        img = cv2.imread(images[i])

        if useSeg:
            seg = cv2.imread(segFiles[i])
            img[seg>0] = 255

        #Draw the day, hour and minute at the top left corner
        cv2.putText(img, "Day: %2d" % days[i], (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        cv2.putText(img, "Time: %2d:%2d" % (hours[i], minutes[i]), (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

        cv2.imshow('Preview Image (show segmentation with "s" key)', img)
        key = cv2.waitKey(1)

        if key == 27:
            break
        # close by clicking the X button
        if cv2.getWindowProperty('Preview Image (show segmentation with "s" key)', cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == ord('c'):
            break
        elif key == ord('s'):
            useSeg = not useSeg

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture')
    parser.add_argument('--preview', action='store_true', default=False, help='Previews the sequence of images')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file (default: config.json)')
    parser.add_argument('--get_bbox', action='store_true', default=False, help='Only gets the bounding box of the root system')
    parser.add_argument('--rerun', action='store_true', default=False, help='Reruns the analysis, even if the results already exist')

    args = parser.parse_args()
        
    conf = json.load(open(args.config))

    if not args.rerun:
        if "rpi" not in str(conf['rpi']):
            rpi = "rpi" + str(conf['rpi'])
        else:
            rpi = str(conf['rpi'])
        conf['rpi'] = rpi
    
    if not args.preview:
        conf['fileKey'] = conf['identifier']
        conf['sequenceLabel'] = conf['identifier'] + '/' + conf['rpi'] + '/' + str(conf['cam']) + '/' + str(conf['plant'])
        conf['Plant'] = 'Arabidopsis thaliana'

        if args.get_bbox:
            plantAnalysis(conf, False, True)
        # elif exists key 'bounding box'
        elif 'bounding_box' in conf and not args.rerun:
            plantAnalysis(conf, True, False)
        else:
            plantAnalysis(conf, False, False)
    else:
        preview(conf)