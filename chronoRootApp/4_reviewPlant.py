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

import argparse
import json 
import cv2
import os
from analysis.utils.fileUtilities import loadPath
import json
import numpy as np

def review(images, segFiles, bbox, conf):
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
    
    cv2.namedWindow('Review plant segmentation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Review plant segmentation', 400, 800)
    cv2.createTrackbar('Overlay segmentation with "s"', 'Review plant segmentation', 0, n-1, lambda x: None)

    useSeg = False

    while True:
        i = cv2.getTrackbarPos('Overlay segmentation with "s"', 'Review plant segmentation')
        img = cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        if useSeg:
            image_overlay = cv2.imread(segFiles[i])
            img = cv2.add(img, image_overlay)

        #Draw the day, hour and minute at the bottom left corner
        cv2.putText(img, "Day: %2d" % days[i], (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
        cv2.putText(img, "Time: %2d:%2d" % (hours[i], minutes[i]), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)

    	# display the image and wait for a keypress
        cv2.imshow("Review plant segmentation", img)

        key = cv2.waitKey(1)

        if key == 27:
            break
        
        # close by clicking the X button
        if cv2.getWindowProperty('Review plant segmentation', cv2.WND_PROP_VISIBLE) < 1:
            return None

        if key == ord('c'):
            break
        elif key == ord('s'):
            useSeg = not useSeg
        elif key == 27:
            return None
        elif key == 13:
            break

        
    
    cv2.destroyAllWindows()
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture')
    parser.add_argument('--path', type=str, default=None, help='Path to the plant analysis folder')

    # READ THE JSON CONFIGURATION FILE

    args = parser.parse_args()

    if args.path is None:
        raise Exception("Error: path is not defined")
    
    json_path = os.path.join(args.path, 'metadata.json')
    conf = json.load(open(json_path))

    bbox = conf['bounding box']
    imagePath = conf['ImagePath']
    seg = os.path.join(args.path, "Images", "SegMulti")

    images = loadPath(imagePath, ext = "*.png")
    segs = loadPath(seg, ext = "*.png")
    
    review(images, segs, bbox, conf)

