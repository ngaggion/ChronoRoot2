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

import cv2 
import numpy as np

def selectROI(image):
        instructions = (
            f"Select ROI\n"
            "1. Click and drag to select region\n"
            "2. Press ENTER TWICE to confirm selection\n"
            "3. Press 'r' to redo selection\n"
            "4. Press 'q' to quit analysis"
        )
        window_name = "Select ROI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 1000)
        
        while True:
            # Create fresh copy of image for display
            img_copy = image.copy()
            
            # Add instructions to the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.5
            thickness = 8
            color = (255, 255, 255)
            
            y0 = 60
            for i, line in enumerate(instructions.split('\n')):
                y = y0 + i * 70
                cv2.putText(img_copy, line, (10, y), font, font_scale, color, thickness)
            
            # Draw ROI
            roi = cv2.selectROI(window_name, img_copy, fromCenter=False, showCrosshair=True)
            
            if roi[2] == 0 or roi[3] == 0:  # If ROI has no area
                key = cv2.waitKey(1)
                if key == ord('q'):  # Check if user pressed 'q' to quit
                    cv2.destroyWindow(window_name)
                    return None
                print("Invalid selection, please try again or press 'q' to quit")
                continue
            
            # Draw the selection for confirmation
            cv2.rectangle(img_copy, (roi[0], roi[1]), 
                         (roi[0] + roi[2], roi[1] + roi[3]), (255, 255, 255), 5)
            cv2.imshow(window_name, img_copy)
            
            print(f"\nSelection made.")
            print("Press 'r' to redo selection or 'q' to quit analysis")
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyWindow(window_name)
                return None
            elif key == ord('r'):
                continue
            else:
                cv2.destroyWindow(window_name)
                return roi
            
pos = None

def mouse_callback(event, x, y, flags, param):
    global pos
    if event == cv2.EVENT_LBUTTONDOWN:
        pos = [x, y]
        
def selectSeed(images, segFiles, bbox, conf):
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

    global pos
    
    cv2.namedWindow('Select seed with cursor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select seed with cursor', 400, 800)
    cv2.setMouseCallback('Select seed with cursor', mouse_callback) #Mouse callback
    cv2.createTrackbar('Overlap segmentation with "s"', 'Select seed with cursor', 0, n-1, lambda x: None)

    useSeg = False

    while True:
        i = cv2.getTrackbarPos('Overlap segmentation with "s"', 'Select seed with cursor')
        img = cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]

        if useSeg:
            seg = cv2.imread(segFiles[i], 0)[bbox[0]:bbox[1],bbox[2]:bbox[3]]

            # Convert grayscale image to color if it's not already
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Define colors for each segment (B,G,R format)
            colors = {
                1: (0, 0, 255),     # Red
                2: (0, 255, 0),     # Green
                3: (255, 0, 0),     # Blue
                4: (0, 255, 255),   # Yellow
            }
            
            # Apply colors for values 1-4
            for val, color in colors.items():
                mask = (seg == val)
                img[mask] = color
            
            # Handle values 5 and above with purple
            high_vals_mask = (seg >= 5)
            img[high_vals_mask] = (255, 0, 255)  # Purple for values 5+

        if pos is not None:
            cv2.circle(img, tuple(pos), 8, [255,0,0], -1)
        
        #Draw the day, hour and minute at the bottom left corner
        cv2.putText(img, "Day: %2d" % days[i], (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
        cv2.putText(img, "Time: %2d:%2d" % (hours[i], minutes[i]), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)

    	# display the image and wait for a keypress
        cv2.imshow("Select seed with cursor", img)

        key = cv2.waitKey(1)

        if key == 27:
            break
        
        # close by clicking the X button
        if cv2.getWindowProperty('Select seed with cursor', cv2.WND_PROP_VISIBLE) < 1:
            return None

        if key == ord('c'):
            break
        if key == ord('q'):
            break
        elif key == ord('s'):
            useSeg = not useSeg
        elif key == 27:
            return None
        elif key == 13:
            break
    
    cv2.destroyAllWindows()
    return pos


def getROIandSeed(conf, images, segFiles):
    last_image = cv2.imread(images[-1])
    try:
        r = selectROI(last_image)
    except:
        return None, None
    
    if r is None:
        return None, None
    elif r[0] == 0 and r[1] == 0 and r[2] == 0 and r[3] == 0:
        return None, None

    bbox = [int(r[1]),int(r[1]+r[3]), int(r[0]),int(r[0]+r[2])]    
    seed = selectSeed(images, segFiles, bbox, conf)
    
    return bbox, seed