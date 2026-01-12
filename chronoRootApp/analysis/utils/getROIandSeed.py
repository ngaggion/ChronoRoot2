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
            f"Select Plant ROI\n"
            "1. Click and drag to select region\n"
            "2. Press ENTER TWICE to confirm selection\n"
            "3. Press 'r' to redo selection\n"
            "4. Press 'q' to quit analysis"
        )
        window_name = "Select Plant ROI"
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
    n = min(len(images), len(segFiles))
    images, segFiles = images[:n], segFiles[:n]

    # Time calculations
    timeStep = conf['timeStep']
    time_mins = np.arange(0, n * timeStep, timeStep)
    minutes = (time_mins % 60).astype('int')
    hours = ((time_mins / 60) % 24).astype('int')
    days = (time_mins // 1440).astype('int')

    global pos
    window_name = 'Select Root Origin'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 900)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.createTrackbar('Overlap segmentation with "s"', window_name, 0, n - 1, lambda x: None)

    useSeg = False
    
    while True:
        i = cv2.getTrackbarPos('Frame', window_name)
        img = cv2.imread(images[i])[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        h, w = img.shape[:2]

        if useSeg:
            seg = cv2.imread(segFiles[i], 0)[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 4: (0, 255, 255)}
            for val, color in colors.items():
                img[seg == val] = color
            img[seg >= 5] = (255, 0, 255)

        # Top Overlay: Large Day and Time
        cv2.putText(img, "Day: %2d" % days[i], (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
        cv2.putText(img, "Time: %2d:%2d" % (hours[i], minutes[i]), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)

        if pos is not None:
            # Precision marker and divider
            cv2.drawMarker(img, tuple(pos), (0, 255, 0), cv2.MARKER_CROSS, 25, 5)
            cv2.line(img, (0, pos[1]), (w, pos[1]), (0, 255, 255), 5)
            cv2.putText(img, "Root Start", (w-250, pos[1]+35), 0, 1.0, (255, 255, 0), 3)

        # Bottom Overlay: Large Instructions in Rows
        row1 = "ENTER: CONFIRM"
        row2 = "S: TOGGLE SEGMENTATION"
        row3 = "ESC/C/Q: CANCEL"
        
        cv2.putText(img, row1, (20, h-130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(img, row2, (20, h-85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(img, row3, (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF

        # Return None on cancel keys
        if key in [ord('c'), ord('q'), 27]:
            cv2.destroyAllWindows()
            return None
        
        # Return position on Enter
        elif key in [13, 10]:
            if pos is not None:
                break
        
        elif key == ord('s'):
            useSeg = not useSeg
        
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return None

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
    
    if seed is None:
        return None, None
    
    return bbox, seed