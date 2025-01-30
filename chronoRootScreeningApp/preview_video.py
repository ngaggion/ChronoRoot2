#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import sys
from typing import List, Tuple

def loadPath(path: str, ext: str = "*") -> List[str]:
    """
    Helper function to load paths with specific extensions.
    Matches the functionality expected by getImages()
    """
    import glob
    if not os.path.exists(path):
        return []
    return sorted(glob.glob(os.path.join(path, ext)))

def getImages(video_dir: str) -> Tuple[List[str], List[str]]:
    """
    Returns lists of image paths and segmentation paths
    """
    # Check if the directory exists and contains png files
    images = loadPath(video_dir, ext="*.png")
    
    # Look for segmentation files
    seg_path = os.path.join(video_dir, 'Segmentation', 'Ensemble')
    if not os.path.exists(seg_path):
        seg_path = os.path.join(video_dir, 'Seg')
    
    seg_files = loadPath(seg_path, ext="*.png")
    
    # Ensure we have matching numbers of files
    n = min(len(images), len(seg_files))
    images = images[:n]
    seg_files = seg_files[:n]
    
    return images, seg_files

def preview_sequence(video_dir: str, time_delta: float = 60.0):
    """
    Preview the sequence of images with the given time delta between frames
    Args:
        video_dir: Directory containing the image sequence
        time_delta: Time in seconds between frames
    """
    # Get image and segmentation paths
    images, seg_files = getImages(video_dir)
    if not images:
        print(f"No images found in directory: {video_dir}")
        return
    
    N = len(images)
    
    # Create time vectors
    time = np.arange(0, N * time_delta, time_delta)  # in minutes
    minutes = (time % 60).astype('int')
    hours = ((time / 60) % 24).astype('int')
    days = (time // 1440).astype('int')
    
    # Create window and trackbar
    window_name = 'Preview Image (show segmentation with "s" key)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)
    cv2.createTrackbar('Frame', window_name, 0, N-1, lambda x: None)
    
    use_seg = False
    while True:
        i = cv2.getTrackbarPos('Frame', window_name)
        
        # Read and process image
        img = cv2.imread(images[i])

        if img is None:
            print(f"Error reading image: {images[i]}")
            continue
            
        if use_seg and i < len(seg_files):
            # Read segmentation mask and original image
            seg = cv2.imread(seg_files[i], cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            if seg is not None:
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
        
        # Add timestamp overlay
        cv2.putText(img, f"Day: {days[i]:2d}", (5, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        cv2.putText(img, f"Time: {hours[i]:02d}:{minutes[i]:02d}", (5, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        
        # Display image
        cv2.imshow(window_name, img)
        
        # Handle key events
        key = cv2.waitKey(1)
        if key == 27 or key == ord('c'):  # ESC or 'c' to quit
            break
        elif key == ord('s'):
            use_seg = not use_seg
            
        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Preview video sequence from images')
    parser.add_argument('--video-dir', required=True,
                      help='Directory containing the image sequence')
    parser.add_argument('--time-delta', type=float, default=60.0,
                      help='Time in seconds between frames (default: 60)')
    
    args = parser.parse_args()
    
    try:
        preview_sequence(args.video_dir, args.time_delta)
    except Exception as e:
        print(f"Error during preview: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()