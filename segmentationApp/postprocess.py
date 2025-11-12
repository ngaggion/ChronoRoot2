"""
Unified postprocessing script for nnUNet segmentation results.
Supports both arabidopsis and tomato postprocessing methods.
Works with single-fold and multi-fold scenarios.
"""

import argparse
import pathlib
import re
import os
import numpy as np
import cv2

def natural_key(string_):
    """Natural sorting for file names with numbers"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def loadPath(search_path, ext='*.*'):
    """Load and sort file paths"""
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key=natural_key)
    return all_files

def generate_colormap(num_classes):
    """Generate qualitative colormap for segmentation visualization"""
    general_colormap = np.array([
        [0, 0, 0],       # Class 0: background (black)
        [255, 0, 0],     # Class 1: red
        [0, 255, 0],     # Class 2: green
        [0, 0, 255],     # Class 3: blue
        [255, 255, 0],   # Class 4: yellow
        [0, 255, 255],   # Class 5: cyan
        [255, 0, 255],   # Class 6: magenta
        [255, 255, 255], # Class 7: white
        [128, 128, 128], # Class 8: gray
        [255, 165, 0],   # Class 9: orange
        [165, 42, 42],   # Class 10: brown
        [255, 192, 203], # Class 11: pink
        [128, 0, 128]    # Class 12: purple
    ])
    
    return general_colormap[:num_classes]

def find_segmentation_folders(seg_path):
    """Find all fold directories in segmentation path"""
    if not os.path.exists(seg_path):
        return []
    
    # Look for Fold_X directories or any directory with segmentation results
    fold_dirs = []
    for item in os.listdir(seg_path):
        item_path = os.path.join(seg_path, item)
        if os.path.isdir(item_path):
            # Check if it contains PNG files (segmentation results)
            if item.startswith("Fold_") or item.startswith("fold_"):
                png_files = list(pathlib.Path(item_path).glob("*.png"))
                if png_files:
                    fold_dirs.append(item)
    
    # If no fold directories found, check if PNG files are directly in seg_path
    if not fold_dirs:
        png_files = list(pathlib.Path(seg_path).glob("*.png"))
        if png_files:
            # Treat the seg_path itself as a fold
            return ["."]
    
    return sorted(fold_dirs)

def postprocess(path, method="arabidopsis", alpha=None, num_classes=7, seg_path=None):
    """
    Unified postprocessing function for both arabidopsis and tomato methods.
    
    Args:
        path: Path to folder containing original images
        method: "arabidopsis" or "tomato" postprocessing method
        alpha: Temporal accumulation weight (default: 0.85 for arabidopsis, 0.99 for tomato)
        num_classes: Number of segmentation classes
        seg_path: Path to segmentation folder (default: path/Segmentation)
    """
    
    # Set default alpha based on method
    if alpha is None:
        alpha = 0.85 if method == "arabidopsis" else 0.60
    
    # Get image list from original folder
    images = loadPath(path, ext="*.png")
    if not images:
        print(f"No PNG images found in {path}")
        return
    
    # Set segmentation path
    if seg_path is None:
        seg_path = os.path.join(path, "Segmentation")
    else:
        seg_path = os.path.join(path, seg_path)
    
    # Find segmentation folders (folds)
    folds = find_segmentation_folders(seg_path)
    
    if not folds:
        print(f"No segmentation folders found in {seg_path}")
        return
    
    print(f"Found {len(folds)} fold(s): {folds}")
    print(f"Using {method} postprocessing with alpha = {alpha}")
    print(f"Processing {len(images)} images with {num_classes} classes")
    
    # Initialize accumulator
    img = cv2.imread(images[0], 0)
    accum = np.zeros((num_classes, *img.shape[:2]), dtype=np.float32)
    
    # Create output directories
    output_path = os.path.join(seg_path, "Ensemble")
    os.makedirs(output_path, exist_ok=True)
    color_path = os.path.join(seg_path, "Ensemble_color")
    os.makedirs(color_path, exist_ok=True)
    
    colormap = generate_colormap(num_classes)
    
    # Process each image
    for image_idx, image_path in enumerate(images):
        image_name = os.path.basename(image_path)
        
        # Ensemble multiple folds if available
        if len(folds) > 1:
            ensemble = np.zeros((num_classes, *img.shape[:2]), dtype=np.float32)
            valid_folds = 0
            
            for fold in folds:
                fold_path = seg_path if fold == "." else os.path.join(seg_path, fold)
                seg_file = os.path.join(fold_path, image_name)
                
                if os.path.exists(seg_file):
                    seg = cv2.imread(seg_file, 0)
                    # Convert to one-hot encoding
                    for i in range(num_classes):
                        ensemble[i] += (seg == i)
                    valid_folds += 1
            
            if valid_folds > 0:
                ensemble /= valid_folds  # Normalize
            else:
                print(f"Warning: No segmentation found for {image_name}")
                continue
                
        else:
            # Single fold case
            fold_path = seg_path if folds[0] == "." else os.path.join(seg_path, folds[0])
            seg_file = os.path.join(fold_path, image_name)
            
            if not os.path.exists(seg_file):
                print(f"Warning: Segmentation not found for {image_name}")
                continue
                
            seg = cv2.imread(seg_file, 0)
            ensemble = np.zeros((num_classes, *seg.shape[:2]), dtype=np.float32)
            for i in range(num_classes):
                ensemble[i] = (seg == i)
        
        # Apply method-specific postprocessing
        if method == "tomato":
            # Tomato method: dilate non-background classes
            dilated_ensemble = np.zeros_like(ensemble)
            for i in range(1, num_classes):
                dilated_ensemble[i] = cv2.dilate(
                    ensemble[i], 
                    np.ones((5, 5), np.uint8), 
                    iterations=2
                )
            dilated_ensemble[0] = ensemble[0]
            
            # Temporal postprocessing
            accum[0] = ensemble[0]
            accum[1:] = float(alpha) * accum[1:] + dilated_ensemble[1:]
            segmentation_class = np.argmax(accum, axis=0)
            
            # Binary mask with morphological closing
            binary_seg = (np.argmax(ensemble, axis=0) > 0).astype('uint8')
            kernel = np.ones((5, 5), np.uint8)
            binary_seg = cv2.dilate(binary_seg, kernel, iterations=1)
            binary_seg = cv2.erode(binary_seg, kernel, iterations=1)
            
            segmentation = binary_seg * segmentation_class
            
        else:  # arabidopsis method (default)
            # Temporal postprocessing
            accum[0] = ensemble[0]
            
            # Accumulate 
            accum[1:] = float(alpha) * accum[1:] + ensemble[1:]
            
            segmentation = np.argmax(accum, axis=0)
        
        # Save segmentation as PNG with class values
        output_file = os.path.join(output_path, f"{os.path.splitext(image_name)[0]}.png")
        cv2.imwrite(output_file, segmentation.astype(np.uint8))
        
        # Save color visualization
        color_segmentation = colormap[segmentation]
        color_segmentation = color_segmentation[:, :, :3].astype(np.uint8)
        color_file = os.path.join(color_path, f"{os.path.splitext(image_name)[0]}.png")
        
        color_segmentation = cv2.cvtColor(color_segmentation, cv2.COLOR_RGB2BGR)
        cv2.imwrite(color_file, color_segmentation)

        if (image_idx + 1) % 10 == 0:
            print(f"  Processed {image_idx + 1}/{len(images)} images...")
    
    print(f"Postprocessing complete! Results saved to:")
    print(f"  - Grayscale: {output_path}")
    print(f"  - Color: {color_path}")

def main():
    parser = argparse.ArgumentParser(description="Postprocess nnUNet segmentation results")
    parser.add_argument("path", help="Path to folder containing original images")
    parser.add_argument("--method", default="arabidopsis", choices=["arabidopsis", "tomato"],
                       help="Postprocessing method to use")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Temporal accumulation weight (default: 0.85 for arabidopsis, 0.60 for tomato)")
    parser.add_argument("--num_classes", type=int, default=7,
                       help="Number of segmentation classes")
    parser.add_argument("--seg_path", default="Segmentation",
                       help="Relative path to segmentation folder")
    
    args = parser.parse_args()
    
    postprocess(
        path=args.path,
        method=args.method,
        alpha=args.alpha,
        num_classes=args.num_classes,
        seg_path=args.seg_path
    )

if __name__ == "__main__":
    main()