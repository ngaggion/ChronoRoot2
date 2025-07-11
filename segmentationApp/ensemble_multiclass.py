import argparse
import pathlib
import re
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def loadPath(search_path, ext='*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key=natural_key)
    return all_files

def generate_colormap(num_classes):
    # We need a qualitative colormap for the segmentation
    
    # Class 0 is background: black
    # Class 1 is the first class: red
    # Class 2 is the second class: green
    # Class 3 is the third class: blue
    # Class 4 is the fourth class: yellow
    # Class 5 is the fifth class: cyan
    # Class 6 is the sixth class: magenta
    # Class 7 is the seventh class: white
    # Class 8 is the eighth class: gray
    # Class 9 is the ninth class: orange
    # Class 10 is the tenth class: brown
    # Class 11 is the eleventh class: pink
    # Class 12 is the twelfth class: purple
    # Using a colormap with upto 12 classes

    general_colormap = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                                 [255, 255, 0], [0, 255, 255], [255, 0, 255],
                                 [255, 255, 255], [128, 128, 128], [255, 165, 0],
                                 [165, 42, 42], [255, 192, 203], [128, 0, 128]])
    
    colormap = general_colormap[:num_classes]
    
    return colormap

def ensemble(path, alpha, num_classes, seg_path=None):
    images = loadPath(path, ext="*.png")
    img = io.imread(images[0], as_gray=True)
    accum = np.zeros((num_classes, *img.shape[:2]), dtype=np.float32)
    if seg_path is None:
        seg_path = os.path.join(path, "Segmentation")
    else:
        seg_path = os.path.join(path, seg_path)
    
    # Minimal fix: explicitly check for directories
    all_paths = loadPath(seg_path, ext="*")
    folds = [os.path.basename(fold) for fold in all_paths if os.path.isdir(fold) and not os.path.basename(fold).startswith("Ensemble")]
    
    print(f"Ensembling {len(folds)} folds.")
    print(f"Postprocessing with alpha = {alpha}.")

    output_path = os.path.join(seg_path, "Ensemble")
    os.makedirs(output_path, exist_ok=True)
    color_path = os.path.join(seg_path, "Ensemble_color")
    os.makedirs(color_path, exist_ok=True)
   
    colormap = generate_colormap(num_classes)
   
    for image in images:
        if len(folds) >= 1:
            ensemble = np.zeros((num_classes, *img.shape[:2]), dtype=np.float32)
            for fold in folds:
                seg = io.imread(os.path.join(seg_path, fold, os.path.basename(image)), as_gray=True)
                # convert to one-hot encoding
                for i in range(num_classes):
                    ensemble[i] += (seg == i)
            ensemble /= len(folds)  # Normalize
        else:
            ensemble = io.imread(os.path.join(seg_path, os.path.basename(image)), as_gray=True)
            ensemble = np.eye(num_classes)[ensemble].transpose(2, 0, 1)
        
        accum[0] = ensemble[0]
        # Only accumulate on the main and lateral roots
        accum[1:3] = float(alpha) * accum[1:3] + ensemble[1:3]
        if num_classes > 3:
            accum[3:] = float(alpha) * accum[3:] + ensemble[3:]
        segmentation = np.argmax(accum, axis=0)
        
        # Save segmentation as PNG with class values
        io.imsave(os.path.join(output_path, f"{os.path.splitext(os.path.basename(image))[0]}.png"),
                  segmentation.astype(np.uint8), check_contrast=False)
       
        # Save segmentation using colormap
        color_segmentation = colormap[segmentation]
        color_segmentation = (color_segmentation[:, :, :3]).astype(np.uint8)
        io.imsave(os.path.join(color_path, f"{os.path.splitext(os.path.basename(image))[0]}.png"),
                  color_segmentation, check_contrast=False)
        
        del ensemble
   
    return

# Example usage:
# ensemble("/path/to/images", alpha=0.5, num_classes=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files and revert them back to original names.")
    parser.add_argument("path", help="Path to the folder containing the segmentation files.")
    parser.add_argument("--alpha", default=0.85, help="Weight for the temporal postprocessing step.")
    parser.add_argument("--num_classes", default=7, help="Number of classes in multiclass segmentation")
    parser.add_argument("--seg_path", default="Segmentation", help="Number of classes in multiclass segmentation")
    args = parser.parse_args()

    ensemble(args.path, args.alpha, args.num_classes, args.seg_path)
