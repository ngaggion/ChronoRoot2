import torch
import numpy as np
from pathlib import Path
from PIL import Image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from datetime import datetime
import time
import json

import warnings
# Remove all warnings from nnunetv2 module
warnings.filterwarnings("ignore", module="nnunetv2")

class nnUNetv2:
    """
    Minimal wrapper for nnUNetv2 inference with PNG output support.
    Designed for 2D architectures.
    """
    
    def __init__(self, model_path: str, device: str = 'auto', verbose: bool = False, use_gaussian: bool = True,
                use_mirroring: bool = True, tile_step_size: float = 0.5):
        """
        Initialize the nnUNet predictor.
        
        Args:
            model_path: Path to trained model folder (contains fold_X subdirectories)
            device: 'auto', 'cuda', 'mps', or 'cpu'
            verbose: Print detailed information during processing
        """
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.predictor = None
        self.verbose = verbose
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.tile_step_size = tile_step_size

        self.perform_everything_on_device = self.device.type != 'cpu'
        
        self._initialize_predictor()

    def _resolve_device(self, device: str) -> torch.device:
        """
        Resolve a requested device string into a valid torch.device.
        Preference order for 'auto': cuda > mps > cpu
        """
        req = (device or "auto").lower().strip()

        if req == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if req == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")

        if req == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            print("MPS not available, falling back to CPU")
            return torch.device("cpu")

        # Default: cpu (and tolerate torch.device('cpu') style inputs)
        return torch.device(req if req in {"cpu"} else "cpu")
    
    def _initialize_predictor(self):
        """Initialize the nnUNet predictor from trained model folder."""
        print(f"Loading model from: {self.model_path}")
        
        self.predictor = nnUNetPredictor(
            tile_step_size=self.tile_step_size,
            use_gaussian=self.use_gaussian,
            use_mirroring=self.use_mirroring,
            perform_everything_on_device=self.perform_everything_on_device,
            device=self.device,
            verbose=self.verbose,
            verbose_preprocessing=self.verbose,
            allow_tqdm=False
        )
        
        self.predictor.initialize_from_trained_model_folder(
            str(self.model_path),
            use_folds=(0,),  # Use fold 0, or specify multiple folds
            checkpoint_name='checkpoint_final.pth'
        )
    
    def _read_png_image(self, image_path: str):
        """
        Read PNG image using OpenCV in grayscale mode.
        
        Args:
            image_path: Path to PNG image
            
        Returns:
            tuple: (image_array, properties_dict)
        """
        
        # Read image in grayscale with PIL
        img = np.array(Image.open(image_path).convert('L'))
        
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        img = img[np.newaxis, np.newaxis, :, :].astype(np.float32)  

        # Create properties dict with spacing info
        # For 2D images, spacing corresponds to (height, width) dimensions
        # Using 1.0 as default pixel spacing if not specified
        properties = {
            'spacing': np.array([999.0, 1.0, 1.0], dtype=np.float64)
        }
        
        if self.verbose:
            print(f"  Image shape: {img.shape}, spacing: {properties['spacing']}")
        
        return img, properties
    
    def _update_metadata(self, metadata_path: Path, data: dict):
        """Internal helper to update the metadata JSON."""
        if not metadata_path:
            return
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        metadata.update(data)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
    def predict_from_folder(self, input_dir: str, output_dir: str, 
                        save_as_png: bool = True, metadata_path: str = None, 
                        processed_files: list = None):
        
        """
        Run inference on all PNG images in a folder.
        Args:
            input_dir: Directory with input PNG images
            output_dir: Directory to save predictions
            save_as_png: If True, save predictions as PNG images
            metadata_path: Path to metadata JSON file for progress updates
            processed_files: List of already processed files to skip (optional)
        """
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        m_path = Path(metadata_path) if metadata_path else None
        
        png_files = sorted(list(input_path.glob('*.png')))
        if not png_files:
            raise ValueError(f"No PNG files found in {input_path}")
        
        total_files = len(png_files)
        print(f"Found {total_files} images in: {input_path}")
        
        # If resume is on, we skip files already in the set.
        if processed_files is not None:
            files_to_process = [f for f in png_files if f.name not in processed_files]
            print(f"Resuming: Skipping {len(processed_files)} images.")
        else:
            files_to_process = png_files
            processed_files = set() # Initialize empty set for metadata tracking
        
        # Initial metadata update
        self._update_metadata(m_path, {"n_images": total_files, "processed_images": len(processed_files)})

        time_per_image = []
        
        for i, png_file in enumerate(files_to_process): 
            t_start = time.time()
            
            img, props = self._read_png_image(png_file)
            result = self.predictor.predict_single_npy_array(img, props, None, None, False)
            
            output_file = output_path / png_file.name
            if save_as_png:
                self._save_array_as_png(result, output_file)
            else:
                np.save(str(output_file.with_suffix('.npy')), result)
            
            t_end = time.time()
            time_per_image.append(t_end - t_start)
            
            processed_files.add(png_file.name)
            
            # Update metadata every image
            self._update_metadata(m_path, {
                "processed_images": len(processed_files),
                "processed_files": list(processed_files),
                "segmentation_progress": round((len(processed_files) / total_files) * 100, 2),
                "last_segmentation_time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "segmentation_average_time_per_image": round(np.mean(time_per_image), 2)
            })
            
            print(f"Processed: {png_file.name}")
        
        if len(processed_files) == total_files and m_path:
            self._update_metadata(m_path, {
                "processed_files": "All",
                "segmentation_status": "Success"
            })
        
        print("Segmentation complete.")
        print(f"\nCompleted processing {total_files} images in {sum(time_per_image):.2f} seconds.")
        print(f"Average time per image: {np.mean(time_per_image):.2f} seconds.")
    
    def predict_single_image(self, image_path: str, output_path: str = None,
                            save_as_png: bool = True):
        """
        Run inference on a single PNG image.
        
        Args:
            image_path: Path to input PNG image
            output_path: Path to save prediction (optional)
            save_as_png: If True, save as PNG
            
        Returns:
            Segmentation array if output_path is None
        """
        
        img, props = self._read_png_image(image_path)
        
        # Run prediction
        result = self.predictor.predict_single_npy_array(
            img, props, None, None, False
        )
        
        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_as_png:
                self._save_array_as_png(result, output_path)
            else:
                # Save as numpy
                np.save(str(output_path), result)
            
        return result
    
    def _save_array_as_png(self, array: np.ndarray, output_path: Path):
        """Save a 2D segmentation array as PNG."""
        # Assume 2D array, squeeze if needed
        if array.ndim > 2:
            array = np.squeeze(array)
        
        # Normalize to 0-255 for visualization
        array = array.astype(np.uint8)
        
        # If output_path doesn't have .png extension, add it
        if output_path.suffix != '.png':
            output_path = output_path.with_suffix('.png')
        
        # Save as PNG
        img = Image.fromarray(array)
        img.save(str(output_path))
