import torch
import numpy as np
from pathlib import Path
from PIL import Image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class nnUNetv2:
    """
    Minimal wrapper for nnUNetv2 inference with PNG output support.
    Designed for 2D architectures.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda', verbose: bool = False, use_gaussian: bool = True,
                 use_mirroring: bool = True, tile_step_size: float = 0.5):
        """
        Initialize the nnUNet predictor.
        
        Args:
            model_path: Path to trained model folder (contains fold_X subdirectories)
            device: 'cuda', 'cpu', or 'mps'
            verbose: Print detailed information during processing
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.predictor = None
        self.verbose = verbose
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.tile_step_size = tile_step_size

        # Check GPU availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
        
        self._initialize_predictor()
    
    def _initialize_predictor(self):
        """Initialize the nnUNet predictor from trained model folder."""
        print(f"Initializing predictor from: {self.model_path}")
        
        self.predictor = nnUNetPredictor(
            tile_step_size=self.tile_step_size,
            use_gaussian=self.use_gaussian,
            use_mirroring=self.use_mirroring,
            perform_everything_on_device=True,
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
        
        print("Predictor initialized successfully!")
    
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
    
    def predict_from_folder(self, input_dir: str, output_dir: str, 
                           save_as_png: bool = True):
        """
        Run inference on all PNG images in input directory.
        
        Args:
            input_dir: Directory containing input PNG images
            output_dir: Directory to save predictions
            save_as_png: If True, convert and save as PNG (for 2D only)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all PNG files
        png_files = sorted(list(input_path.glob('*.png')))
        
        if not png_files:
            raise ValueError(f"No PNG files found in {input_path}")
        
        print(f"Found {len(png_files)} PNG images in: {input_path}")
        print(f"Saving results to: {output_path}")
        
        results = []
        for png_file in png_files:
            print(f"Processing: {png_file.name}")
            
            # Read image
            img, props = self._read_png_image(png_file)
            
            # Run prediction
            result = self.predictor.predict_single_npy_array(
                img, props, None, None, False
            )
            
            # Save result
            output_file = output_path / png_file.name
            if save_as_png:
                self._save_array_as_png(result, output_file)
            else:
                np.save(str(output_file.with_suffix('.npy')), result)
            
            results.append(result)
            print(f"  -> Saved: {output_file.name}")
        
        print(f"\nCompleted processing {len(results)} images!")
        return results
    
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
