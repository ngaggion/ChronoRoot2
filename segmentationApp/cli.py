#!/usr/bin/env python3
"""
CLI interface for nnUNet segmentation and postprocessing.
"""

import argparse
from pathlib import Path
import sys

# Import existing modules
from nnUNet_wrapper import nnUNetv2
from postprocess import postprocess
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(
        description='nnUNet CLI for segmentation and postprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input', 
                       help='Input folder containing images to segment')
    
    # Model/species selection
    parser.add_argument('--species', default='arabidopsis', 
                       choices=['arabidopsis', 'tomato'],
                       help='Species/model to use (default: arabidopsis)')
    
    # Optional arguments
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode - disable test-time augmentation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Postprocessing options
    parser.add_argument('--postprocess-only', action='store_true',
                       help='Only run postprocessing (skip segmentation)')
    parser.add_argument('--alpha', type=float,
                       help='Alpha parameter for postprocessing (default: 0.85 for arabidopsis, 0.50 for tomato)')

    args = parser.parse_args()
    
    # Set paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Determine model path based on species
    script_dir = Path(__file__).parent.resolve()
    model_name = "Arabidopsis" if args.species == "arabidopsis" else "Tomato"
    model_path = script_dir / "models" / model_name
    
    if not model_path.exists():
        print(f"Error: Model not found at: {model_path}")
        sys.exit(1)
    
    output_path = input_path / 'Segmentation' / 'Fold_0'
    
    # Run segmentation unless postprocess-only
    if not args.postprocess_only:
        print(f"\n=== Segmentation ===")
        print(f"Species: {args.species}")
        print(f"Input: {input_path}")
        print(f"Fast mode: {args.fast}")
        
        # Initialize model
        model = nnUNetv2(
            model_path=str(model_path),
            device=args.device,
            verbose=args.verbose,
            use_gaussian=True,
            use_mirroring=not args.fast,
            tile_step_size=0.5
        )
        
        # Run prediction
        try:
            results = model.predict_from_folder(
                input_dir=str(input_path),
                output_dir=str(output_path),
                save_as_png=True
            )
            print(f"✓ Segmented {len(results)} images")
        except Exception as e:
            print(f"✗ Segmentation failed: {e}")
            sys.exit(1)
    
    # Run postprocessing
    print(f"\n=== Postprocessing ===")
    print(f"Method: {args.species}")
    print(f"Alpha: {args.alpha if args.alpha else 'default'}")
    
    try:
        postprocess(
            path=str(input_path),
            method=args.species,
            alpha=args.alpha,
            seg_path='Segmentation'
        )
        print(f"✓ Postprocessing complete")
    except Exception as e:
        print(f"✗ Postprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()