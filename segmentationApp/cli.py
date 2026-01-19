#!/usr/bin/env python3
"""
CLI interface for nnUNet segmentation and postprocessing.
"""

import argparse
from pathlib import Path
import sys
import json
import warnings
from datetime import datetime

# Import existing modules
from nnUNet_wrapper import nnUNetv2
from postprocess import postprocess

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(
        description='nnUNet CLI for segmentation and postprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input', 
                       help='Input folder containing images to segment')
    
    # Optional Output Path
    parser.add_argument('--output', '-o',
                       help='Optional custom output directory. If provided, results go to '
                            'OUTPUT/Segmentation/Fold_0 instead of inside the input folder.')
    
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
                       help='Alpha parameter for postprocessing')

    args = parser.parse_args()
    
    # 1. Path Resolution
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Determine result base (either input folder or custom output)
    result_base = Path(args.output).resolve() if args.output else input_path
    
    # Final segmentation folder: result_base/Segmentation/Fold_0
    seg_folder_name = 'Segmentation'
    output_path = result_base / seg_folder_name / 'Fold_0'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # If args.alpha is not provided, set default based on species
    if args.alpha is None:
        args.alpha = 0.85 if args.species == 'arabidopsis' else 0.60

    # 2. Model setup
    script_dir = Path(__file__).parent.resolve()
    model_name = "Arabidopsis" if args.species == "arabidopsis" else "Tomato"
    model_path = script_dir / "models" / model_name
    
    if not model_path.exists() and not args.postprocess_only:
        print(f"Error: Model not found at: {model_path}")
        sys.exit(1)
     
    # 3. Generate Metadata JSON with day, hour, and other info
    metadata = {
        "images_path": str(input_path),
        "segmentation_path": str(result_base / seg_folder_name),
        "model_species": args.species,
        "alpha": args.alpha,
        "date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }
    
    if args.postprocess_only:
        metadata["note"] = "Postprocessing only"
    elif args.fast:
        metadata["note"] = "Fast mode"
        metadata["model_path"] = str(model_path)
    else:
        metadata["note"] = "Standard mode"
        metadata["model_path"] = str(model_path)   
    
    metadata_file = result_base / 'segmentation_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Metadata written to {metadata_file}")
    
    # 4. Run segmentation
    if not args.postprocess_only:
        print(f"\n=== Segmentation ===")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        
        model = nnUNetv2(
            model_path=str(model_path),
            device=args.device,
            verbose=args.verbose,
            use_gaussian=True,
            use_mirroring=not args.fast,
            tile_step_size=0.5
        )
        
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
    
    # 5. Run Postprocessing
    print(f"\n=== Postprocessing ===")
    try:
        postprocess(
            path=str(input_path),
            method=args.species,
            alpha=args.alpha,
            seg_path=str(result_base / seg_folder_name)
        )
        print(f"✓ Postprocessing complete")
    except Exception as e:
        print(f"✗ Postprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()