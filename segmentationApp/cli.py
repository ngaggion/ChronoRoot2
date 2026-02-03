#!/usr/bin/env python3
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Suppress external library warnings for a cleaner CLI output
warnings.filterwarnings("ignore")

from postprocess import postprocess
from nnUNet_wrapper import nnUNetv2

def get_available_models():
    """Scans the local 'models/' directory for available nnU-Net species folders."""
    models_dir = Path(__file__).parent.resolve() / "models"
    if not models_dir.exists():
        return []
    return [d.name for d in models_dir.iterdir() if d.is_dir()]

def update_metadata(file_path, data, mode='update'):
    """
    Handles JSON metadata persistence.
    mode='update': Merges new data into existing JSON.
    mode='set': Overwrites/Initializes the JSON file.
    """
    metadata = {}
    if mode == 'update' and file_path.exists():
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
            
    metadata.update(data)
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def main():
    # 1. SETUP & ARGPARSE
    available_models = get_available_models()
    
    description = """
ChronoRoot 2.0 - Automated Plant Root Segmentation Pipeline
-----------------------------------------------------------
1. nnU-Net Segmentation: Generates raw masks from PNG root images.
2. Temporal Postprocessing: Refines masks via temporal weighted trailing average.

Usage:
  Standard:  python cli.py /path/to/images --model Tomato
  Resume:    python cli.py /path/to/images --model Arabidopsis --resume
  Refine:    python cli.py /path/to/images --model Tomato --alpha 0.7 --postprocess-only
    """

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required Arguments
    parser.add_argument('input', help='Input folder containing PNG images.')
    parser.add_argument('--model', '-m', required=True, choices=available_models,
                        help='Pre-trained model to use. Folders must exist in models/')

    # Processing Parameters
    proc_group = parser.add_argument_group('Processing Parameters')
    proc_group.add_argument('--alpha', '-a', type=float, 
                            help='Temporal alpha (0.0-1.0). Default: Arabidopsis=0.85, Tomato=0.60')
    proc_group.add_argument('--output', '-o', help='Custom output path. Default: Input folder.')
    proc_group.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Default: cuda')
    proc_group.add_argument('--fast', action='store_true', help='Disable mirroring for speed.')

    # Execution Flow
    exec_group = parser.add_argument_group('Execution Modes')
    exec_group.add_argument('--postprocess-only', action='store_true', help='Skip segmentation.')
    exec_group.add_argument('--resume', action='store_true', help='Resume from metadata state.')

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)
        
    result_base = Path(args.output).resolve() if args.output else input_path
    segmentation_dir = result_base / 'Segmentation'
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = segmentation_dir / 'segmentation_metadata.json'
    
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir / "models" / args.model

    processed_files = None
    if args.resume:
        if not metadata_file.exists():
            args.resume = False
        else:
            try:
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)
                    if meta.get("segmentation_status") == "Success":
                        print("Previous segmentation complete. Skipping to postprocessing.")
                        args.postprocess_only = True
                                                
                    elif isinstance(meta.get("processed_files"), list):
                        processed_files = set(meta["processed_files"])
                        print(f"Resuming: {len(processed_files)} images found in metadata.")
            except Exception as e:
                print(f"Warning: Could not parse metadata for resume: {e}")

    # Set default alpha if not provided
    if args.alpha is None:
        args.alpha = 0.85 if "arabidopsis" in args.model.lower() else 0.60

    # Initialize metadata file for a new run
    if not args.postprocess_only and not args.resume:
        update_metadata(metadata_file, {
            "input_path": str(input_path),
            "output_path": str(result_base),
            "alpha_used": args.alpha,
            "model": args.model,
            "fast_mode": args.fast,
            "segmentation_status": "Not started",
            "postprocessing_status": "Not started"
        }, mode='set')
    
    # check if the metadata contains alpha_used and model keys, if not add them
    if metadata_file.exists():
        update_metadata(metadata_file, {
            "alpha_used": args.alpha,
            "model": args.model
        })

    if not args.postprocess_only:
        print(f"\n--- Stage 1: Segmentation (Model: {args.model}) ---")
        update_metadata(metadata_file, {
            "segmentation_status": "Started",
            "segmentation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        try:
            model = nnUNetv2(
                model_path=str(model_path), 
                device=args.device, 
                use_mirroring=not args.fast
            )
            
            model.predict_from_folder(
                input_dir=str(input_path), 
                output_dir=str(segmentation_dir / 'Fold_0'),
                save_as_png=True,
                metadata_path=metadata_file,
                processed_files=processed_files
            )
            update_metadata(metadata_file, {"segmentation_status": "Success"})
        except Exception as e:
            update_metadata(metadata_file, {"segmentation_status": f"Error: {str(e)}"})
            print(f"Segmentation failed: {e}")
            sys.exit(1)

    # check if resume and postprocessing only
    if args.resume and args.postprocess_only:
        try:
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
                     # Check if the postprocessing was also completed
            if meta.get("postprocessing_status") == "Success":
                # if the model and the alpha are the same, we can skip postprocessing
                print("Previous postprocessing also complete.")
                print(f"Previous model: {meta.get('model')}, Current model: {args.model}")
                print(f"Previous alpha: {meta.get('alpha_used')}, Current alpha: {args.alpha}")
                if (meta.get("model") == args.model and 
                    meta.get("alpha_used") == args.alpha):
                    print("Postprocessing also complete with same parameters. Exiting.")
                    sys.exit(0)     
        except Exception as e:
            print(f"Error reading metadata for postprocessing: {e}")
            sys.exit(1)
    
    print(f"\n--- Stage 2: Temporal Postprocessing (α={args.alpha}) ---")
    update_metadata(metadata_file, {
        "postprocessing_status": "Started",
        "postprocessing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "alpha_used": args.alpha
    })
            
    try:
        finished = postprocess(
            path=str(input_path), 
            alpha=args.alpha, 
            seg_path=str(segmentation_dir),
            metadata_path=metadata_file
        )
        
        if not finished:
            raise RuntimeError("Postprocessing did not complete successfully.")
        
        update_metadata(metadata_file, {"postprocessing_status": "Success"})
        print(f"Pipeline finished successfully at {datetime.now().strftime('%H:%M:%S')}")
    except Exception as e:
        update_metadata(metadata_file, {"postprocessing_status": f"Error: {str(e)}"})
        print(f"Postprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()