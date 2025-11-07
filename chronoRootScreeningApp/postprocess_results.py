import os
import argparse
import pandas as pd
import json
from data_processing.germination_analysis import GerminationAnalyzer
from data_processing.plant_analysis import PlantGrowthAnalyzer

# ignore future warnings from pandas
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def merge_analysis_files(project_dir: str, name_mapping_file: str = None) -> pd.DataFrame:
    """
    Merge all seeds.tsv files from different analyses into one dataframe.
    
    Args:
        project_dir: Project directory containing analysis folders
        name_mapping_file: Path to JSON file with group name mappings
    """
    analysis_dir = os.path.join(project_dir, 'analysis')
    all_data = []
    
    # Load name mapping if provided
    name_mapping = {}
    if name_mapping_file and os.path.exists(name_mapping_file):
        try:
            with open(name_mapping_file, 'r') as f:
                name_mapping = json.load(f)
        except Exception as e:
            print(f"Error loading name mapping: {str(e)}")
    
    # Get all analysis folders
    analyses = [d for d in os.listdir(analysis_dir)
                if os.path.isdir(os.path.join(analysis_dir, d))]
    
    for analysis_id in analyses:
        analysis_path = os.path.join(analysis_dir, analysis_id, 'seeds.tsv')
        metadata_path = os.path.join(analysis_dir, analysis_id, 'group_info.json')
        
        if os.path.exists(analysis_path):
            try:
                df = pd.read_csv(analysis_path, sep='\t')
                
                # make Group, UID string columns
                df['Group'] = df['Group'].astype(str)
                df['UID'] = df['UID'].astype(str)
                
                # Store original group names before mapping
                df['OriginalGroup'] = df['Group']
                
                # Apply name mapping if provided
                if name_mapping:
                    df['Group'] = df['Group'].map(lambda x: name_mapping.get(x, x))
                
                metadata = pd.read_json(metadata_path)
                metadata = metadata.rename(columns={
                    "group_names": "Group",
                    "seed_counts": "SeedCount"
                })
                
                # metadata group names should be string
                metadata['Group'] = metadata['Group'].astype(str)
                
                # Apply name mapping to metadata too
                if name_mapping:
                    metadata['OriginalGroup'] = metadata['Group']
                    metadata['Group'] = metadata['Group'].map(lambda x: name_mapping.get(x, x))
                
                df = df.merge(metadata, on='Group')
                df['Video'] = analysis_id
                
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {analysis_path}: {str(e)}")
    
    if not all_data:
        raise ValueError("No valid seeds.tsv files found in any analysis folder")
    
    return pd.concat(all_data, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(
        description='Post-process germination analysis results'
    )
    parser.add_argument(
        '--project-dir',
        required=True,
        help='Project directory containing analysis folders'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.25,
        help='Delta time between measurements (hours)'
    )
    parser.add_argument(
        '--name-mapping',
        type=str,
        help='JSON file with group name mappings'
    )
    parser.add_argument(
        '--add-time-before-photo',
        type=int,
        default=0,
        help='Add time before first photo (in integer hours)'
    )
    args = parser.parse_args()
        
    try:
        # Setup
        results_dir = os.path.join(args.project_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Load and merge data with name mapping
        print("Merging analysis files...")
        combined_data = merge_analysis_files(args.project_dir, args.name_mapping)
        combined_data.to_csv(
            os.path.join(results_dir, 'Raw_Data.tsv'),
            sep='\t',
            index=False
        )
                
        # Run germination analysis
        print("Running germination analysis...")
        germ_analyzer = GerminationAnalyzer(
            data=combined_data,
            output_dir=results_dir,
            dt=args.dt,
            add_time_before_photo=args.add_time_before_photo
        )
        germ_analyzer.analyze()
        
        # Run plant growth analysis
        print("Analyzing plant growth...")
        plant_analyzer = PlantGrowthAnalyzer(
            data=combined_data,
            output_dir=results_dir,
            add_time_before_photo=args.add_time_before_photo
        )
        plant_analyzer.analyze_all_parameters()

        print("Analysis complete!")
        print(f"Results saved in: {results_dir}")
                    
    except Exception as e:
        print(f"Error during post-processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()