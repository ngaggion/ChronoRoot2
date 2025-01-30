import argparse
import pathlib
import re
import os

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def loadPath(search_path, ext='*.*'):
    data_root = pathlib.Path(search_path)
    # Make sure the path exists
    if not data_root.exists():
        raise FileNotFoundError(f"Path {search_path} does not exist")
    all_files = list(data_root.glob(ext))
    # Filter out directories, only keep files
    all_files = [str(path) for path in all_files if path.is_file()]
    all_files.sort(key=natural_key)
    return all_files
    
def rename_files(path):
    """
    Rename files to the nnUNet format while maintaining a name mapping.
    Returns a dictionary mapping new names to original names.
    """
    files = loadPath(path, ext="*.png")
    if not files:
        print(f"No PNG files found in {path}")
        return {}
    
    # First, check if we already have a mapping file
    mapping_file = os.path.join(path, "name_mapping.txt")
    if os.path.exists(mapping_file):
        print("Found existing name mapping file. Removing it to start fresh...")
        os.remove(mapping_file)
    
    # Clear any existing renamed files to avoid conflicts
    existing_renamed = list(pathlib.Path(path).glob("image_*_0000.png"))
    for file in existing_renamed:
        try:
            os.remove(file)
            print(f"Removed existing renamed file: {file}")
        except OSError as e:
            print(f"Warning: Could not remove {file}: {e}")
    
    name_mapping = {}
    counter = 0
    
    # First pass: collect all original files that need renaming
    files_to_rename = []
    for file in sorted(files, key=natural_key):
        file_path = pathlib.Path(file)
        if not re.match(r'image_\d{3}_0000\.png$', file_path.name):
            files_to_rename.append(file_path)
    
    # Second pass: do the actual renaming
    for file_path in files_to_rename:
        new_name = f"image_{counter:03d}_0000.png"
        new_path = file_path.parent / new_name
        
        try:
            #print(f"Renaming: {file_path.name} -> {new_name}")
            os.rename(file_path, new_path)
            name_mapping[new_name] = file_path.name
            counter += 1
        except OSError as e:
            print(f"Error renaming {file_path}: {e}")
    
    return name_mapping


def revert_file_names(path, name_mapping):
    """
    Revert files back to their original names.
    """
    for new_name, original_name in name_mapping.items():
        new_path = pathlib.Path(path) / new_name
        original_path = pathlib.Path(path) / original_name
        
        if not new_path.exists():
            print(f"Warning: {new_path} does not exist")
            continue            
        try:
            #print(f"Reverting: {new_name} -> {original_name}")
            os.rename(new_path, original_path)
        except OSError as e:
            print(f"Error reverting {new_path}: {e}")

def revert_seg_file_names(path, name_mapping):
    if not pathlib.Path(path).exists():
        print(f"Segmentation path {path} does not exist")
        return
        
    for new_name, original_name in name_mapping.items():
        if '/Fold_' not in str(path):
            # Handle color and classes files
            for suffix in ['_color.png', '_classes.png']:
                new_name_mod = new_name.replace(".png", suffix)
                new_path = pathlib.Path(path) / new_name_mod
                original_path = pathlib.Path(path) / original_name.replace(".png", suffix)
                
                if not new_path.exists():
                    print(f"Warning: {new_path} does not exist")
                    continue
                    
                if original_path.exists():
                    print(f"Warning: Cannot revert to {original_path}, file already exists")
                    continue
                    
                try:
                    os.rename(new_path, original_path)
                except OSError as e:
                    print(f"Error reverting {new_path}: {e}")
        else:
            # Handle Fold directory files
            new_name = new_name.replace("_0000.png", ".png")
            new_path = pathlib.Path(path) / new_name
            original_path = pathlib.Path(path) / original_name
            
            if not new_path.exists():
                print(f"Warning: {new_path} does not exist")
                continue
                
            if original_path.exists():
                print(f"Warning: Cannot revert to {original_path}, file already exists")
                continue
                
            try:
                os.rename(new_path, original_path)
            except OSError as e:
                print(f"Error reverting {new_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files and revert them back to original names.")
    parser.add_argument("path", help="Path to the folder containing files to be renamed.")
    parser.add_argument("--revert", action="store_true", help="Revert file names to original names.")
    parser.add_argument("--revert_seg", action="store_true", help="Revert segmentation file names to original names.")
    parser.add_argument("--segpath", help="Path to the segmentation folder containing files to be renamed.")
    args = parser.parse_args()

    # Validate path existence
    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist")
        exit(1)

    mapping_file = os.path.join(args.path, "name_mapping.txt")

    if args.revert:
        if not os.path.exists(mapping_file):
            print(f"Error: Name mapping file not found at {mapping_file}")
            exit(1)
        try:
            with open(mapping_file, "r") as f:
                name_mapping = {line.split(',')[0]: line.split(',')[1].strip() for line in f}
            revert_file_names(args.path, name_mapping)
        except Exception as e:
            print(f"Error reverting files: {e}")
            
    elif args.revert_seg:
        if not os.path.exists(mapping_file):
            print(f"Error: Name mapping file not found at {mapping_file}")
            exit(1)
        if args.segpath == "None" or not args.segpath:
            print("Error: --segpath is required when using --revert_seg")
            exit(1)
        try:
            with open(mapping_file, "r") as f:
                name_mapping = {line.split(',')[0]: line.split(',')[1].strip() for line in f}
            revert_seg_file_names(args.segpath, name_mapping)
        except Exception as e:
            print(f"Error reverting segmentation files: {e}")
            
    else:
        print("RENAMING")
        if os.path.exists(os.path.join(args.path, "image_000_0000.png")):
            print("Names were already changed.")
        else:
            try:
                name_mapping = rename_files(args.path)
                if name_mapping:
                    with open(mapping_file, "w") as f:
                        for new_name, original_name in name_mapping.items():
                            f.write(f"{new_name},{original_name}\n")
                else:
                    print("No files were renamed")
            except Exception as e:
                print(f"Error renaming files: {e}")