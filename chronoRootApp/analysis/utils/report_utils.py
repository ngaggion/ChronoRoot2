import re
import pathlib
import os
import numpy as np

def natural_key(string_text):
    """
    Sorts strings containing numbers in a human-friendly way 
    (e.g., 'File 2' comes before 'File 10').
    See http://www.codinghorror.com/blog/archives/001018.html
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_text)]

def load_paths(search_path, ext='*.*'):
    """
    Glob patterns for files and sorts them naturally.
    """
    data_root = pathlib.Path(search_path)
    # The glob pattern needs to be consistent with pathlib usage
    # If ext contains wildcards like *.*, we use it directly
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key=natural_key)
    return all_files

def ensure_directory(path):
    """Safely creates a directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass

def unit_vector(vector):
    """Returns the unit vector of the vector."""
    norm = np.linalg.norm(vector)
    if norm == 0: 
        return vector
    return vector / norm

def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'.
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # Clip to handle floating point errors outside [-1, 1]
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))