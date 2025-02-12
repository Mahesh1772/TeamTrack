import pandas as pd
import argparse
from pathlib import Path

def check_duplicates(gt_file):
    """Check for duplicate entries in MOT gt.txt file"""
    # Read the gt.txt file
    df = pd.read_csv(gt_file, header=None, 
                     names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    
    # Group by frame and id
    for (frame, id), group in df[df.duplicated(['frame', 'id'], keep=False)].groupby(['frame', 'id']):
        if len(group) > 1:
            # Check if coordinates are different
            coords = group[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
            if not (coords[0] == coords[1]).all():
                print(f"\nFound different coordinates for frame {frame}, id {id} in {gt_file}:")
                print(group[['bb_left', 'bb_top', 'bb_width', 'bb_height']])
                return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mot-dir', type=str, required=True, help='MOT format directory')
    args = parser.parse_args()
    
    mot_dir = Path(args.mot_dir)
    
    # Check all gt.txt files
    has_different_coords = False
    for gt_file in mot_dir.glob("**/gt/gt.txt"):
        if check_duplicates(gt_file):
            has_different_coords = True
            
    if not has_different_coords:
        print("\nAll duplicates have identical coordinates!")

if __name__ == "__main__":
    main() 