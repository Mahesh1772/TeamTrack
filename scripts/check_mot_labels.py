import os
from pathlib import Path
import pandas as pd
import numpy as np

def check_mot_data(mot_path):
    print(f"\nChecking MOT format data in: {mot_path}")
    
    gt_file = Path(mot_path) / 'gt' / 'gt.txt'
    if not gt_file.exists():
        print(f"ERROR: GT file not found at {gt_file}")
        return
    
    # Read MOT format data
    cols = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
    df = pd.read_csv(gt_file, header=None, names=cols)
    
    # Analyze object sizes
    df['area'] = df['bb_width'] * df['bb_height']
    
    print("\nObject Size Analysis:")
    print(f"Minimum area: {df['area'].min():.2f}")
    print(f"Maximum area: {df['area'].max():.2f}")
    
    # Group by ID and analyze sizes
    id_analysis = df.groupby('id').agg({
        'area': ['mean', 'std', 'count'],
        'conf': 'first'
    }).reset_index()
    
    print("\nAnalysis by Object ID:")
    print(id_analysis.sort_values(('area', 'mean')))  # Show smallest objects first
    
    # Analyze by confidence value
    print("\nConfidence Value Distribution:")
    conf_counts = df['conf'].value_counts()
    print(conf_counts)
    
    # Potential ball detection
    small_objects = df[df['area'] < df['area'].median() * 0.3]  # Objects significantly smaller
    print("\nPotential Ball Detections:")
    print(f"Found {len(small_objects)} small objects that might be balls")
    
    # Detailed analysis of small objects
    if not small_objects.empty:
        print("\nSmall Object Statistics:")
        print(f"Average size: {small_objects['area'].mean():.2f}")
        print("\nSample of potential ball detections:")
        print(small_objects[['frame', 'id', 'bb_width', 'bb_height', 'conf']].head())
    
    # Save analysis for conversion
    potential_ball_ids = small_objects['id'].unique()
    print(f"\nPotential ball IDs: {potential_ball_ids}")
    
    return potential_ball_ids

def suggest_conversion_strategy():
    print("\n=== Conversion Strategy ===")
    print("Based on the analysis, we should:")
    print("1. Use object size and confidence to identify balls")
    print("2. Update the conversion script to map:")
    print("   - Small objects (potential balls) to class 1")
    print("   - Other objects to class 0 (players)")
    print("\nSuggested conversion logic:")
    print("""
    def determine_class(row):
        if row['id'] in ball_ids:  # or use size/confidence criteria
            return 1  # ball
        return 0  # player
    """)

if __name__ == "__main__":
    # Updated path to match your structure
    mot_path = "data/teamtrack-mot/Handball_SideView-test/2nd_fisheye_90-120"
    ball_ids = check_mot_data(mot_path)
    suggest_conversion_strategy() 