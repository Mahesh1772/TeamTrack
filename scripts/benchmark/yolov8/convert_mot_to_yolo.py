import os
import glob
from pathlib import Path
import shutil
import argparse
import numpy as np

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_SILENT'] = 'true'

def convert_mot_to_yolo(mot_path, output_base):
    """Convert MOT format to YOLO format using confidence value to identify balls
    Args:
        mot_path: Path to MOT sequence
        output_base: Base path for YOLO output
    """
    print(f"Converting {mot_path}")
    
    # Read gt.txt
    gt_file = os.path.join(mot_path, 'gt', 'gt.txt')
    if not os.path.exists(gt_file):
        print(f"No gt.txt found in {mot_path}")
        return
        
    # Read image dimensions from seqinfo.ini
    seqinfo_file = os.path.join(mot_path, 'seqinfo.ini')
    if not os.path.exists(seqinfo_file):
        print(f"No seqinfo.ini found in {mot_path}")
        return
        
    img_width = None
    img_height = None
    
    with open(seqinfo_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'imwidth' in line.lower():
                try:
                    img_width = int(line.split('=')[1].strip())
                except:
                    print(f"Error parsing imwidth in {seqinfo_file}")
            elif 'imheight' in line.lower():
                try:
                    img_height = int(line.split('=')[1].strip())
                except:
                    print(f"Error parsing imheight in {seqinfo_file}")
    
    if img_width is None or img_height is None:
        print(f"Could not find image dimensions in {seqinfo_file}")
        return
    
    print(f"Found image dimensions: {img_width}x{img_height}")
    
    # Setup output directories
    seq_name = os.path.basename(os.path.dirname(mot_path))
    output_path = os.path.join(output_base, seq_name)
    images_path = os.path.join(output_path, 'images')
    labels_path = os.path.join(output_path, 'labels')
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    
    # Copy images
    for img in glob.glob(os.path.join(mot_path, 'img1', '*.jpg')):
        shutil.copy2(img, images_path)
    
    # Convert annotations using confidence value
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        
    # Group by frame
    frame_dict = {}
    ball_count = 0
    player_count = 0
    
    for line in lines:
        frame_id, track_id, x, y, w, h, conf, class_id, *_ = map(float, line.strip().split(','))
        frame_id = int(frame_id)
        
        # Simple class determination based on confidence value
        yolo_class = 1 if conf == 2 else 0  # 1 for ball (conf=2), 0 for player (conf=1)
        
        if yolo_class == 1:
            ball_count += 1
        else:
            player_count += 1
        
        # Convert to YOLO format (normalized)
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        width = w / img_width
        height = h / img_height
        
        if frame_id not in frame_dict:
            frame_dict[frame_id] = []
        frame_dict[frame_id].append(f"{yolo_class} {x_center} {y_center} {width} {height}")
    
    # Write YOLO format files
    for frame_id in frame_dict:
        label_file = os.path.join(labels_path, f"{frame_id:06d}.txt")
        with open(label_file, 'w') as f:
            f.write('\n'.join(frame_dict[frame_id]))
    
    print(f"Conversion statistics for {seq_name}:")
    print(f"- Total frames: {len(frame_dict)}")
    print(f"- Ball instances: {ball_count}")
    print(f"- Player instances: {player_count}")
    
    return output_path

def create_data_yaml(output_dir):
    """Create data.yaml for YOLO training"""
    # Convert to absolute path
    abs_path = os.path.abspath(output_dir)
    
    yaml_content = f"""
path: {abs_path}  # dataset root dir
train: Handball_SideView-train/images  # train images
val: Handball_SideView-val/images  # val images
test: Handball_SideView-test/images  # test images

names:
  0: player
  1: ball
"""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    return yaml_path

def main():
    # 1. Setup paths
    mot_base = "data/teamtrack-mot"
    yolo_base = "data/teamtrack-yolo/Handball_SideView"
    os.makedirs(yolo_base, exist_ok=True)
    
    print(f"Converting MOT data from {mot_base} to YOLO format in {yolo_base}")
    
    # 2. Convert each split
    for split in ["Handball_SideView-train", "Handball_SideView-val", "Handball_SideView-test"]:
        split_path = os.path.join(mot_base, split)
        if os.path.exists(split_path):
            print(f"\nProcessing {split}...")
            for seq in os.listdir(split_path):
                seq_path = os.path.join(split_path, seq)
                if os.path.isdir(seq_path):
                    convert_mot_to_yolo(seq_path, yolo_base)
    
    # 3. Create data.yaml
    yaml_path = create_data_yaml(yolo_base)
    print(f"\nCreated dataset configuration at: {yaml_path}")

if __name__ == "__main__":
    main()
