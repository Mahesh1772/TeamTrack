import os
from pathlib import Path
import numpy as np

def check_dataset_split(labels_path, split_name="unknown"):
    total_images = 0
    images_with_balls = 0
    images_with_players = 0
    ball_instances = 0
    player_instances = 0
    
    print(f"\nChecking {split_name} set in {labels_path}")
    
    if not os.path.exists(labels_path):
        print(f"ERROR: {split_name} labels path does not exist: {labels_path}")
        return
    
    for label_file in Path(labels_path).glob('*.txt'):
        total_images += 1
        try:
            labels = np.loadtxt(label_file)
            if labels.size == 0:
                continue
                
            # Handle single label case
            if labels.ndim == 1:
                labels = labels.reshape(1, -1)
            
            # Count instances
            balls = labels[labels[:, 0] == 1]
            players = labels[labels[:, 0] == 0]
            
            if len(balls) > 0:
                images_with_balls += 1
                ball_instances += len(balls)
            if len(players) > 0:
                images_with_players += 1
                player_instances += len(players)
                
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
    
    print(f"\n{split_name} Set Statistics:")
    print(f"Total images: {total_images}")
    print(f"Images with balls: {images_with_balls} ({(images_with_balls/total_images)*100:.2f}% of images)")
    print(f"Images with players: {images_with_players} ({(images_with_players/total_images)*100:.2f}% of images)")
    print(f"Total ball instances: {ball_instances}")
    print(f"Total player instances: {player_instances}")
    
    if ball_instances == 0:
        print("\n⚠️ WARNING: No ball instances found! This will cause training issues.")
    
    return total_images, ball_instances, player_instances

def main():
    yaml_path = 'data/teamtrack-yolo/Handball_SideView/data.yaml'
    
    # Load yaml
    with open(yaml_path, 'r') as f:
        import yaml
        data = yaml.safe_load(f)
    
    root = Path(data['path'])
    
    # Check both training and validation sets
    train_path = root / 'Handball_SideView-train' / 'labels'
    val_path = root / 'Handball_SideView-val' / 'labels'
    
    train_stats = check_dataset_split(train_path, "Training")
    val_stats = check_dataset_split(val_path, "Validation")
    
    # Print summary and recommendations
    print("\n=== Summary and Recommendations ===")
    if train_stats[1] == 0 or val_stats[1] == 0:
        print("\n❌ Critical Issues Found:")
        print("- No ball instances detected in one or both splits")
        print("\nPossible solutions:")
        print("1. Check if labels were correctly converted from MOT format")
        print("2. Verify class IDs in the conversion process (ball should be class 1)")
        print("3. Make sure ball annotations were not filtered out during conversion")
        print("4. Consider re-running the dataset conversion pipeline")
    else:
        ratio = val_stats[1] / train_stats[1]
        print(f"\nBall instance ratio (val/train): {ratio:.2f}")
        if ratio < 0.1 or ratio > 0.3:
            print("⚠️ Warning: Unusual validation/training ratio for ball instances")

if __name__ == "__main__":
    main() 