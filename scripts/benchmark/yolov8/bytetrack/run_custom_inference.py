from ultralytics import YOLO
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in format [x, y, w, h]"""
    # Convert to x1, y1, x2, y2
    box1_x1 = box1[0] - box1[2]/2
    box1_y1 = box1[1] - box1[3]/2
    box1_x2 = box1[0] + box1[2]/2
    box1_y2 = box1[1] + box1[3]/2
    
    box2_x1 = box2[0] - box2[2]/2
    box2_y1 = box2[1] - box2[3]/2
    box2_x2 = box2[0] + box2[2]/2
    box2_y2 = box2[1] + box2[3]/2
    
    # Calculate intersection
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    # Calculate union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def evaluate_predictions(predictions, ground_truth, frame_count):
    """Calculate evaluation metrics for predictions vs ground truth"""
    metrics = {
        'total_frames': frame_count,
        'frames_with_gt': len(ground_truth),
        'frames_with_pred': len(predictions),
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'mean_iou': 0,
        'position_mse': 0,
        'matched_predictions': []
    }
    
    iou_threshold = 0.25  # Reduced from 0.5 to be more lenient
    total_iou = 0
    position_errors = []
    
    # For each frame with ground truth
    for frame_idx in ground_truth:
        gt_box = ground_truth[frame_idx]
        if frame_idx in predictions:
            pred_box = predictions[frame_idx]
            
            # Debug print
            print(f"\nFrame {frame_idx}:")
            print(f"GT box: {gt_box}")
            print(f"Pred box: {pred_box}")
            
            iou = calculate_iou(gt_box, pred_box)
            print(f"IoU: {iou}")
            
            if iou >= iou_threshold:
                metrics['true_positives'] += 1
                total_iou += iou
                # Calculate position error (using center points)
                gt_center = np.array([gt_box[0], gt_box[1]])
                pred_center = np.array([pred_box[0], pred_box[1]])
                position_errors.append(np.sqrt(np.sum((gt_center - pred_center) ** 2)))
                metrics['matched_predictions'].append({
                    'frame': frame_idx,
                    'gt_box': gt_box.tolist(),
                    'pred_box': pred_box.tolist(),
                    'iou': iou
                })
            else:
                metrics['false_positives'] += 1
        else:
            metrics['false_negatives'] += 1
    
    # Add remaining predictions as false positives
    metrics['false_positives'] += len(set(predictions.keys()) - set(ground_truth.keys()))
    
    # Calculate final metrics
    if metrics['true_positives'] > 0:
        metrics['mean_iou'] = total_iou / metrics['true_positives']
        metrics['position_mse'] = np.mean(position_errors) if position_errors else 0
    
    # Calculate precision, recall, and F1
    precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
    recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
    
    return metrics

def get_best_detection(detections, confidence_threshold=0.3):
    """Get the highest confidence detection above threshold"""
    if len(detections) == 0:
        return None
        
    # Filter by confidence threshold
    valid_dets = detections[detections.conf >= confidence_threshold]
    if len(valid_dets) == 0:
        return None
        
    # Get highest confidence detection
    best_idx = valid_dets.conf.argmax()
    return valid_dets[best_idx]

def run_inference_on_videos():
    # Get absolute path to TeamTrack directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    teamtrack_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    
    # Base directories
    base_dir = r"C:\Users\Demo-user\Documents\Gameplay"
    video_base_dir = os.path.join(base_dir, "videos", "game1")
    gt_base_dir = os.path.join(base_dir, "labels", "game1")
    
    # List of clips to process
    clips = [f'Clip{i}' for i in range(1, 13)]
    
    # Load the model
    model_path = os.path.join(teamtrack_dir, "yolov8/Handball_Sideview_trained30/weights/best.pt")
    model = YOLO(model_path)
    tracker_config = os.path.join(current_dir, "tracker_config.yaml")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    print(f"Using model: {model_path}")
    
    # Initialize cumulative metrics
    cumulative_metrics = {
        'total_frames': 0,
        'frames_with_gt': 0,
        'frames_with_pred': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'total_iou': 0,
        'position_errors': []
    }

    for clip in clips:
        clip_lower = clip.lower()
        # Update video path to match new structure
        video_path = os.path.join(video_base_dir, clip, f"{clip}-1.mp4")
        # Update ground truth path to match new structure
        gt_dir = os.path.join(gt_base_dir, clip_lower)
        
        if not os.path.exists(video_path):
            print(f"WARNING: Video not found at {video_path}, skipping...")
            continue
            
        if not os.path.exists(gt_dir):
            print(f"WARNING: Ground truth directory not found at {gt_dir}, skipping...")
            continue
            
        print(f"\nProcessing video: {video_path}")
        print(f"Using ground truth from: {gt_dir}")
        
        # Create output directory
        output_dir = os.path.join(teamtrack_dir, "custom_video_results", clip)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Run inference
            results = model.track(
                source=video_path,
                save=True,
                save_txt=True,
                save_conf=True,
                conf=0.3,
                iou=0.5,
                imgsz=1024,
                tracker=tracker_config,
                project=output_dir,
                name="results",
                exist_ok=True,
                stream=True
            )
            
            # Collect predictions and ground truth
            predictions = {}  # frame_idx -> box
            ground_truth = {}  # frame_idx -> box
            frame_count = 0
            
            # Load ground truth
            for gt_file in Path(gt_dir).glob('*.txt'):
                frame_idx = int(gt_file.stem)  # Frame number from filename
                with open(gt_file) as f:
                    line = f.readline().strip()
                    if line:  # if file is not empty
                        cls_id, x, y, w, h = map(float, line.split())
                        if cls_id == 0:  # ball class
                            # Debug print
                            print(f"\nLoading GT frame {frame_idx}:")
                            print(f"Raw values: {cls_id}, {x}, {y}, {w}, {h}")
                            ground_truth[frame_idx] = np.array([x, y, w, h])
            
            # Process results
            for frame_idx, r in enumerate(results, 1):
                frame_count += 1
                if len(r.boxes) > 0:
                    ball_dets = r.boxes[r.boxes.cls == 0]
                    best_detection = get_best_detection(ball_dets)
                    if best_detection is not None:
                        box = best_detection.xywh[0].cpu().numpy()
                        # Debug print
                        print(f"\nPrediction frame {frame_idx}:")
                        print(f"Box values: {box}")
                        # Normalize coordinates if needed
                        x, y, w, h = box
                        predictions[frame_idx] = np.array([x/r.orig_shape[1], y/r.orig_shape[0], 
                                                         w/r.orig_shape[1], h/r.orig_shape[0]])
            
            # Evaluate predictions
            metrics = evaluate_predictions(predictions, ground_truth, frame_count)
            
            # Update cumulative metrics
            cumulative_metrics['total_frames'] += metrics['total_frames']
            cumulative_metrics['frames_with_gt'] += metrics['frames_with_gt']
            cumulative_metrics['frames_with_pred'] += metrics['frames_with_pred']
            cumulative_metrics['true_positives'] += metrics['true_positives']
            cumulative_metrics['false_positives'] += metrics['false_positives']
            cumulative_metrics['false_negatives'] += metrics['false_negatives']
            cumulative_metrics['total_iou'] += metrics.get('mean_iou', 0) * metrics['true_positives']
            cumulative_metrics['position_errors'].extend(metrics.get('position_errors', []))
            
            # Add frame details to metrics
            metrics['frames_analyzed'] = {
                'total_frames': frame_count,
                'frames_with_ball_gt': len(ground_truth),
                'frames_with_ball_pred': len(predictions),
                'frames_with_matches': metrics['true_positives']
            }
            
            # Save metrics
            metrics_path = os.path.join(output_dir, "results", "metrics.json")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"\nMetrics for {clip}:")
            print(f"Total Frames: {frame_count}")
            print(f"Frames with Ground Truth Ball: {len(ground_truth)}")
            print(f"Frames with Predicted Ball: {len(predictions)}")
            print(f"True Positives: {metrics['true_positives']}")
            print(f"False Positives: {metrics['false_positives']}")
            print(f"False Negatives: {metrics['false_negatives']}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1 Score: {metrics['f1_score']:.3f}")
            print(f"Mean IoU: {metrics['mean_iou']:.3f}")
            print(f"Position MSE: {metrics['position_mse']:.3f}")
            
        except Exception as e:
            print(f"ERROR processing {clip}: {e}")
            continue
            
        print(f"Completed processing {clip}")

    # Calculate final cumulative metrics
    if cumulative_metrics['true_positives'] > 0:
        cumulative_metrics['mean_iou'] = cumulative_metrics['total_iou'] / cumulative_metrics['true_positives']
        cumulative_metrics['position_mse'] = np.mean(cumulative_metrics['position_errors']) if cumulative_metrics['position_errors'] else 0

    # Calculate precision, recall, and F1 for cumulative metrics
    precision = cumulative_metrics['true_positives'] / (cumulative_metrics['true_positives'] + cumulative_metrics['false_positives']) if (cumulative_metrics['true_positives'] + cumulative_metrics['false_positives']) > 0 else 0
    recall = cumulative_metrics['true_positives'] / (cumulative_metrics['true_positives'] + cumulative_metrics['false_negatives']) if (cumulative_metrics['true_positives'] + cumulative_metrics['false_negatives']) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    cumulative_metrics.update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

    # Save cumulative metrics to a file
    summary_file_path = os.path.join(teamtrack_dir, "custom_video_results", "summary_metrics.txt")
    with open(summary_file_path, 'w') as f:
        for key, value in cumulative_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Summary metrics saved to {summary_file_path}")

if __name__ == "__main__":
    run_inference_on_videos()