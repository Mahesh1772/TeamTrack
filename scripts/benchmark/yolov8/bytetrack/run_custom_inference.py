from ultralytics import YOLO
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error

def calculate_iou(box1, box2):
    """Calculate IOU between two boxes in x,y,w,h format"""
    # For box1 (ground truth)
    box1_x1 = box1[0] - box1[2]/2
    box1_y1 = box1[1] - box1[3]/2
    box1_x2 = box1[0] + box1[2]/2
    box1_y2 = box1[1] + box1[3]/2
    
    # For box2 (prediction)
    if isinstance(box2, dict):
        box2 = box2['box']  # Extract numpy array from dict
    
    box2_x1 = box2[0] - box2[2]/2
    box2_y1 = box2[1] - box2[3]/2
    box2_x2 = box2[0] + box2[2]/2
    box2_y2 = box2[1] + box2[3]/2
    
    # Calculate intersection
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_predictions(predictions, ground_truth, frame_count, tracks):
    """Calculate evaluation metrics including tracking information"""
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
    
    # Add trajectory analysis
    trajectory_metrics = analyze_trajectories(tracks, frame_count)
    metrics['trajectory'] = trajectory_metrics
    
    # Add tracking-specific metrics
    metrics['tracking'] = {
        'track_switches': count_track_switches(predictions),
        'track_fragmentations': count_track_fragmentations(tracks),
        'tracking_consistency': calculate_tracking_consistency(tracks)
    }
    
    return metrics

def get_best_detection(detections, confidence_threshold=0.1):
    """Get the highest confidence detection above threshold"""
    if len(detections) == 0:
        return None
        
    # Filter by confidence threshold
    valid_dets = detections[detections.conf >= confidence_threshold]
    if len(valid_dets) == 0:
        return None
    
    # Get highest confidence detection
    best_idx = valid_dets.conf.argmax()
    best_det = valid_dets[best_idx]
    return best_det

def process_frame_results(r, frame_idx, frame_shape):
    """Process single frame results and return detection info"""
    if len(r.boxes) > 0:
        ball_dets = r.boxes[r.boxes.cls == 0]
        best_detection = get_best_detection(ball_dets)
        if best_detection is not None:
            box = best_detection.xywh[0].cpu().numpy()
            track_id = best_detection.id.cpu().numpy()[0] if best_detection.id is not None else None
            x, y, w, h = box
            return {
                'box': np.array([x/frame_shape[1], y/frame_shape[0], 
                               w/frame_shape[1], h/frame_shape[0]]),
                'track_id': track_id,
                'confidence': best_detection.conf.cpu().numpy()[0]
            }
    return None

def analyze_trajectories(tracks, frame_count):
    """Analyze tracking trajectories"""
    trajectory_metrics = {
        'num_tracks': len(tracks),
        'track_lengths': [],
        'track_gaps': [],
        'track_velocities': []
    }
    
    for track_id, detections in tracks.items():
        # Sort detections by frame number
        sorted_dets = sorted(detections, key=lambda x: x['frame'])
        
        # Calculate track length
        track_length = len(sorted_dets)
        trajectory_metrics['track_lengths'].append(track_length)
        
        # Calculate gaps in tracking
        frame_numbers = [d['frame'] for d in sorted_dets]
        gaps = [frame_numbers[i+1] - frame_numbers[i] - 1 for i in range(len(frame_numbers)-1)]
        if gaps:
            trajectory_metrics['track_gaps'].extend(gaps)
        
        # Calculate velocities between consecutive frames
        for i in range(len(sorted_dets)-1):
            pos1 = sorted_dets[i]['box'][:2]
            pos2 = sorted_dets[i+1]['box'][:2]
            velocity = np.linalg.norm(pos2 - pos1)
            trajectory_metrics['track_velocities'].append(velocity)
    
    # Calculate summary statistics
    trajectory_metrics.update({
        'avg_track_length': np.mean(trajectory_metrics['track_lengths']),
        'avg_track_gap': np.mean(trajectory_metrics['track_gaps']) if trajectory_metrics['track_gaps'] else 0,
        'avg_velocity': np.mean(trajectory_metrics['track_velocities']) if trajectory_metrics['track_velocities'] else 0,
        'tracking_coverage': sum(trajectory_metrics['track_lengths']) / frame_count
    })
    
    return trajectory_metrics

def count_track_switches(predictions):
    """Count number of times tracking switches between different track IDs"""
    switches = 0
    prev_track_id = None
    
    # Sort predictions by frame number
    sorted_frames = sorted(predictions.keys())
    
    for frame_idx in sorted_frames:
        if 'track_id' in predictions[frame_idx]:
            current_track_id = predictions[frame_idx]['track_id']
            if prev_track_id is not None and current_track_id != prev_track_id:
                switches += 1
            prev_track_id = current_track_id
    
    return switches

def count_track_fragmentations(tracks):
    """Count number of track fragmentations (gaps in tracking)"""
    fragmentations = 0
    
    for track_id, detections in tracks.items():
        # Sort detections by frame number
        sorted_dets = sorted(detections, key=lambda x: x['frame'])
        
        # Check for gaps in frame numbers
        for i in range(len(sorted_dets) - 1):
            if sorted_dets[i+1]['frame'] - sorted_dets[i]['frame'] > 1:
                fragmentations += 1
    
    return fragmentations

def calculate_tracking_consistency(tracks):
    """Calculate tracking consistency score based on trajectory smoothness"""
    if not tracks:
        return 0.0
        
    consistency_scores = []
    
    for track_id, detections in tracks.items():
        if len(detections) < 3:  # Need at least 3 points for consistency check
            continue
            
        # Sort detections by frame number
        sorted_dets = sorted(detections, key=lambda x: x['frame'])
        
        # Calculate consistency based on velocity changes
        velocities = []
        for i in range(len(sorted_dets) - 1):
            pos1 = sorted_dets[i]['box'][:2]
            pos2 = sorted_dets[i+1]['box'][:2]
            velocity = np.linalg.norm(pos2 - pos1)
            velocities.append(velocity)
        
        # Calculate velocity consistency (lower variance = more consistent)
        if len(velocities) > 1:
            velocity_std = np.std(velocities)
            velocity_mean = np.mean(velocities)
            consistency = 1.0 / (1.0 + velocity_std/velocity_mean) if velocity_mean > 0 else 0
            consistency_scores.append(consistency)
    
    # Return average consistency across all tracks
    return np.mean(consistency_scores) if consistency_scores else 0.0

def run_inference_on_videos():
    # Get absolute path to TeamTrack directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    teamtrack_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    
    # Base directories
    base_dir = r"C:\Users\Demo-user\Documents\Gameplay"
    video_base_dir = os.path.join(base_dir, "videos", "game1")
    gt_base_dir = os.path.join(base_dir, "labels", "game1")
    
    # Process all available clips
    clips = []
    for i in range(1, 13):  # Clips 1-12
        clip_name = f'Clip{i}'
        clip_path = os.path.join(video_base_dir, clip_name, f"{clip_name}-1.mp4")
        if os.path.exists(clip_path):
            clips.append(clip_name)
            print(f"Found video: {clip_path}")
    
    if not clips:
        print("No video clips found!")
        return
        
    print(f"Will process these clips: {clips}")
    
    # Load the model
    model_path = os.path.join(teamtrack_dir, "yolov8/Handball_Sideview_trained30/weights/best.pt")
    model = YOLO(model_path)
    
    # Verify model loaded correctly
    print(f"Model info:")
    print(f"- Task: {model.task}")
    print(f"- Names: {model.names}")
    
    tracker_config = os.path.join(current_dir, "tracker_config.yaml")
    
    print(f"Model path exists: {os.path.exists(model_path)}")
    print(f"Model path: {model_path}")
    
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
        video_path = os.path.join(video_base_dir, clip, f"{clip}-1.mp4")
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
            # Verify video can be read
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"ERROR: Could not open video: {video_path}")
                continue
            
            ret, test_frame = cap.read()
            if not ret:
                print(f"ERROR: Could not read frame from video: {video_path}")
                continue
            
            print(f"Video dimensions: {test_frame.shape}")
            
            # Test detection on first frame
            test_results = model.predict(test_frame, conf=0.1)
            print(f"Test frame detections: {len(test_results[0].boxes)}")
            
            cap.release()
            
            # Setup video writer
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_path = os.path.join(output_dir, "results", Path(video_path).name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            # Run inference with lower confidence threshold
            results = model.track(
                source=video_path,
                save=False,  # We'll save manually
                save_txt=True,
                save_conf=True,
                conf=0.1,  # Lower confidence threshold for ByteTrack
                iou=0.3,   # Lower IOU threshold
                imgsz=1024,
                tracker=tracker_config,
                project=output_dir,
                name="results",
                exist_ok=True,
                stream=True,
                verbose=False
            )
            
            # Collect predictions and ground truth
            predictions = {}  # frame_idx -> detection info
            tracks = {}      # track_id -> list of frame detections
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
            frame_count = 0
            detection_count = 0
            
            for frame_idx, r in enumerate(results, 1):
                frame_count += 1
                orig_img = r.orig_img
                
                if len(r.boxes) > 0:
                    detection = process_frame_results(r, frame_idx, r.orig_shape)
                    if detection:
                        detection_count += 1
                        if frame_count % 50 == 0:  # Print every 50th frame
                            print(f"\nFrame {frame_count}:")
                            print(f"Detection: {detection}")
                        
                        box = detection['box']
                        track_id = detection['track_id']
                        conf = detection['confidence']
                        
                        # Store normalized coordinates
                        predictions[frame_idx] = box
                        
                        # Store track information
                        if track_id is not None:
                            if track_id not in tracks:
                                tracks[track_id] = []
                            tracks[track_id].append({
                                'frame': frame_idx,
                                'box': box,
                                'conf': conf
                            })
                        
                        # Draw detection
                        x, y, w, h = box * np.array([r.orig_shape[1], r.orig_shape[0], 
                                                   r.orig_shape[1], r.orig_shape[0]])
                        x1 = int(x - w/2)
                        y1 = int(y - h/2)
                        x2 = int(x + w/2)
                        y2 = int(y + h/2)
                        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'ball {conf:.2f}'
                        if track_id is not None:
                            label += f' id:{track_id}'
                        cv2.putText(orig_img, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                writer.write(orig_img)
            
            print(f"\nProcessing complete for {clip}:")
            print(f"Total frames processed: {frame_count}")
            print(f"Frames with detections: {detection_count}")
            print(f"Detection rate: {detection_count/frame_count*100:.1f}%")
            
            if len(ground_truth) == 0:
                print(f"WARNING: No ground truth found for {clip}")
            else:
                print(f"Ground truth frames: {len(ground_truth)}")
            
            # Evaluate predictions
            metrics = evaluate_predictions(predictions, ground_truth, frame_count, tracks)
            
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
            print(f"True Positives: {metrics['true_positives']}")
            print(f"False Positives: {metrics['false_positives']}")
            print(f"False Negatives: {metrics['false_negatives']}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1 Score: {metrics['f1_score']:.3f}")
            
            if 'trajectory' in metrics:
                traj = metrics['trajectory']
                print(f"\nTrajectory Analysis:")
                print(f"Number of tracks: {traj['num_tracks']}")
                print(f"Average track length: {traj['avg_track_length']:.1f} frames")
                print(f"Tracking coverage: {traj['tracking_coverage']*100:.1f}%")
            
            if 'tracking' in metrics:
                track = metrics['tracking']
                print(f"\nTracking Metrics:")
                print(f"Track switches: {track['track_switches']}")
                print(f"Track fragmentations: {track['track_fragmentations']}")
                print(f"Tracking consistency: {track['tracking_consistency']:.3f}")
            
            # After processing all frames, add:
            writer.release()
            cap.release()
            
        except Exception as e:
            print(f"ERROR processing {clip}:")
            import traceback
            traceback.print_exc()
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