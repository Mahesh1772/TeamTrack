from ultralytics import YOLO
import os
import cv2
import numpy as np

def is_in_court(box, frame_height, frame_width):
    """Filter detections to only include players on the court"""
    x1, y1, x2, y2 = box
    
    # Define court boundaries (adjusted for your specific video)
    court_top = frame_height * 0.35     # Increased top margin to exclude audience
    court_bottom = frame_height * 0.85   # Adjusted bottom margin
    court_left = frame_width * 0.1      # Increased left margin
    court_right = frame_width * 0.9     # Adjusted right margin
    
    # Get box dimensions
    box_height = y2 - y1
    box_width = x2 - x1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Additional filters
    min_player_height = frame_height * 0.1  # Minimum expected player height
    max_player_height = frame_height * 0.3  # Maximum expected player height
    
    # Check all conditions
    is_valid_size = min_player_height < box_height < max_player_height
    is_in_bounds = (court_left < center_x < court_right and 
                   court_top < center_y < court_bottom)
    
    return is_valid_size and is_in_bounds

def run_inference_on_video():
    # Video path
    video_path = r"C:\Users\Demo-user\Documents\Gameplay\Clip4\Clip4-1.mp4"
    
    # Load the trained model
    model_path = "yolov8/Handball_Sideview_trained18/weights/best.pt"
    model = YOLO(model_path)
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Output directory
    output_dir = "custom_video_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tracker config path
    tracker_config = "scripts/benchmark/yolov8/bytetrack/tracker_config.yaml"
    
    # Run inference with tracking
    results = model.track(
        source=video_path,
        save=True,              # Save video
        save_txt=True,          # Save tracking results in MOT format
        save_conf=True,         # Save confidence scores
        conf=0.3,              # Increased confidence threshold
        iou=0.5,              # Increased IOU threshold
        imgsz=1024,            # Image size
        tracker=tracker_config, # Use tracker config file
        project=output_dir,     # Save results to custom_video_results
        name="clip4_results",   # Subfolder name
        stream=True            # Stream inference to prevent RAM issues
    )
    
    # Process results
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        if boxes and boxes.id is not None:
            # Filter boxes to only include court players
            valid_boxes = [box for box in boxes if is_in_court(box.xyxy[0], frame_height, frame_width)]
            print(f"Frame tracked. Found {len(valid_boxes)} players on court")
    
    print(f"Results saved to {output_dir}/clip4_results")

if __name__ == "__main__":
    run_inference_on_video()