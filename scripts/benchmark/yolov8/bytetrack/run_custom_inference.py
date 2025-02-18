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
    # Get absolute path to TeamTrack directory
    current_dir = os.path.dirname(os.path.abspath(__file__))  # bytetrack directory
    teamtrack_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    
    # Create output directory
    output_dir = os.path.join(teamtrack_dir, "custom_video_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory at: {output_dir}")
    
    # Video path
    video_path = r"C:\Users\Demo-user\Documents\Gameplay\Clip4\Clip4-1.mp4"
    
    # Load the trained model
    model_path = os.path.join(teamtrack_dir, "yolov8/Handball_Sideview_trained21/weights/best.pt")
    model = YOLO(model_path)
    
    # Tracker config path
    tracker_config = os.path.join(current_dir, "tracker_config.yaml")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
        
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at {video_path}")
        return
        
    print(f"Using model: {model_path}")
    print(f"Processing video: {video_path}")
    
    try:
        # Run inference with tracking
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
            name="clip4_results",   
            exist_ok=True,         
            stream=True            
        )
        
        # Force processing of results
        for r in results:
            pass  # Process each frame
            
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return

if __name__ == "__main__":
    run_inference_on_video()