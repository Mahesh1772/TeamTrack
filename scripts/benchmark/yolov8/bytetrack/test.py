import argparse
import os
import tempfile
from pathlib import Path

import trackeval
import numpy as np
from sportslabkit.logger import logger
from teamtrack.utils import (load_yaml, make_image_paths, make_yaml,
                             parse_args, run_trackeval, save_results, visualize_results)
from ultralytics import YOLO


def main(cfg):
    logger.info(cfg)
    tracker_name = cfg.TRACKER.NAME
    
    # List of all clips to process
    clips = [f'clip{i}' for i in range(4, 13)]  # clips 4 through 12
    
    for clip in clips:
        logger.info(f"\nProcessing {clip}")
        
        # Update output directory to include clip name
        output_dir = os.path.join(cfg.OUTPUT.ROOT, 
                                cfg.DATASET.NAME + f"-{cfg.DATASET.SUBSET}", 
                                tracker_name,
                                clip)
        
        if os.path.exists(output_dir) and not cfg.OUTPUT.OVERWRITE:
            logger.warning(f"Output directory {output_dir} already exists, skipping")
            continue

        model = YOLO(cfg.TRACKER.YOLOV8.MODEL_PATH)
        logger.info(model)

        # Update image directory path for current clip
        image_dir = os.path.join(
            cfg.DATASET.ROOT, 
            cfg.DATASET.NAME, 
            cfg.DATASET.SUBSET,
            cfg.DATASET.GAME,
            clip,
            "images"
        )
        logger.info(f"Tracking images in {image_dir}")
        
        parameters = {key.lower():value for key, value in dict(cfg.TRACKER.YOLOV8).items()}
        d = load_yaml(cfg.TRACKER.YOLOV8.CONFIGURATION_PATH)
        parameters.update(d)
        logger.info(parameters)
        
        # Process the image sequence
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_yaml = make_yaml(tmpdir, parameters)
            results = model.track(
                source=image_dir,
                stream=True,
                tracker=tmp_yaml,
                imgsz=parameters["imgsz"],
                vid_stride=parameters["vid_stride"],
                conf=parameters["conf"],
            )
            
            # Save results for the image sequence
            output_path = os.path.join(output_dir, "data", f"{clip}_sequence.txt")
            save_results(results, output_path)

        if cfg.OUTPUT.SAVE_VIDEO:
            # Create a video from tracking results
            text_path = os.path.join(output_dir, "data", f"{clip}_sequence.txt")
            save_path = os.path.join(output_dir, "videos", f"{clip}_sequence.mp4")
            logger.info(f"Visualizing tracking results to {save_path}")
            visualize_results(image_dir, text_path, save_path, is_image_sequence=True)

        if cfg.OUTPUT.SAVE_EVAL:
            eval_config = trackeval.Evaluator.get_default_eval_config()
            dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
            metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
            eval_config.update(dict(cfg.TRACKEVAL.EVAL))
            dataset_config.update(dict(cfg.TRACKEVAL.DATASET))
            metrics_config.update(dict(cfg.TRACKEVAL.METRICS))
            output_res, output_msg = run_trackeval(eval_config, dataset_config, metrics_config)
                
    output_dir = cfg.OUTPUT.ROOT

if __name__ == "__main__":
    main(parse_args())
