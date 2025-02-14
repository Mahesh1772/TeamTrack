import os
import argparse
import torch
from ultralytics import YOLO

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_SILENT'] = 'true'

# More aggressive memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/teamtrack-yolo/Handball_SideView/data.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--classes', type=int, nargs='+', default=None)
    return parser.parse_args()

def main(args):
    # Aggressive memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print(f"Starting training with:")
    print(f"- Data: {args.data}")
    print(f"- Image size: {args.imgsz}")
    print(f"- Batch size: {args.batch}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Workers: {args.workers}")
    print(f"- Classes: All")
    
    # Initialize model
    model = YOLO('models/yolov8/Handball_Sideview.pt')
    
    # Train with memory-optimized parameters
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project='yolov8',
        name='Handball_Sideview_trained',
        device=args.device,
        save_period=1,
        plots=True,
        cache=False,
        close_mosaic=10,
        workers=args.workers,
        amp=True,
        half=True,
        single_cls=False,
        rect=True,
        optimizer='AdamW',
        nbs=64,
        overlap_mask=False,
        patience=50
    )
    
    # Copy results
    os.makedirs('models/yolov8', exist_ok=True)
    import shutil
    shutil.copytree('yolov8/Handball_Sideview_trained', 
                    'models/yolov8/Handball_Sideview_trained', 
                    dirs_exist_ok=True)

if __name__ == '__main__':
    args = parse_args()
    main(args) 