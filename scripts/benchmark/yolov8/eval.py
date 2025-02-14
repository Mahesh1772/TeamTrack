import os
import argparse
from ultralytics import YOLO
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8/Handball_Sideview_trained18/weights/best.pt')
    parser.add_argument('--data', type=str, default='data/teamtrack-yolo/Handball_SideView/data.yaml')
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--split', type=str, default='test')
    return parser.parse_args()

def main(args):
    print(f"Evaluating model: {args.model}")
    print(f"Using data: {args.data}")
    
    # Load model
    model = YOLO(args.model)
    
    # Run validation
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=True
    )
    
    metrics = results.box
    class_names = ['player', 'ball']
    
    print("\nDetailed Performance Metrics:")
    print(f"{'Class':<10} {'P':<8} {'R':<8} {'F1':<8} {'mAP50':<8} {'mAP50-95':<8}")
    print("-" * 58)
    
    # Overall metrics using mean_results
    mean_p, mean_r, mean_map50, mean_map = metrics.mean_results()
    print(f"{'all':<10} {mean_p:.3f}    {mean_r:.3f}    {metrics.f1.mean():.3f}    {mean_map50:.3f}    {mean_map:.3f}")
    
    # Per-class metrics
    for i, name in enumerate(class_names):
        p, r, map50, map = metrics.class_result(i)
        print(f"{name:<10} {p:.3f}    {r:.3f}    {metrics.f1[i]:.3f}    {map50:.3f}    {map:.3f}")
    
    # Save metrics to JSON
    output_dir = Path('yolov8/eval_results')
    output_dir.mkdir(exist_ok=True)
    
    maps = metrics.maps  # Get the maps array directly
    
    metrics_dict = {
        'summary': {
            'mAP50': float(mean_map50),
            'mAP50-95': float(mean_map),
            'precision': float(mean_p),
            'recall': float(mean_r),
            'f1': float(metrics.f1.mean())
        },
        'per_class': {
            class_name: {
                'precision': float(metrics.p[i]),
                'recall': float(metrics.r[i]),
                'f1': float(metrics.f1[i]),
                'mAP50': float(maps[i]),  # Use the array directly
                'mAP50-95': float(metrics.class_result(i)[3])
            } for i, class_name in enumerate(class_names)
        },
        'speed': results.speed
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    args = parse_args()
    main(args) 