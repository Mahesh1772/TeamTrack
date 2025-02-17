import os
import argparse
from glob import glob
import configparser
import cv2
import sportslabkit
import numpy as np
from joblib import Parallel, delayed


def csv2txt(csv_file, text_file):
    print(f"Converting {csv_file} to {text_file}")
    
    # Load data
    df = sportslabkit.load_df(csv_file)
    
    # Extract ball data before conversion
    ball_cols = [col for col in df.columns if 'BALL' in str(col)]
    ball_data = df[ball_cols].dropna()
    
    # Store ball positions for matching
    ball_positions = {}  # frame -> (left, top, width, height)
    for frame in ball_data.index:
        ball_positions[frame] = (
            ball_data.loc[frame, ('BALL', 'BALL', 'bb_left')],
            ball_data.loc[frame, ('BALL', 'BALL', 'bb_top')],
            ball_data.loc[frame, ('BALL', 'BALL', 'bb_width')],
            ball_data.loc[frame, ('BALL', 'BALL', 'bb_height')]
        )
    
    # Convert to MOT format
    bbdf = df.to_mot_format().dropna()
    
    # Initialize class_id column with default player class
    bbdf['class_id'] = 1
    
    # Set class_id=2 for ball detections by matching positions
    for idx, row in bbdf.iterrows():
        frame = int(row['frame'])
        if frame in ball_positions:
            ball_pos = ball_positions[frame]
            # Match ball position (with small tolerance for float comparison)
            if (abs(row['bb_left'] - ball_pos[0]) < 0.1 and 
                abs(row['bb_top'] - ball_pos[1]) < 0.1 and 
                abs(row['bb_width'] - ball_pos[2]) < 0.1 and 
                abs(row['bb_height'] - ball_pos[3]) < 0.1):
                bbdf.loc[idx, 'class_id'] = 2
    
    # Set confidence to 1.0 for all detections (ground truth)
    bbdf['conf'] = 1.0
    
    # Verify we have both classes
    n_players = (bbdf['class_id'] == 1).sum()
    n_balls = (bbdf['class_id'] == 2).sum()
    print(f"Found {n_players} player detections and {n_balls} ball detections in {csv_file}")
    
    # Select and order columns correctly for MOT format
    mot_cols = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'class_id', 'conf']
    bbdf = bbdf[mot_cols]
    
    # Add -1,-1 columns
    bbdf['-1_1'] = -1
    bbdf['-1_2'] = -1
    
    # Ensure numeric format and save
    arr = bbdf.values.astype(float)
    np.savetxt(text_file, arr, fmt='%d', delimiter=',')

def process_csv_file(csv_file, output_dir, dataset_name, subset, subset_dir):
    # Prepare sequence directory structure
    seq_name = os.path.splitext(os.path.basename(csv_file))[0]
    seq_dir = os.path.join(output_dir, f"{dataset_name}-{subset}", seq_name)
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "gt"), exist_ok=True)

    txt_file = os.path.join(seq_dir, "gt", "gt.txt")
    csv2txt(csv_file, txt_file)

    # Create seqinfo.ini file
    mp4_file = os.path.join(subset_dir, "videos", f"{seq_name}.mp4")
    assert os.path.exists(mp4_file), f"Video file {mp4_file} does not exist."

    # Extract video properties
    video = cv2.VideoCapture(mp4_file)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    seq_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    im_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Save frames as images
    img_dir = os.path.join(seq_dir, "img1")
    os.makedirs(img_dir, exist_ok=True)

    current_frame = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        img_name = f"{current_frame:06d}.jpg"  # create a name of each image: 000001.jpg, 000002.jpg and so on
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, frame)
        current_frame += 1

    video.release()

    config = configparser.ConfigParser()
    config["Sequence"] = {
        "name": seq_name,
        "imDir": "img1",
        "frameRate": frame_rate,
        "seqLength": seq_length,
        "imWidth": im_width,
        "imHeight": im_height,
        "imExt": ".jpg",
    }
    seq_info_file = os.path.join(seq_dir, "seqinfo.ini")
    with open(seq_info_file, "w", encoding="utf-8") as configfile:
        config.write(configfile)

    return seq_dir

def parse_args():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Convert TeamTrack data to MOTChallenge format.")
    parser.add_argument("--teamtrack-dir", type=str, default="./datasets/teamtrack", help="Path to the TeamTrack dataset.")
    parser.add_argument("--output-dir", type=str, default="./datasets/teamtrack-mot", help="Output directory for the converted dataset.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of jobs to run in parallel.")

    return parser.parse_args()


def main(args):
    # Directories
    teamtrack_dir = args.teamtrack_dir
    output_dir = args.output_dir

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each dataset
    dataset_names = [
        #"Basketball_SideView",
        #"Basketball_SideView2",
        #"Basketball_TopView",
        "Handball_SideView",
        #"Soccer_SideView",
        #"Soccer_TopView",
    ]
    for dataset_name in dataset_names:
        dataset_dir = os.path.join(teamtrack_dir, dataset_name)
        assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist."

        dirs = {
            "train": [],
            "val": [],
            "test": [],
        }
        # Create directories
        for subset in ["train", "test", "val"]:
            subset_dir = os.path.join(dataset_dir, subset)
            assert os.path.exists(subset_dir), f"Subset directory {subset_dir} does not exist."

            csv_files = glob(os.path.join(subset_dir, "annotations", "*.csv"))
            dirs[subset] = Parallel(n_jobs=args.n_jobs)(delayed(process_csv_file)(csv_file, output_dir, dataset_name, subset, subset_dir) for csv_file in csv_files)

        dirs["all"] = dirs["train"] + dirs["test"] + dirs["val"]
        dirs["trainval"] = dirs["train"] + dirs["val"]

        # Create sequence files
        seqmaps_dir = os.path.join(output_dir, "seqmaps")
        os.makedirs(seqmaps_dir, exist_ok=True)

        for subset in ["train", "test", "val", "trainval", "all"]:
            seq_file_path = os.path.join(seqmaps_dir, f"{dataset_name}-{subset}.txt")
            with open(seq_file_path, "w", encoding="utf-8") as file:
                print(f"Writing {seq_file_path}")

                seq_names = []
                for seq_path in dirs[subset]:
                    seq_name = os.path.basename(seq_path)
                    seq_names.append(seq_name)
                file.write("\n".join(["names"] + sorted(seq_names)))


if __name__ == "__main__":
    main(parse_args())

# python scripts/convert_teamtrack_to_mot.py --teamtrack-dir ./data/teamtrack-nosplit --output-dir ./data/teamtrack-mot --n-jobs -1
