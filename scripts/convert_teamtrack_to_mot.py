import os
import argparse
from glob import glob
import configparser
import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def load_df(csv_file):
    """Load CSV file into DataFrame"""
    # Read with the first 3 rows as headers to understand the structure
    df = pd.read_csv(csv_file, header=[0,1,2])
    return df

def to_mot_format(df, sport):
    """Convert DataFrame to MOT format"""
    # Create empty DataFrame for MOT format
    mot_df = pd.DataFrame()
    
    # Get all unique combinations of TeamID and PlayerID
    team_player_cols = [(team, player) for team, player in zip(df.columns.get_level_values(0), df.columns.get_level_values(1))]
    
    rows = []
    # Iterate through each frame
    for frame_idx, row in df.iterrows():
        # For each team/player combination
        for (team, player) in team_player_cols:
            # Skip if it's not a valid player column (e.g., might be frame number column)
            if player not in ['0','1','2','3','4','5','6','7','8','9','10','11','BALL']:
                continue
                
            # Get bounding box values
            bb_height = row[(team, player, 'bb_height')]
            bb_left = row[(team, player, 'bb_left')]
            bb_top = row[(team, player, 'bb_top')]
            bb_width = row[(team, player, 'bb_width')]
            
            # Skip if any value is NaN
            if pd.isna([bb_height, bb_left, bb_top, bb_width]).any():
                continue
                
            # Create MOT format row
            mot_row = {
                'frame': frame_idx + 1,  # MOT format uses 1-based frame numbers
                'id': int(f"{team}{player}" if player != "BALL" else "-1"),  # Create unique ID combining team and player
                'bb_left': bb_left,
                'bb_top': bb_top,
                'bb_width': bb_width,
                'bb_height': bb_height,
                'conf': 1,  # confidence score
                'x': -1,    # 3D position (not used)
                'y': -1,
                'z': -1
            }
            rows.append(mot_row)
    
    # Create final DataFrame
    mot_df = pd.DataFrame(rows)
    return mot_df

def csv2txt(csv_file, text_file):
    print(f"Converting {csv_file} to {text_file}")
    sport = csv_file.split(os.sep)[-4].split("_")[0].lower()
    df = load_df(csv_file)
    bbdf = to_mot_format(df, sport).dropna()
    arr = bbdf.values
    np.savetxt(text_file, arr, fmt="%d", delimiter=",")

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

    # Modified to match your folder name
    dataset_names = [
        "handball_side",  # Changed from "Handball_SideView" to match your structure
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
