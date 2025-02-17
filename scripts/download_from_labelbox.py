"""Script to download data from Labelbox.
"""

import argparse
import os
from pathlib import Path

import ndjson
import requests
from dotenv import load_dotenv
from labelbox import Client

from sportslabkit.dataframe import BBoxDataFrame
from sportslabkit.logger import tqdm, logger

load_dotenv()

project_name_id_map = {
    "D_Soccer_Tsukuba": "clahkbqgx1hae09wf4wc88oou",
    "F_Soccer_Tsukuba3": "cldmd6hrx08ej07yt2rydfine",
    "F_Handball_Nagoya_undistorted": "cl9w34kua0xv808tw3jayanqi",
    "D_Basketball_China": "cl561fensmydl071632q4hxm6",
    "F_Basketball_Tokai": "cl8s58o8p1b8i073z0lk8hocv",
}

LABELBOX_API_KEY = os.getenv("LABELBOX_API_KEY")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--project_name",
    help="The project ID to download from. Default to downloading all data",
    default="all",
)
parser.add_argument(
    "--save_root", help="The path to save the downloaded data to.", default="./data"
)
parser.add_argument(
    "--make_viz",
    help="Whether to make visualizations of the data.",
    action="store_true",
)
parser.add_argument(
    "--ignore_errors", help="Whether to ignore errors.", action="store_true"
)
parser.add_argument(
    "--overwrite", help="Whether to overwrite existing data.", action="store_true"
)
args = parser.parse_args()


def download_video(video_url: str, save_path: str):
    """Download a video from Labelbox.

    Args:
        video_url (str): The URL to download the video from.
        save_path (str): The path to save the video to.
    """
    with open(save_path, "wb") as file:
        file.write(requests.get(video_url).content)


def download_annotations(annotations_url: str, save_path: str):
    """Download annotations from Labelbox.

    Args:
        annotations_url (str): The URL to download the annotations from.
        save_path (str): The path to save the annotations to.
    """
    headers = {"Authorization": f"Bearer {LABELBOX_API_KEY}"}
    annotations = ndjson.loads(requests.get(annotations_url, headers=headers).text)
    home_team_key = "0"
    away_team_key = "1"
    ball_key = "BALL"

    d = {home_team_key: {}, away_team_key: {}, ball_key: {}}

    for annotation in annotations:
        for frame_annotation in annotation["objects"]:
            frame_number = annotation["frameNumber"]
            bbox = frame_annotation["bbox"]

            if frame_annotation["title"] == ball_key:
                team_id = ball_key
                player_id = ball_key
            else:
                team_id, player_id = frame_annotation["title"].split("_")

            if team_id not in [home_team_key, away_team_key, ball_key]:
                continue
            
            if d[team_id].get(player_id) is None:
                d[team_id][player_id] = {}
            d[team_id][player_id][frame_number] = [
                bbox["left"],
                bbox["top"],
                bbox["width"],
                bbox["height"],
            ]

    bbdf = BBoxDataFrame.from_dict(d)
    bbdf.to_csv(save_path)
    return bbdf


if __name__ == "__main__":
    save_root = Path(args.save_root)

    if args.project_name == "all":
        project_names = project_name_id_map.keys()
        project_ids = project_name_id_map.values()
    else:
        project_names = [args.project_name]
        project_ids = [project_name_id_map[args.project_name]]

    for project_name, project_id in zip(project_names, project_ids):
        # Set up the client.
        client = Client(api_key=LABELBOX_API_KEY)
        project = client.get_project(project_id)

        export_url = project.export_labels()
        exports = requests.get(export_url, timeout=60).json()

        for export_data in (pbar := tqdm(exports)):
            external_id = Path(export_data["External ID"])
            pbar.set_description(f"Downloading {external_id}")

            video_url = export_data["Labeled Data"]
            annotations_url = export_data["Label"].get("frames")
            if annotations_url is None:
                logger.info(f"No annotations for {external_id}")
                continue

            save_dir = save_root / project_name
            mp4_save_path = save_dir / "videos" / external_id.with_suffix(".mp4")
            csv_save_path = save_dir / "annotations" / external_id.with_suffix(".csv")
            viz_save_path = save_dir / "viz_results" / external_id.with_suffix(".mp4")

            mp4_save_path.parent.mkdir(parents=True, exist_ok=True)
            csv_save_path.parent.mkdir(parents=True, exist_ok=True)
            viz_save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                if mp4_save_path.exists() and not args.overwrite:
                    logger.info(f"Skipping {external_id} in {project_name}")
                    continue
                download_video(video_url, mp4_save_path)
                bbdf = download_annotations(annotations_url, csv_save_path)

                if args.make_viz:
                    bbdf.visualize_frames(mp4_save_path, viz_save_path)
            except Exception as e:
                if args.ignore_errors:
                    logger.info(
                        f"Error downloading {external_id} in {project_name}: {e}"
                    )
                else:
                    raise e
