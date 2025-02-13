import sportslabkit
import pandas as pd

# Load a single CSV file
csv_file = "./data/teamtrack/Handball_SideView/test/annotations/2nd_fisheye_90-120.csv"

# Load and print raw data
df = sportslabkit.load_df(csv_file)

# Print ball data specifically
print("\nBall data:")
ball_cols = [col for col in df.columns if 'BALL' in str(col)]
print(df[ball_cols].head())

# Convert to MOT format
bbdf = df.to_mot_format()

# Add class_id column (1 for players, 2 for ball)
# We need to track which rows came from the ball columns
ball_data = df[ball_cols].dropna()
ball_frames = ball_data.index

print("\nBall frames:", ball_frames[:5])
print("\nTotal frames with ball:", len(ball_frames))

# Print sample of converted data
print("\nSample converted data:")
print(bbdf.head()) 