"""
Utility: Convert per-frame CSV predictions to competition JSON format.

This script converts frame-level one-hot encoded CSV outputs into the
temporal event JSON format required by the ICPR 2026 RARE-VISION competition.

Usage:
    python utils/make_json.py --csv_dir results/ --output results/submission.json

The JSON format groups contiguous frames with identical active label sets
into temporal events:
{
  "videos": [
    {
      "video_id": "ukdd_navi_00051",
      "events": [
        {"start": 0, "end": 1500, "label": ["stomach"]},
        {"start": 1501, "end": 2000, "label": ["stomach", "erosion"]},
        ...
      ]
    }
  ]
}
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from tqdm import tqdm


USED_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve",
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]


def df_to_events(df, video_id, label_columns, index_col="index"):
    """Convert a DataFrame of frame-level predictions to grouped events."""
    df = df.sort_values(index_col).reset_index(drop=True)
    df[index_col] = df[index_col].astype(int)

    def active_labels(row):
        return tuple(sorted(
            lbl for lbl in label_columns if lbl in row and row[lbl] == 1
        ))

    df["active"] = df.apply(active_labels, axis=1)

    events = []
    if df.empty:
        return {"video_id": video_id, "events": []}

    current_labels = df.loc[0, "active"]
    start_idx = int(df.loc[0, index_col])

    for i in range(1, len(df)):
        idx = int(df.loc[i, index_col])
        labels = df.loc[i, "active"]
        if labels != current_labels:
            events.append({
                "start": start_idx,
                "end": idx - 1,
                "label": list(current_labels),
            })
            start_idx = idx
            current_labels = labels

    last_idx = int(df.loc[len(df) - 1, index_col])
    events.append({
        "start": start_idx,
        "end": last_idx,
        "label": list(current_labels),
    })

    return {"video_id": video_id, "events": events}


def build_json_from_csvs(csv_dir, output_json, prefix="test_"):
    """Build competition JSON from per-video frame prediction CSVs."""
    csv_path = Path(csv_dir)
    videos = []

    csv_files = sorted(csv_path.glob(f"{prefix}*_frames.csv"))
    if not csv_files:
        csv_files = sorted(csv_path.glob("*.csv"))

    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        # Extract video ID from filename
        stem = csv_file.stem
        if stem.startswith(prefix):
            video_id = stem[len(prefix):].replace("_frames", "")
        else:
            video_id = stem

        df = pd.read_csv(csv_file)
        label_cols = [c for c in USED_LABELS if c in df.columns]

        # Determine index column
        idx_col = "index"
        if idx_col not in df.columns:
            for candidate in ["frame_number", "frame", "frame_no"]:
                if candidate in df.columns:
                    idx_col = candidate
                    break

        video_events = df_to_events(
            df, video_id=video_id,
            label_columns=label_cols,
            index_col=idx_col
        )
        videos.append(video_events)

    result = {"videos": videos}
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {len(videos)} videos to {output_json}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert frame CSVs to competition JSON"
    )
    parser.add_argument(
        "--csv_dir", type=str, default="./results",
        help="Directory containing per-video frame prediction CSVs"
    )
    parser.add_argument(
        "--output", type=str, default="./results/submission.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--prefix", type=str, default="test_",
        help="CSV filename prefix to match"
    )
    args = parser.parse_args()
    build_json_from_csvs(args.csv_dir, args.output, args.prefix)
