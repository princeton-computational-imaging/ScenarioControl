"""Write the training JSONL files for a directory of paired clips.

Layout (one id per clip, same frame count and frame rate in both videos)::

    $DATASET_PATH/train/<id>.mp4        real front-camera clip (target)
    $DATASET_PATH/src_videos/<id>.mp4   wireframe control: frame 0 = real first frame, frames 1+ = wireframes

    python -m scenariocontrol_video.data $DATASET_PATH [--captions captions.json]

writes ``train/video_dataset_file.jsonl`` (one fixed caption, used by first_frame)
and, with ``--captions`` (``{"<id>": "<caption>", ...}``), ``train/video_dataset_text.jsonl``
(per-clip captions, used by prompt_only; clips without a caption are left out).
"""

import argparse
import json
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_path")
    ap.add_argument("--captions", help="JSON file mapping clip id -> caption")
    args = ap.parse_args(argv)

    from cosmos_framework.data.vfm.local_datasets.helper import get_video_metadata

    from scenariocontrol_video.export import FIXED_CAPTION

    root = Path(args.dataset_path).resolve()
    captions = json.loads(Path(args.captions).read_text()) if args.captions else None
    fixed, text, skipped = [], [], 0
    for video in sorted((root / "train").glob("*.mp4")):
        control = root / "src_videos" / video.name
        if not control.exists():
            skipped += 1
            continue
        meta, cmeta = get_video_metadata(str(video)), get_video_metadata(str(control))
        if meta["total_frames"] != cmeta["total_frames"]:  # misaligned pair
            skipped += 1
            continue
        n = meta["total_frames"]
        rec = {"uuid": video.stem, "duration": round(n / meta["fps"], 4), "width": meta["width"],
               "height": meta["height"], "vision_path": str(video), "control_path": str(control)}
        window = {"start_frame": 0, "end_frame": n - 1, "temporal_interval": 1}
        fixed.append(dict(rec, t2w_windows=[dict(window, caption=FIXED_CAPTION)]))
        if captions and captions.get(video.stem):
            text.append(dict(rec, t2w_windows=[dict(window, caption=captions[video.stem])]))
    for name, recs in [("video_dataset_file.jsonl", fixed), ("video_dataset_text.jsonl", text if captions else None)]:
        if recs is not None:
            with open(root / "train" / name, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
            print(f"{len(recs)} clips -> {root / 'train' / name}")
    print(f"skipped {skipped} clips (no control video or different frame count)")


if __name__ == "__main__":
    main()
