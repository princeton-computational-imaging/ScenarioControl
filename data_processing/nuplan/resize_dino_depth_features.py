"""
Normalizes the depth maps cached by extract_dino_depth_features.py to a single, consistent
(height, width) size. The depth estimator's raw output resolution can vary slightly across
frames, but the LDM's image-conditioning pipeline (see cfgs/dataset/nuplan_ldm.yaml's
depth_map_shape and nn_modules/dit.py) requires every cached depth map to share the exact
same shape so samples can be batched together. dino_feats are left untouched.

Example:
  python resize_dino_depth_features.py \\
      --src-root=$SCRATCH_ROOT/dino_depth_features/nuplan/dinov3_zoedepth \\
      --dst-root=$SCRATCH_ROOT/dino_depth_features/nuplan/dinov3_zoedepth_resized
"""
import os
import shutil
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm


def process_npz(src_path: str, src_root: str, dst_root: str, target_size: tuple):
    """Resize an .npz's depth map to target_size=(width, height) if needed; otherwise copy it as-is."""
    try:
        rel_path = os.path.relpath(src_path, src_root)
        dst_path = os.path.join(dst_root, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        data = np.load(src_path)
        depths = data["depths"]
        dino_feats = data["dino_feats"]

        H, W = depths.shape[-2:]
        if (H, W) == (target_size[1], target_size[0]):
            shutil.copy2(src_path, dst_path)
            return "copied"

        if depths.ndim == 3 and depths.shape[0] == 1:
            depths_2d = depths.squeeze(0)
        else:
            depths_2d = depths

        depths_resized_2d = cv2.resize(depths_2d, target_size, interpolation=cv2.INTER_LINEAR)
        depths_resized = np.expand_dims(depths_resized_2d, axis=0)

        np.savez_compressed(dst_path, depths=depths_resized, dino_feats=dino_feats)
        return "resized"

    except Exception as e:
        return f"error: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_src = os.path.join(os.environ.get("SCRATCH_ROOT", "."), "dino_depth_features", "nuplan", "dinov3_zoedepth")
    parser.add_argument("--src-root", type=str, default=default_src,
                        help="Directory of .npz files written by extract_dino_depth_features.py")
    parser.add_argument("--dst-root", type=str, default=default_src + "_resized",
                        help="Output directory for the normalized .npz files")
    parser.add_argument("--target-width", type=int, default=672)
    parser.add_argument("--target-height", type=int, default=384)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()

    target_size = (args.target_width, args.target_height)  # cv2.resize expects (width, height)

    npz_files = list(Path(args.src_root).rglob("*.npz"))
    print(f"Found {len(npz_files)} .npz files under {args.src_root}")
    print(f"Using {args.workers} parallel workers")

    resized, copied, errors = 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_npz, str(f), args.src_root, args.dst_root, target_size)
            for f in npz_files
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Resizing .npz files"):
            status = fut.result()
            if status == "resized":
                resized += 1
            elif status == "copied":
                copied += 1
            else:
                errors += 1
                print(f"Error: {status}")

    print("Done.")
    print(f"Resized: {resized}")
    print(f"Copied (already correct size): {copied}")
    print(f"Errors: {errors}")
    print(f"All outputs saved to: {args.dst_root}")
