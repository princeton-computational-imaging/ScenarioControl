"""Checks that generate_scene.py computes the same image conditioning (DINOv3 patch features + depth map)
as the dataset scripts data_processing/nuplan/extract_dino_depth_features.py + resize_dino_depth_features.py.

Needs a GPU and the DINOv3 / ZoeDepth weights (README, Quick start). Run from the repo root:
    python tests/test_image_features.py [--image front.jpg]   # default: a synthetic 1920x1080 image
"""
import argparse
import os
import pickle
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.nuplan.extract_dino_depth_features import DINOV3_MODEL_ID, NuPlanCamPKLDataset, run_dataset_loop
from data_processing.nuplan.resize_dino_depth_features import process_npz
from generate_scene import CAMERA_SIZE, image_features, to_camera_frame


def _synthetic_image():
    rng = np.random.RandomState(0)
    w, h = CAMERA_SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 / w, yy * 255 / h, (xx + yy) % 256], -1) + rng.randint(0, 40, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_image_features(image_path=None, depth_hw=(384, 672)):
    img = to_camera_frame(image_path) if image_path else _synthetic_image()
    with tempfile.TemporaryDirectory() as tmp:
        # one-frame "dataset" in the layout extract_dino_depth_features.py reads (lossless PNG, so both paths see the same pixels)
        os.makedirs(f'{tmp}/img/CAM_F0'), os.makedirs(f'{tmp}/pkl/test')
        Image.fromarray(img).save(f'{tmp}/img/CAM_F0/frame.png')
        cam = {'filename_jpg': 'CAM_F0/frame.png', 'intrinsic': np.eye(3), 'rotation': [1, 0, 0, 0],
               'translation': [0, 0, 0], 'ego_pose': [0, 0, 0, 1, 0, 0, 0]}
        with open(f'{tmp}/pkl/test/frame_0.pkl', 'wb') as f:
            pickle.dump({'idx': 0, 'cam_info': {'CAM_F0': cam}, 'ego_state_og': [[0, 0, 0], [1, 0, 0, 0]]}, f)
        ds = NuPlanCamPKLDataset(f'{tmp}/pkl', 'test', f'{tmp}/img', pattern='*_0.pkl', front_only=True)
        run_dataset_loop(ds, batch_size=1, out_root=f'{tmp}/raw', feat_backbone_id=DINOV3_MODEL_ID, resize_feats=False,
                         resize_depth=True, num_workers=1, prefetch_factor=2, limit=0)  # the script's defaults
        assert process_npz(f'{tmp}/raw/test/frame_0_dino_depths.npz', f'{tmp}/raw', f'{tmp}/resized', depth_hw[::-1]) != 'error'
        ref = np.load(f'{tmp}/resized/test/frame_0_dino_depths.npz')
        ref_feats, ref_depths = ref['dino_feats'], ref['depths']

    feats, depths = image_features(img, depth_hw)
    assert feats.shape == ref_feats.shape == (1, 384, 14, 14) and depths.shape == ref_depths.shape == (1, *depth_hw)
    print(f'max |diff|: dino_feats {np.abs(feats - ref_feats).max():.2e}, depths {np.abs(depths - ref_depths).max():.2e}')
    np.testing.assert_allclose(feats, ref_feats, rtol=0, atol=1e-4)
    np.testing.assert_allclose(depths, ref_depths, rtol=0, atol=1e-4)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', help='front-camera image (default: synthetic)')
    test_image_features(ap.parse_args().image)
    print('ok')
