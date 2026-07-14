#!/usr/bin/env bash

CODE_DIR="$PROJECT_ROOT/data_processing/nuplan"

cd "$CODE_DIR"
python extract_dino_depth_features.py \
  --pkl-root="$DATASET_ROOT/scenario_control_nuplan_temp_8cam_4xsubsampled/sledge_raw" \
  --img-root="$NUPLAN_DATA_FOLDER/sensor_blobs" \
  --split=all

# normalize all cached depth maps to a single consistent shape, since the LDM's image-conditioning
# pipeline requires every batched sample to share the same depth_map shape (see cfgs/dataset/nuplan_ldm.yaml)
python resize_dino_depth_features.py \
  --src-root="$SCRATCH_ROOT/dino_depth_features/nuplan/dinov3_zoedepth" \
  --dst-root="$SCRATCH_ROOT/dino_depth_features/nuplan/dinov3_zoedepth_resized"
