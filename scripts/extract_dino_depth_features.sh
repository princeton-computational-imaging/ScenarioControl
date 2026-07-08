#!/usr/bin/env bash

CODE_DIR="$PROJECT_ROOT/data_processing/nuplan"

cd "$CODE_DIR"
python extract_dino_depth_features.py \
  --pkl-root="$DATASET_ROOT/scenario_control_nuplan_temp_8cam_4xsubsampled/sledge_raw" \
  --img-root="$NUPLAN_DATA_FOLDER/sensor_blobs" \
  --split=all
