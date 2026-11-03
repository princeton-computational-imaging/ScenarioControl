# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Modified by the ScenarioControl authors: `WireframeControlDataset.process_one_sample` and
# `get_wireframe_control_dataset` follow `SFTDataset.process_one_sample` and `get_sft_dataset`
# in `cosmos_framework/data/vfm/local_datasets/sft_dataset.py`
# (NVIDIA/cosmos-framework@fa63e9934a38eb8127a0c3d6b23969fd4d0e32a7), extended to decode a
# frame-synced control video and emit `video=[control, target]`.
"""Training dataset: paired (wireframe control, real) driving clips for Cosmos3 SFT.

Each JSONL record is a standard Cosmos3 SFT record (``uuid``, ``duration``, ``width``,
``height``, ``vision_path``, ``t2w_windows`` with a ``caption``) plus ``control_path``,
the frame-synced wireframe video (see ``scenariocontrol_video.data``). Both videos are
decoded with the same window, temporal stride, resize and centre crop, and the sample
carries ``video=[control, target]`` for ``OmniMoTControlConcatModel``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from typing import Any

import numpy as np
import torch
from cosmos_framework.data.vfm.local_datasets.helper import ffmpeg_decode_video, get_video_metadata
from cosmos_framework.data.vfm.local_datasets.sft_dataset import (
    SFTDataset,
    _flatten_metadata_by_window,
    _load_sft_metadata_from_s3,
    _select_caption,
    get_sft_dataset,
)
from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.utils import log


def _to_float(chunk: np.ndarray) -> torch.Tensor:
    """[T,H,W,3] uint8 -> [3,T,H,W] float32 in [-1, 1] (float items keep the multi-item path 5-D)."""
    return torch.from_numpy(np.ascontiguousarray(np.transpose(chunk, (3, 0, 1, 2)))).float() / 127.5 - 1.0


class WireframeControlDataset(SFTDataset):
    """SFTDataset that also decodes the paired wireframe control video.

    ``drop_control_rgb_frame0``: control frame 0 is the real first frame (frames 1+
    are wireframes). When set, it is replaced by wireframe frame 1 before encoding, so
    the model never sees the real image through the control (prompt-only variant).
    """

    def __init__(self, *args: Any, control_path_key: str = "control_path",
                 share_vision_temporal_positions: bool = False, drop_control_rgb_frame0: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert self.conditioning_config is not None, "conditioning_config is required"
        self.control_path_key = control_path_key
        self.share_vision_temporal_positions = share_vision_temporal_positions
        self.drop_control_rgb_frame0 = drop_control_rgb_frame0

    def _decode(self, path, start, end, interval, resize_hw, drop_frame0=False) -> list[np.ndarray]:
        frames = []
        for idx, frame in enumerate(ffmpeg_decode_video(path, scale_hw=resize_hw, num_threads=2)):
            if idx > end:
                break
            if idx >= start and (idx - start) % interval == 0 and not (drop_frame0 and idx == start):
                frames.append(frame)
        if drop_frame0 and frames:
            frames = [frames[0]] + frames  # keep the 4n+1 length with the first wireframe
        return frames

    def process_one_sample(self, metadata: dict) -> dict | None:
        control_path = metadata.get(self.control_path_key)
        if control_path is None:
            log.warning(f"No '{self.control_path_key}' for sample {metadata.get('uuid')}; skipping.")
            return None

        windows = metadata["t2w_windows"]
        win_idx = random.randrange(len(windows))
        window = windows[win_idx]

        input_w, input_h = metadata["width"], metadata["height"]
        target_w, target_h = self.output_sizes[metadata["aspect_ratio"]]
        ratio = max(target_w / input_w, target_h / input_h)
        resize_h, resize_w = round(input_h * ratio), round(input_w * ratio)
        crop_y, crop_x = round((resize_h - target_h) / 2), round((resize_w - target_w) / 2)

        video_path = metadata["vision_path"]
        try:
            info = get_video_metadata(video_path)
        except Exception as e:
            log.warning(f"ffprobe failed for {video_path}: {e}; skipping.")
            return None
        original_fps, total_frames = info["fps"], info["total_frames"]
        window_start = window["start_frame"]
        actual_end = min(window["end_frame"], total_frames - 1)
        frames_in_window = actual_end - window_start + 1

        if self.num_video_frames == -1:
            interval, start, end = window["temporal_interval"], window_start, actual_end
        else:
            if frames_in_window < self.num_video_frames:
                log.warning(f"Not enough frames in window: {metadata['uuid']}")
                return None
            if self.temporal_interval_mode == "force_one":
                interval = 1
            elif self.temporal_interval_mode == "max_30fps":
                interval = max(1, int(original_fps / 30.0))
            elif self.temporal_interval_mode == "entire_chunk":
                interval = max(1, frames_in_window // self.num_video_frames)
            else:
                raise ValueError(f"Unknown temporal_interval_mode: {self.temporal_interval_mode}")
            span = (self.num_video_frames - 1) * interval + 1
            if self.frame_selection_mode == "first":
                start = window_start
            elif self.frame_selection_mode == "center":
                start = window_start + (frames_in_window - span) // 2
            elif self.frame_selection_mode == "random":
                start = window_start + random.randint(0, max(0, frames_in_window - span))
            else:
                raise ValueError(f"Unknown frame_selection_mode: {self.frame_selection_mode}")
            end = start + span - 1
        fps = original_fps / interval

        real = self._decode(video_path, start, end, interval, (resize_h, resize_w))
        ctrl = self._decode(control_path, start, end, interval, (resize_h, resize_w), self.drop_control_rgb_frame0)
        if not real or not ctrl:
            log.warning(f"No frames decoded for {metadata['uuid']}; skipping.")
            return None
        tcf = self.temporal_compression_factor
        t = (min(len(real), len(ctrl)) - 1) // tcf * tcf + 1  # equal lengths, 4n+1 for the VAE
        crop = (slice(0, t), slice(crop_y, crop_y + target_h), slice(crop_x, crop_x + target_w))
        real_video, wf_video = _to_float(np.stack(real)[crop]), _to_float(np.stack(ctrl)[crop])

        selected = _select_caption(window)
        if selected is None:
            log.warning(f"No caption in window for {metadata['uuid']}; skipping.")
            return None
        caption_key, caption, used_json = selected
        cond_fps = fps if self.conditioning_fps < 0 else self.conditioning_fps
        if self.conditioning_fps_noise_std > 0:
            cond_fps = cond_fps * float(np.exp(np.random.randn() * self.conditioning_fps_noise_std))
        if self.caption_suffix and not used_json:
            caption = (caption + " " + self.caption_suffix).strip()
        if self.cfg_dropout_keep_metadata and self.cfg_dropout_rate > 0 and random.random() < self.cfg_dropout_rate:
            caption = ""
        if self.append_duration_fps_timestamps and not used_json:
            caption += f" The video is {t / cond_fps:.1f} seconds long and is of {cond_fps:.0f} FPS."
        if self.append_resolution_info and not used_json:
            caption += f" This video is of {target_h}x{target_w} resolution."
        caption = caption.strip()
        if not self.cfg_dropout_keep_metadata and self.cfg_dropout_rate > 0 and random.random() < self.cfg_dropout_rate:
            caption = ""
        text_ids, caption = self._tokenize_caption(caption)

        # Clean (conditioning) latent frames for the target, from conditioning_config {n: prob}.
        t_latent = 1 + (t - 1) // tcf
        options, weights = list(self.conditioning_config.keys()), list(self.conditioning_config.values())
        num_cond = min(random.choices(options, weights=weights, k=1)[0], t_latent - 1)

        image_size = torch.tensor([target_h, target_w, target_h, target_w], dtype=torch.float32)
        return dict(
            __key__=f"{metadata['uuid']}_w{win_idx}",
            __url__=video_path,
            fps=original_fps,
            n_orig_video_frames=total_frames,
            chunk_index=win_idx,
            frame_start=start,
            frame_end=end,
            num_frames=t,
            video=[wf_video, real_video],  # control first, target last
            num_multiplier=interval,
            conditioning_fps=cond_fps,
            image_size=[image_size, image_size],
            padding_mask=torch.zeros((1, target_h, target_w), dtype=torch.float32),
            ai_caption=caption,
            sampled_caption_style=caption_key,
            text_token_ids=torch.tensor(text_ids),
            sequence_plan=SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=list(range(num_cond)),
                share_vision_temporal_positions=self.share_vision_temporal_positions,
            ),
        )


def get_wireframe_control_dataset(jsonl_paths: str | list[str], *, control_path_key: str = "control_path",
                                  share_vision_temporal_positions: bool = False,
                                  drop_control_rgb_frame0: bool = False, min_frames: int = 16,
                                  **kwargs: Any) -> WireframeControlDataset:
    """Build the dataset from JSONL path(s).

    ``get_sft_dataset`` resolves the defaults but drops clips shorter than 61 frames, so the
    metadata is reloaded with ``min_frames`` and ``control_path`` is re-attached by uuid.
    """
    base = get_sft_dataset(jsonl_paths=jsonl_paths, **kwargs)
    paths = [jsonl_paths] if isinstance(jsonl_paths, str) else list(jsonl_paths)

    import boto3  # the loader expects a client; unused for local paths

    s3 = boto3.client("s3", **(base.s3_credentials or {}))
    metadata: list[dict] = []
    for i, p in enumerate(paths):
        metadata += _load_sft_metadata_from_s3(s3, p, min_frames=min_frames, uuid_prefix=f"{i}/" if len(paths) > 1 else "",
                                               min_short_edge=int(kwargs.get("min_short_edge", 0)))
    if kwargs.get("sample_by_window", False):
        metadata = _flatten_metadata_by_window(metadata)
    metadata.sort(key=lambda m: hashlib.sha256(m["uuid"].encode("utf-8")).hexdigest())
    if not metadata:
        log.warning(f"0 records with >= {min_frames} frames in {paths}")

    controls: dict[str, str] = {}
    for p in paths:
        with (gzip.open if p.endswith(".gz") else open)(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if control_path_key in rec:
                        controls[str(rec["uuid"])] = rec[control_path_key]
    for m in metadata:
        m[control_path_key] = controls.get(str(m["uuid"]).split("/")[-1])

    return WireframeControlDataset(
        metadata=metadata,
        num_video_frames=base.num_video_frames,
        resolution=base.resolution,
        s3_credentials=base.s3_credentials,
        temporal_interval_mode=base.temporal_interval_mode,
        frame_selection_mode=base.frame_selection_mode,
        tokenizer_config=kwargs.get("tokenizer_config"),
        cfg_dropout_rate=base.cfg_dropout_rate,
        use_system_prompt=base.use_system_prompt,
        max_caption_tokens=base.max_caption_tokens,
        append_duration_fps_timestamps=base.append_duration_fps_timestamps,
        append_resolution_info=base.append_resolution_info,
        cfg_dropout_keep_metadata=base.cfg_dropout_keep_metadata,
        caption_suffix=base.caption_suffix,
        conditioning_fps=base.conditioning_fps,
        conditioning_fps_noise_std=base.conditioning_fps_noise_std,
        conditioning_config=kwargs.get("conditioning_config"),
        temporal_compression_factor=base.temporal_compression_factor,
        control_path_key=control_path_key,
        share_vision_temporal_positions=share_vision_temporal_positions,
        drop_control_rgb_frame0=drop_control_rgb_frame0,
    )
