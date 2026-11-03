"""Hydra experiments for ScenarioControl video generation (registered on import).

==================================  ===================================================
``scenariocontrol_video_first_frame``  LoRA + control_embedder + inpaint_embedder (first-
                                    frame anchor as input channels) -> ``first-frame``
``scenariocontrol_video_prompt_only``  per-clip captions, no image, control frame 0
                                    replaced by wireframe frame 1 -> ``prompt-only``
==================================  ===================================================

``*_infer`` are the matching inference graphs (base Nano graph with the model swapped).
Training data paths come from ``$DATASET_PATH`` (see ``scenariocontrol_video.data``).
"""

import copy

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.experiment.sft.vision_sft_nano import vision_sft_nano
from cosmos_framework.utils.lazy_config import LazyCall as L
from hydra.core.config_store import ConfigStore

from scenariocontrol_video.dataset import get_wireframe_control_dataset
from scenariocontrol_video.model import OmniMoTControlConcatModel

PREFIX = "scenariocontrol_video"
cs = ConfigStore.instance()

for _name, _inpaint in [("mot_control_concat_fsdp", False), ("mot_control_concat_inpaint_fsdp", True)]:
    cs.store(
        group="model",
        package="_global_",
        name=_name,
        node=dict(
            trainer=dict(distributed_parallelism="fsdp"),
            model=L(OmniMoTControlConcatModel)(
                config=OmniMoTModelConfig(parallelism=ParallelismConfig(data_parallel_shard_degree=8)),
                control_dropout_p=0.1,
                inpaint_anchor=_inpaint,
                _recursive_=False,
            ),
        ),
    )


def _with_model(model_name: str, name: str):
    node = copy.deepcopy(vision_sft_nano)
    node["job"]["name"] = name
    # The training dataloader is never built at inference; don't require $DATASET_PATH there.
    node["dataloader_train"]["dataloader"]["datasets"]["video"]["dataset"]["jsonl_paths"] = ["unused.jsonl"]
    for entry in node["defaults"]:
        if not isinstance(entry, str) and "override /model" in entry:
            entry["override /model"] = model_name
            return node
    raise RuntimeError("no 'override /model' entry in vision_sft_nano defaults")


def _train(name, conditioning_config, inpaint=False, jsonl="video_dataset_file.jsonl", drop_control_rgb_frame0=False):
    node = _with_model("mot_control_concat_inpaint_fsdp" if inpaint else "mot_control_concat_fsdp", name)
    node["dataloader_train"]["dataloader"]["datasets"]["video"]["dataset"] = L(get_wireframe_control_dataset)(
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        max_caption_tokens=2048,
        caption_suffix="",
        cfg_dropout_keep_metadata=False,
        cfg_dropout_rate=0.1,
        conditioning_config=conditioning_config,
        conditioning_fps=-1,
        conditioning_fps_noise_std=0.0,
        frame_selection_mode="first",
        jsonl_paths=[f"${{oc.env:DATASET_PATH}}/train/{jsonl}"],
        min_short_edge=0,
        num_video_frames=-1,  # whole clip (41 frames)
        resolution="704",
        sample_by_window=False,
        temporal_compression_factor=4,
        temporal_interval_mode="max_30fps",
        use_system_prompt=False,
        tokenizer_config="${model.config.vlm_config.tokenizer}",
        share_vision_temporal_positions=False,
        drop_control_rgb_frame0=drop_control_rgb_frame0,
    )
    return node


# conditioning_config: {number of clean latent frames: probability}. For the first-frame
# variant the clean frames become anchor channels instead (see model.py).
EXPERIMENTS = {
    f"{PREFIX}_first_frame": _train(f"{PREFIX}_first_frame", {0: 0.1, 1: 0.8, 2: 0.1}, inpaint=True),
    f"{PREFIX}_prompt_only": _train(f"{PREFIX}_prompt_only", {0: 1.0}, jsonl="video_dataset_text.jsonl",
                                     drop_control_rgb_frame0=True),
    f"{PREFIX}_first_frame_infer": _with_model("mot_control_concat_inpaint_fsdp", f"{PREFIX}_first_frame_infer"),
    f"{PREFIX}_prompt_only_infer": _with_model("mot_control_concat_fsdp", f"{PREFIX}_prompt_only_infer"),
}

for _name, _node in EXPERIMENTS.items():
    cs.store(group="experiment", package="_global_", name=_name, node=_node)
