"""Convert between training checkpoints (DCP) and the Hub layout (``config.json`` + ``adapter.safetensors``).

    # trained DCP -> Hub variant dir (trained tensors only, bf16)
    scenariocontrol export adapter <run>/checkpoints/<checkpoint>/model OUT/prompt-only --variant prompt-only

    # base model (+ optional released adapter) -> DCP to start or continue training from
    scenariocontrol export dcp OUT_DCP [--from-variant prompt-only|<variant dir>]

Runs on CPU. ``dcp`` holds the whole base model in memory (~32 GB).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

BASE_MODEL = {"repo_id": "nvidia/Cosmos3-Nano", "revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92"}
VAE = {"repo_id": "Wan-AI/Wan2.2-TI2V-5B", "revision": "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
       "filename": "Wan2.2_VAE.pth"}
LORA = {"rank": 128, "alpha": 64, "target_modules": ["q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen"]}
FIXED_CAPTION = "城市中一辆汽车在道路上行驶的逼真视频。"  # "A realistic video of a car driving on a city road."
NEGATIVE_PROMPT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
                   "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
                   "三条腿，背景人很多，倒着走")
RECIPE = {"resolution": "704", "aspect_ratio": "16,9", "fps": 10, "num_steps": 35, "guidance": 6.0, "shift": 10.0,
          "control_guidance": 1.0, "num_video_frames_per_chunk": 41, "share_vision_temporal_positions": False}
VARIANT_CONFIGS = {
    "first-frame": {
        "experiment": "scenariocontrol_video_first_frame_infer",
        "trained_modules": ["lora_", "control_embedder", "inpaint_embedder"],
        "drop_control_rgb_frame0": False,
        "recipe": dict(RECIPE, prompt=FIXED_CAPTION, negative_prompt=NEGATIVE_PROMPT),
    },
    "prompt-only": {
        "experiment": "scenariocontrol_video_prompt_only_infer",
        "trained_modules": ["lora_", "control_embedder"],
        "drop_control_rgb_frame0": True,
        "recipe": dict(RECIPE, negative_prompt=""),
    },
}


def make_config(variant: str, num_params: int) -> dict:
    return {"variant": variant, "base_model": BASE_MODEL, "vae": VAE, "lora": LORA, **VARIANT_CONFIGS[variant],
            "num_trained_parameters": num_params}


def export_adapter(dcp_dir, out_dir, variant: str) -> dict:
    """Write ``adapter.safetensors`` (trained tensors, bf16, ``net.`` prefix stripped) and ``config.json``."""
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file
    from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader

    reader = FileSystemReader(str(dcp_dir))
    meta = reader.read_metadata().state_dict_metadata
    prefixes = VARIANT_CONFIGS[variant]["trained_modules"]
    keys = sorted(k for k in meta if k.startswith("net.") and any(p in k for p in prefixes))
    has_inpaint = any("inpaint_embedder" in k for k in meta)
    if has_inpaint != (variant == "first-frame") or not any("control_embedder" in k for k in keys):
        raise ValueError(f"{dcp_dir} does not look like a {variant} checkpoint")
    sd = {k: torch.empty(meta[k].size, dtype=meta[k].properties.dtype) for k in keys}
    dcp.load(sd, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=True))
    adapter = {k.removeprefix("net."): v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
    num = sum(v.numel() for v in adapter.values())
    # LoRA r128 on 36 layers x q/k/v/o_proj_moe_gen + control_embedder (+ inpaint_embedder).
    expected = {"first-frame": 124_272_640, "prompt-only": 123_469_824}[variant]
    assert num == expected, f"{num} trained parameters, expected {expected} for {variant} with the released LoRA config"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(adapter, str(out / "adapter.safetensors"), metadata={"format": "pt"})
    config = make_config(variant, num)
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(adapter)} tensors, {num / 1e9:.4f}B params -> {out}")
    return config


def export_dcp(out_dir, variant_dir=None, token=None) -> None:
    """Write a training DCP (``<out>/model``): the pinned base, plus the adapter of ``variant_dir`` if given."""
    import torch.distributed.checkpoint as dcp
    from cosmos_framework.checkpoint.dcp import CustomSavePlanner
    from cosmos_framework.inference.model import _diffusers_files_to_keys, _diffusers_to_net_key, _diffusers_weight_map
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.filesystem import FileSystemWriter

    from scenariocontrol_video.pipeline import BASE_PATTERNS

    base = Path(snapshot_download(BASE_MODEL["repo_id"], revision=BASE_MODEL["revision"],
                                  allow_patterns=BASE_PATTERNS, token=token))
    sd = {}
    for rel, diff_keys in _diffusers_files_to_keys(_diffusers_weight_map(base)).items():
        with safe_open(str(base / rel), framework="pt") as f:
            for k in diff_keys:
                net_key = _diffusers_to_net_key(k, rel)
                if net_key is not None:
                    sd["net." + net_key] = f.get_tensor(k)
    if variant_dir is not None:
        if not Path(variant_dir).is_dir() and variant_dir in VARIANT_CONFIGS:  # released variant from the Hub
            from scenariocontrol_video.pipeline import DEFAULT_REPO

            variant_dir = Path(snapshot_download(DEFAULT_REPO, allow_patterns=[f"{variant_dir}/*"], token=token)) / variant_dir
        sd.update({"net." + k: v for k, v in load_file(str(Path(variant_dir) / "adapter.safetensors")).items()})
    size = sum(v.numel() * v.element_size() for v in sd.values())
    writer = FileSystemWriter(str(Path(out_dir) / "model"), thread_count=max(1, math.ceil(size / (5 * 1024**3))))  # 5 GB shards
    dcp.save(state_dict=sd, storage_writer=writer, planner=CustomSavePlanner())
    print(f"wrote {len(sd)} tensors ({size / 1e9:.1f} GB) -> {out_dir}/model")


def main(argv=None, prog=None) -> None:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("adapter", help="trained DCP -> config.json + adapter.safetensors")
    a.add_argument("dcp_dir", help="checkpoint dir holding .metadata and *.distcp (<run>/checkpoints/<checkpoint>/model)")
    a.add_argument("out_dir")
    a.add_argument("--variant", required=True, choices=sorted(VARIANT_CONFIGS))
    d = sub.add_parser("dcp", help="base model (+ adapter) -> training DCP")
    d.add_argument("out_dir")
    d.add_argument("--from-variant", help="released variant name (first-frame, prompt-only) or a local variant dir")
    args = ap.parse_args(argv)
    if args.cmd == "adapter":
        export_adapter(args.dcp_dir, args.out_dir, args.variant)
    else:
        export_dcp(args.out_dir, args.from_variant)


if __name__ == "__main__":
    main()
