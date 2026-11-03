# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Modified by the ScenarioControl authors: `ScenarioControlVideoPipeline._sample` follows
# `cosmos_framework.inference.transfer.generate_transfer_sample`
# (NVIDIA/cosmos-framework@fa63e9934a38eb8127a0c3d6b23969fd4d0e32a7), reduced to one
# pre-computed control video, with the output frame rate taken from the control video and
# control frame 0 replaced by frame 1 for the prompt-only variant. The rest is new.
"""ScenarioControl video pipeline: wireframe control (or BEV rollout) -> front-camera video.

    from scenariocontrol_video import ScenarioControlVideoPipeline

    pipe = ScenarioControlVideoPipeline.from_hub("pci-lab/scenariocontrol-videogen", variant="prompt-only")
    video = pipe(wireframe="control.mp4", prompt="A rainy evening drive through a city.")
    video.save("out.mp4")
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_REPO = "pci-lab/scenariocontrol-videogen"
VARIANTS = ("first-frame", "prompt-only")
# Base-model files the network needs (diffusers layout; the VAE comes from Wan-AI, see config.json).
BASE_PATTERNS = ["model.safetensors.index.json", "model_index.json", "transformer/*", "vision_encoder/*"]


@dataclass
class VideoOutput:
    video: torch.Tensor  # [C, T, H, W] float in [0, 1]
    fps: float
    quality: int | None = None

    @property
    def frames(self):
        """[T, H, W, 3] uint8 numpy array."""
        return (self.video.permute(1, 2, 3, 0) * 255).round().to(torch.uint8).numpy()

    def save(self, path) -> Path:
        from cosmos_framework.tools.visualize.video import save_img_or_video

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_img_or_video(self.video, str(path.with_suffix("")), fps=self.fps, quality=self.quality)
        return path.with_suffix(".mp4")


def _token(token):
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


@contextlib.contextmanager
def _framework_root():
    """cosmos-framework resolves some config files relative to its install root."""
    import cosmos_framework

    with contextlib.chdir(Path(cosmos_framework.__file__).parents[1]):
        yield


def build_model(config: dict, vae_path: str, base_dir: str | None):
    """Build the control model on the current CUDA device; load the base weights if ``base_dir``."""
    import importlib

    from cosmos_framework.inference.model import Cosmos3OmniModel
    from cosmos_framework.utils import misc
    from cosmos_framework.utils.config_helper import override
    from cosmos_framework.utils.lazy_config import instantiate

    import scenariocontrol_video.experiments  # noqa: F401  (registers the experiments)

    lora = config["lora"]
    opts = [
        f"experiment={config['experiment']}",
        "model.config.lora_enabled=true",
        f"model.config.lora_rank={lora['rank']}",
        f"model.config.lora_alpha={lora['alpha']}",
        f"model.config.lora_target_modules='{','.join(lora['target_modules'])}'",
        f"model.config.tokenizer.vae_path={vae_path}",
        "model.config.rectified_flow_inference_config.scheduler_type=unipc",
    ]
    Cosmos3OmniModel.before_load_model()
    with _framework_root(), torch.no_grad():
        cfg = importlib.import_module("cosmos_framework.configs.base.config").make_config()
        cfg = override(cfg, ["--", *opts])
        # Single process, single GPU, no torch.compile.
        for k, v in dict(enable_inference_mode=True, data_parallel_shard_degree=1, data_parallel_replicate_degree=1,
                         context_parallel_shard_degree=1, cfg_parallel_shard_degree=1).items():
            setattr(cfg.model.config.parallelism, k, v)
        cfg.model.config.compile.enabled = False
        cfg.model.config.compile.use_cuda_graphs = False
        cfg.model.config.activation_checkpointing.mode = "none"
        cfg.model.config.ema.enabled = False
        cfg.validate()
        cfg.freeze()
        misc.set_random_seed(seed=0, by_rank=True)
        # cuDNN autotuning (the training config's setting) picks VAE kernels by timing and makes
        # outputs vary between runs; leave it off so a seed always gives the same video.
        torch.backends.cudnn.benchmark = False
        model = instantiate(cfg.model).cuda()
        model.on_train_start()
        if base_dir is not None:
            load_base(model, Path(base_dir), trained_prefixes=config["trained_modules"])
    return model


def trained_keys(state_dict, prefixes) -> list[str]:
    return sorted(k for k in state_dict if any(p in k for p in prefixes))


def load_base(model, base_dir: Path, trained_prefixes) -> None:
    """Load every frozen tensor of ``model.net`` from the diffusers-layout base checkpoint."""
    import torch.distributed.checkpoint as dcp
    from cosmos_framework.inference.model import _DiffusersHuggingFaceStorageReader, _DiffusersLoadPlanner
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    sd = get_model_state_dict(model.net)
    skip = set(trained_keys(sd, trained_prefixes))
    frozen = {k: v for k, v in sd.items() if k not in skip}
    # The planner raises if any frozen tensor has no source in the base checkpoint.
    dcp.load(state_dict=frozen, storage_reader=_DiffusersHuggingFaceStorageReader(base_dir),
             planner=_DiffusersLoadPlanner(base_dir))


def load_adapter(model, adapter_path, trained_prefixes) -> None:
    """Load ``adapter.safetensors`` strictly: its keys must be exactly the trained keys."""
    from safetensors.torch import load_file

    adapter = load_file(str(adapter_path))
    sd = model.net.state_dict()
    expected = set(trained_keys(sd, trained_prefixes))
    missing, unexpected = expected - set(adapter), set(adapter) - expected
    if missing or unexpected:
        raise KeyError(f"adapter mismatch: {len(missing)} missing (e.g. {sorted(missing)[:3]}), "
                       f"{len(unexpected)} unexpected (e.g. {sorted(unexpected)[:3]})")
    with torch.no_grad():
        for k, v in adapter.items():
            assert sd[k].shape == v.shape, f"{k}: {tuple(v.shape)} vs {tuple(sd[k].shape)}"
            sd[k].copy_(v)


def _reduce_memory(model) -> None:
    """For 32 GB GPUs: lm_head to CPU (unused for video) and int8 weight-only linears (not LoRA)."""
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    lm = model.net.language_model
    lm.lm_head.to("cpu")
    quantize_(lm, Int8WeightOnlyConfig(),
              filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and "lora" not in fqn and "lm_head" not in fqn)
    torch.cuda.empty_cache()


class ScenarioControlVideoPipeline:
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config

    @property
    def variant(self) -> str:
        return self.config["variant"]

    # -------------------------------------------------------------- loading --
    @classmethod
    def from_hub(cls, repo_id: str = DEFAULT_REPO, variant: str = "prompt-only", device: str = "cuda",
                 token: str | None = None, cache_dir: str | None = None, lowvram: bool = False):
        """Download ``<repo_id>/<variant>/`` and build the pipeline (see :meth:`from_pretrained`)."""
        from huggingface_hub import snapshot_download

        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        local = snapshot_download(repo_id, allow_patterns=[f"{variant}/*"], token=_token(token), cache_dir=cache_dir)
        return cls.from_pretrained(os.path.join(local, variant), device=device, token=token, cache_dir=cache_dir,
                                   lowvram=lowvram)

    @classmethod
    def from_pretrained(cls, model_dir, device: str = "cuda", token: str | None = None, cache_dir: str | None = None,
                        lowvram: bool = False):
        """Build from a local variant dir (``config.json`` + ``adapter.safetensors``).

        The base model (``nvidia/Cosmos3-Nano`` at the pinned revision) and the Wan2.2 VAE
        are fetched from the Hub cache. ``lowvram`` quantises the frozen base to int8
        so inference fits a 32 GB GPU (the bf16 load still peaks near 31 GB).
        """
        from huggingface_hub import hf_hub_download, snapshot_download

        if not str(device).startswith("cuda"):
            raise ValueError("ScenarioControlVideoPipeline needs a CUDA device")
        if ":" in str(device):
            torch.cuda.set_device(torch.device(device))
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        base, vae = config["base_model"], config["vae"]
        base_dir = snapshot_download(base["repo_id"], revision=base["revision"], allow_patterns=BASE_PATTERNS,
                                     token=_token(token), cache_dir=cache_dir)
        vae_path = hf_hub_download(vae["repo_id"], vae["filename"], revision=vae["revision"], token=_token(token),
                                   cache_dir=cache_dir)
        model = build_model(config, vae_path, base_dir)
        load_adapter(model, model_dir / "adapter.safetensors", config["trained_modules"])
        if lowvram:
            _reduce_memory(model)
        model.eval()
        return cls(model, config)

    # ------------------------------------------------------------ inference --
    def __call__(self, wireframe=None, bev=None, prompt: str | None = None, image=None, num_frames: int = 41,
                 seed: int = 43, output_dir=None, **recipe_overrides) -> VideoOutput:
        """Generate a front-camera video.

        Args:
            wireframe: control video (``.mp4``) whose frame 0 is the real first frame
                (or a wireframe) and frames 1+ are wireframes.
            bev: directory of ``frame_XXXX.pkl`` rollout frames, rendered to a control video.
                Exactly one of ``wireframe`` / ``bev``.
            prompt: scene description. Required for ``prompt-only``; ``first-frame`` was
                trained with one fixed caption and uses it by default.
            image: real first frame (``.png``/``.jpg``). Required for ``first-frame``.
            num_frames: frames to generate (41, the training length, by default). Longer
                controls are generated as independent 41-frame chunks; this is untested.
            seed: sampling seed.
            output_dir: where a rendered ``control.mp4`` is kept (``bev`` only).
            recipe_overrides: e.g. ``num_steps``, ``guidance``, ``shift``, ``negative_prompt``.
        """
        if (wireframe is None) == (bev is None):
            raise ValueError("pass exactly one of wireframe= or bev=")
        if self.variant == "first-frame" and image is None:
            raise ValueError("the first-frame variant needs image= (the real first frame)")
        if self.variant == "prompt-only" and image is not None:
            raise ValueError("the prompt-only variant takes no image")
        recipe = dict(self.config["recipe"], **recipe_overrides)
        prompt = prompt if prompt is not None else recipe.pop("prompt", None)
        recipe.pop("prompt", None)
        if not prompt:
            raise ValueError("prompt= is required for the prompt-only variant")

        with tempfile.TemporaryDirectory() as tmp:
            if bev is not None:
                from scenariocontrol_video.render import make_control_video

                out = Path(output_dir) if output_dir else Path(tmp)
                wireframe = make_control_video(bev, out / "control.mp4", image=image, num_frames=num_frames)
            sample = dict(name="sample", prompt=prompt, seed=int(seed), max_frames=int(num_frames),
                          wsm={"control_path": str(wireframe)}, show_control_condition=False,
                          num_conditional_frames=0, num_first_chunk_conditional_frames=1 if image else 0,
                          output_dir=tmp, **recipe)
            if image is not None:
                sample["vision_path"] = str(image)
            return self._sample(sample)

    @torch.no_grad()
    def _sample(self, sample: dict) -> VideoOutput:
        from cosmos_framework.inference.args import OmniSampleOverrides
        from cosmos_framework.inference.inference import _get_prompt_sample_data
        from cosmos_framework.inference.transfer import _get_num_chunks, build_transfer_batch
        from cosmos_framework.inference.vision import pad_temporal_frames, read_and_resize_media, uint8_to_normalized_float

        model = self.model
        args = OmniSampleOverrides.model_validate(sample).build_sample(model_config=model.config)
        read = lambda p: read_and_resize_media(Path(p), resolution=args.resolution, aspect_ratio=args.aspect_ratio,  # noqa: E731
                                               max_frames=args.max_frames)[0]
        control = read(sample["wsm"]["control_path"])  # [C, T, H, W] uint8
        if self.config["drop_control_rgb_frame0"] and control.shape[1] >= 2:
            # Control frame 0 may be the real image; the prompt-only model never saw it (as in training).
            control = control.clone()
            control[:, 0] = control[:, 1]
        image = read(args.vision_path) if args.vision_path is not None else None
        fps = float(self.config["recipe"]["fps"])  # cadence of the control video, as in training
        height, width = control.shape[2], control.shape[3]

        total = control.shape[1]
        tcf = model.config.tokenizer.temporal_compression_factor
        chunk = 1 if total == 1 else args.num_video_frames_per_chunk
        chunk = math.ceil((chunk - 1) / tcf) * tcf + 1
        num_chunks, stride = _get_num_chunks(total, chunk, args.num_conditional_frames)
        control = pad_temporal_frames(control, max(total, chunk))

        prompt_data = _get_prompt_sample_data(args.model_copy(update={"num_frames": chunk, "fps": int(round(fps))}),
                                              model, h=height, w=width, device="cuda")
        prompt = prompt_data[model.input_caption_key][0]
        negative = prompt_data.get("neg_" + model.input_caption_key, [None])[0]

        model.eval()
        outputs = []
        for i in range(num_chunks):
            ctrl = uint8_to_normalized_float(pad_temporal_frames(control[:, i * stride: i * stride + chunk], chunk))
            target = torch.zeros_like(ctrl)
            n_cond = 0
            if i == 0 and image is not None and args.num_first_chunk_conditional_frames > 0:
                n_cond = min(args.num_first_chunk_conditional_frames, image.shape[1])
                target[:, :n_cond] = uint8_to_normalized_float(image[:, :n_cond])
                target[:, n_cond:] = target[:, n_cond - 1: n_cond].expand(-1, chunk - n_cond, -1, -1)
            batch = build_transfer_batch(
                control_videos=[ctrl], target_video=target, num_frames=chunk, height=height, width=width, fps=fps,
                num_conditional_frames=n_cond, temporal_compression_factor=tcf, prompt_key=model.input_caption_key,
                prompt=prompt, negative_prompt=negative,
                share_vision_temporal_positions=bool(args.share_vision_temporal_positions),
            )
            out = model.generate_samples_from_batch(
                batch, sampler=None, guidance=args.guidance, guidance_interval=args.guidance_interval,
                control_guidance=args.control_guidance, control_guidance_interval=args.control_guidance_interval,
                seed=[args.seed + i], n_sample=1, has_negative_prompt=negative is not None, num_steps=args.num_steps,
                shift=args.shift, sigma_max=args.sigma_max, normalize_cfg=args.normalize_cfg,
            )
            outputs.append(model.decode(out["vision"][-1]).clamp(-1, 1).cpu())
        video = torch.cat(outputs, dim=2)[:, :, :total]
        return VideoOutput(((1.0 + video.squeeze(0)) / 2).clamp(0, 1), fps=fps, quality=args.video_save_quality)
