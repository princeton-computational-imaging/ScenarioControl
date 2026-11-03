# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Modified by the ScenarioControl authors: `OmniMoTControlConcatModel.build_net` is a copy
# of `cosmos_framework.model.vfm.omni_mot_model.OmniMoTModel.build_net`
# (NVIDIA/cosmos-framework@fa63e9934a38eb8127a0c3d6b23969fd4d0e32a7) with the network class
# swapped and the new control layers unfrozen after LoRA injection. Everything else is new.
"""Channel-concat wireframe control for Cosmos3-Nano.

The wireframe control video is VAE-encoded like the target and added to the noisy
target tokens through a zero-initialised projection::

    tokens = vae2llm(patchify(noisy_target)) + control_embedder(patchify(control))

This equals widening the patch embedding to take ``[target, control]`` channels
(Wan-Fun style), without changing any pretrained weight shape. The first-frame
variant adds a second zero-initialised projection, ``inpaint_embedder``, that
takes the anchor as ``[mask, masked_video_latent]`` channels; no latent frame is
held clean and every frame is denoised.

Batches carry ``video=[control, target]``. ``get_data_and_condition`` moves the
control latent out of the sequence into a stash, and ``denoise`` (used by both
training and sampling, including both CFG branches) attaches it to the packed
sequence, where ``_encode_vision`` adds it to the target tokens.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from cosmos_framework.model.vfm.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.vfm.mot.parallelize_vfm_network import parallelize_vfm_network
from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel
from cosmos_framework.model.vfm.utils.data_and_condition import GenerationDataClean
from cosmos_framework.utils import misc
from cosmos_framework.utils.flags import DEVICE, Device
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

TEMPORAL_COMPRESSION = 4  # Wan2.2 VAE: latent 0 <-> pixel 0, latent k <-> pixels 4k-3..4k


class ControlConcatCosmos3VFMNetwork(Cosmos3VFMNetwork):
    """Cosmos3VFMNetwork plus the zero-init ``control_embedder`` (and ``inpaint_embedder``)."""

    def __init__(self, language_model, config: Cosmos3VFMNetworkConfig, inpaint_anchor: bool = False):
        super().__init__(language_model=language_model, config=config)
        assert config.vision_gen, "control-concat requires vision_gen"
        # No bias: the widened patch embedding has one bias, which stays in vae2llm.
        self.control_embedder = nn.Linear(self.patch_latent_dim, self.hidden_size, bias=False)
        self.inpaint_anchor = bool(inpaint_anchor)
        if self.inpaint_anchor:
            p = self.latent_patch_size
            # p*p mask dims (broadcast per frame) + the masked-video latent patch.
            self.inpaint_embedder = nn.Linear(p * p + self.patch_latent_dim, self.hidden_size, bias=False)

    def init_weights(self, buffer_device: torch.device | None):
        # Runs after materialisation, so the zero init sticks: at step 0 the model is the base model.
        super().init_weights(buffer_device=buffer_device)
        torch.nn.init.zeros_(self.control_embedder.weight)
        if self.inpaint_anchor:
            torch.nn.init.zeros_(self.inpaint_embedder.weight)

    def _packed_frame_mask_features(self, masks, latents) -> torch.Tensor:
        """Per-token mask features in ``patchify_and_pack_latents`` order (t-major)."""
        p = self.latent_patch_size
        feats = []
        for m, lat in zip(masks, latents):
            h, w = lat.shape[-2:]
            feats.append(m.repeat_interleave(((h + p - 1) // p) * ((w + p - 1) // p))[:, None].expand(-1, p * p))
        return torch.cat(feats, dim=0)

    def _encode_vision(self, packed_seq, packed_sequence, target_dtype, fps=None):
        original_latent_shapes = super()._encode_vision(packed_seq, packed_sequence, target_dtype, fps)
        vision = packed_seq.vision
        if vision is None or vision.tokens is None:
            return original_latent_shapes
        idx = vision.sequence_indexes

        control_tokens = getattr(vision, "control_tokens", None)
        if control_tokens is not None:
            assert len(control_tokens) == len(vision.tokens)
            for c, t in zip(control_tokens, vision.tokens):
                assert c.shape[-3:] == t.shape[-3:], f"control latent {tuple(c.shape)} != target {tuple(t.shape)}"
            # Same patchify as the target, so control token t lands on target token t.
            packed_control, _ = self.patchify_and_pack_latents(control_tokens, vision.token_shapes)
            emb = self.control_embedder(
                packed_control.to(device=packed_sequence.device, dtype=self.control_embedder.weight.dtype)
            )
            packed_sequence[idx] = packed_sequence[idx] + emb.to(dtype=packed_sequence.dtype)

        inpaint_tokens = getattr(vision, "inpaint_tokens", None)
        if self.inpaint_anchor and inpaint_tokens is not None:
            masks = vision.inpaint_masks
            assert masks is not None and len(inpaint_tokens) == len(vision.tokens)
            packed_inp, _ = self.patchify_and_pack_latents(inpaint_tokens, vision.token_shapes)
            packed_mask = self._packed_frame_mask_features(masks, inpaint_tokens).to(
                device=packed_inp.device, dtype=packed_inp.dtype
            )
            inp = torch.cat([packed_mask, packed_inp], dim=1)  # Wan channel order: [mask, masked video]
            emb = self.inpaint_embedder(inp.to(device=packed_sequence.device, dtype=self.inpaint_embedder.weight.dtype))
            packed_sequence[idx] = packed_sequence[idx] + emb.to(dtype=packed_sequence.dtype)

        return original_latent_shapes


class OmniMoTControlConcatModel(OmniMoTModel):
    """OmniMoTModel that consumes ``video=[control, target]`` batches via channel concat."""

    def __init__(self, config, control_dropout_p: float = 0.1, inpaint_anchor: bool = False):
        assert 0.0 <= control_dropout_p < 1.0, f"control_dropout_p out of range: {control_dropout_p}"
        self.control_dropout_p = control_dropout_p
        self.inpaint_anchor = bool(inpaint_anchor)
        # Refreshed by every get_data_and_condition call, consumed by denoise().
        self._control_latents: list[torch.Tensor] | None = None
        self._inpaint_latents: list[torch.Tensor] | None = None
        self._inpaint_masks: list[torch.Tensor] | None = None
        super().__init__(config)

    def build_net(self, dtype: torch.dtype):
        with torch.device("meta"):
            assert self.vlm_config.model_instance is not None, "Model instance should be specified"

            language_model = lazy_instantiate(self.vlm_config.model_instance)

            num_train_timesteps = self.config.rectified_flow_inference_config.num_train_timesteps
            network_config = Cosmos3VFMNetworkConfig(
                vlm_config=language_model.config,
                latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
                latent_downsample_factor=self.config.latent_downsample_factor,
                latent_channel_size=self.config.state_ch,
                max_latent_h=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_w=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_t=self.config.state_t,
                rope_h_extrapolation_ratio=self.config.diffusion_expert_config.rope_h_extrapolation_ratio,
                rope_w_extrapolation_ratio=self.config.diffusion_expert_config.rope_w_extrapolation_ratio,
                rope_t_extrapolation_ratio=self.config.diffusion_expert_config.rope_t_extrapolation_ratio,
                enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
                base_fps=self.config.diffusion_expert_config.base_fps,
                vision_gen=self.config.vision_gen,
                action_gen=self.config.action_gen,
                sound_gen=self.config.sound_gen,
                position_embedding_type=self.config.diffusion_expert_config.position_embedding_type,
                joint_attn_implementation=self.config.joint_attn_implementation,
                timestep_scale=1.0 / float(num_train_timesteps) * self.config.diffusion_expert_config.timestep_range,
                action_dim=self.config.max_action_dim,
                num_embodiment_domains=self.config.num_embodiment_domains,
                temporal_compression_factor_vision=self.tokenizer_vision_gen.temporal_compression_factor,
                natten_parameter_list=self.config.natten_parameter_list,
                video_temporal_causal=self.config.video_temporal_causal,
                sound_dim=self.config.sound_dim,
                sound_latent_fps=self.config.sound_latent_fps,
            )
            network_config._attn_implementation_internal = "eager"
            net = ControlConcatCosmos3VFMNetwork(
                language_model=language_model, config=network_config, inpaint_anchor=self.inpaint_anchor
            )
            net.pad_for_cuda_graphs = self.config.compile.use_cuda_graphs

            if getattr(self.config, "lora_enabled", False):
                net = self.add_lora(
                    net,
                    lora_rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_target_modules=self.config.lora_target_modules,
                )
                # LoRA injection freezes every non-LoRA parameter; the new control layers train too.
                net.control_embedder.weight.requires_grad_(True)
                if self.inpaint_anchor:
                    net.inpaint_embedder.weight.requires_grad_(True)

        self.install_attention_dispatch(net)

        net = parallelize_vfm_network(
            net,
            parallel_dims=self.parallel_dims,
            compile_config=self.config.compile,
            ac_config=self.config.activation_checkpointing,
        )

        with misc.timer("meta to cuda and broadcast model states"):
            net = net.to(dtype=dtype)
            net.to_empty(device=DEVICE)
            if DEVICE == Device.CUDA:
                net.init_weights(buffer_device=DEVICE)
                if getattr(self.config, "lora_enabled", False):
                    self._init_lora_weights_post_materialization(net)

        return net

    def get_data_and_condition(self, data_batch: dict, iteration: int = 1) -> GenerationDataClean:
        gdc = super().get_data_and_condition(data_batch, iteration=iteration)
        n_items = gdc.num_vision_items_per_sample
        if n_items is None:
            self._control_latents = None
            return gdc
        assert all(n == 2 for n in n_items), f"expected [control, target] per sample, got {n_items}"
        controls = gdc.x0_tokens_vision[0::2]
        targets = gdc.x0_tokens_vision[1::2]
        for c, t in zip(controls, targets):
            assert c.shape == t.shape, f"control latent {tuple(c.shape)} != target latent {tuple(t.shape)}"
        raw = gdc.raw_state_vision
        raw_targets = raw[1::2] if raw is not None else []

        # Control dropout: zero the whole control latent per sample (training only).
        if self.control_dropout_p > 0 and bool(getattr(self.net, "training", False)):
            controls = [torch.zeros_like(c) if random.random() < self.control_dropout_p else c for c in controls]

        self._control_latents = list(controls)
        gdc.x0_tokens_vision = list(targets)
        gdc.raw_state_vision = list(raw_targets) if raw is not None else None
        # From here on the packer sees a standard single-item sample.
        gdc.num_vision_items_per_sample = None

        if self.inpaint_anchor:
            self._build_inpaint_anchor_channels(data_batch, targets, raw_targets)
        return gdc

    def _build_inpaint_anchor_channels(self, data_batch: dict, targets, raw_targets) -> None:
        """Turn each plan's ``condition_frame_indexes_vision`` (latent frames) into anchor channels.

        mask[t] = 1 for anchored latent frames; the masked-video latent is the VAE
        encoding of the target pixels with all non-anchored frames zeroed. The plan's
        indexes are then cleared in place, so no frame is held clean.
        """
        plans = data_batch.get("sequence_plan")
        assert plans is not None and len(plans) == len(targets)
        tcf = TEMPORAL_COMPRESSION
        latents, masks = [], []
        for i, plan in enumerate(plans):
            tgt = targets[i]  # [1, C, T, H, W]
            t_latent = tgt.shape[2]
            idxs = [int(k) for k in (plan.condition_frame_indexes_vision or []) if int(k) < t_latent]
            mask = torch.zeros(t_latent, dtype=torch.float32, device=tgt.device)
            if idxs:
                mask[idxs] = 1.0
                raw = raw_targets[i]  # [1, 3, T, H, W] normalised pixels
                keep = set()
                for k in idxs:
                    keep.update([0] if k == 0 else range(tcf * k - (tcf - 1), tcf * k + 1))
                keep = sorted(f for f in keep if f < raw.shape[2])
                masked = torch.zeros_like(raw)
                masked[:, :, keep] = raw[:, :, keep]
                with torch.no_grad():
                    ml = self.encode(masked).contiguous().float()
                ml = ml[:, :, :, : tgt.shape[3], : tgt.shape[4]]  # same padding crop as the target
                assert ml.shape == tgt.shape, f"inpaint latent {tuple(ml.shape)} != target {tuple(tgt.shape)}"
                latents.append(ml)
            else:
                latents.append(torch.zeros_like(tgt))  # unanchored sample
            masks.append(mask)
            plan.condition_frame_indexes_vision = []
        self._inpaint_latents = latents
        self._inpaint_masks = masks

    def denoise(self, net=None, data_batch_packed=None, fps_vision=None, fps_action=None, fps_sound=None,
                memory=None) -> dict:
        vision = data_batch_packed.vision if data_batch_packed is not None else None
        if vision is not None and self._control_latents is not None:
            assert len(self._control_latents) == len(vision.tokens), (
                "get_data_and_condition must run before every denoise"
            )
            vision.control_tokens = [c.to(device="cuda", non_blocking=True) for c in self._control_latents]
        if self.inpaint_anchor and vision is not None and self._inpaint_latents is not None:
            assert len(self._inpaint_latents) == len(vision.tokens)
            vision.inpaint_tokens = [x.to(device="cuda", non_blocking=True) for x in self._inpaint_latents]
            vision.inpaint_masks = [m.to(device="cuda", non_blocking=True) for m in self._inpaint_masks]
        return super().denoise(
            net=net,
            data_batch_packed=data_batch_packed,
            fps_vision=fps_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
            memory=memory,
        )

