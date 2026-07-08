"""
Extracts DINOv3 patch features and monocular depth maps for nuPlan camera frames,
and caches them to disk (one .npz per frame) for use as image-conditioning inputs
(see nn_modules/dit.py's DiT._prepare_image_tokens, which consumes 'dino_feats'/'depths').

Example (single GPU):
  python extract_dino_depth_features.py --split=test --pkl-root=<sledge_raw dir> --img-root=<nuplan sensor_blobs dir>

Example (multi-GPU via torchrun):
  torchrun --standalone --nnodes=1 --nproc_per_node=6 extract_dino_depth_features.py \
      --split=test --pkl-root=<sledge_raw dir> --img-root=<nuplan sensor_blobs dir>
"""
import os
import gc
import glob
import pickle
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy import typing as npt
from numpy.linalg import inv
from pyquaternion import Quaternion
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ["TRANSFORMERS_NO_TF"] = "1"; os.environ["TRANSFORMERS_NO_FLAX"] = "1"; os.environ["TRANSFORMERS_NO_JAX"] = "1"
from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


DEFAULT_CAM_ORDER = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]

DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


# ===========================
# Distributed utils
# ===========================
def get_rank_worldsize():
    # Works with torchrun and also manual single-proc
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, rank, world_size


def ddp_init_if_needed():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


# ===========================
# DINOv3 feature extractor (HuggingFace)
# ===========================
class DINOv3PatchFeatures(nn.Module):
    """
    HuggingFace DINOv3 -> spatial features.
    Supports ViT (tokens -> grid) and ConvNeXt (already [B,C,H,W]).
    """
    def __init__(self, model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
                 resize_feats: bool = False,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.resize_feats = resize_feats
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id, torch_dtype=torch.float16)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

        # Nominal input (usually square)
        # Prefer processor.size (dict), fall back to model.config.image_size
        size = self.processor.size
        if isinstance(size, dict):
            self.input_h = int(size.get("height", size.get("shortest_edge", 224)))
            self.input_w = int(size.get("width", size.get("shortest_edge", 224)))
        else:
            self.input_h = self.input_w = int(getattr(self.model.config, "image_size", 224))

        # For ViT variants we need patch size and #register tokens
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.num_reg = int(getattr(self.model.config, "num_register_tokens", 0))

        self.is_convnext = self.model.__class__.__name__.lower().startswith("dinov3convnext")

        print(f"[DINOv3PatchFeatures] {model_id} — input {self.input_h}x{self.input_w}, "
              f"{'ConvNeXt' if self.is_convnext else 'ViT'}")

    @torch.inference_mode()
    @torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    def forward(self, x_bnc3hw: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        x_bnc3hw: [B,N,3,H,W] float in [0..1] or [0..255]
        Returns:
          feats [B,N,C,Hf,Wf], and (H_in, W_in) used by the processor
        """
        B, N, C, H, W = x_bnc3hw.shape
        Bn = B * N

        # Convert to a list of HWC arrays for the processor (it handles resize/normalize)
        imgs = []
        flat = x_bnc3hw.view(Bn, C, H, W)
        for i in range(Bn):
            t = flat[i]
            if t.max() <= 1.5:
                arr = (t.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = (t.permute(1, 2, 0).detach().cpu().numpy()).clip(0, 255).astype(np.uint8)
            imgs.append(arr)

        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(**inputs)  # for ViT: BaseModelOutputWithPooling; ConvNeXt similar API

        if self.is_convnext:
            # last_hidden_state: [Bn, C, Hf, Wf]
            feat_BnC_hw = out.last_hidden_state
            C, Hf, Wf = feat_BnC_hw.shape[1], feat_BnC_hw.shape[2], feat_BnC_hw.shape[3]
            feats = feat_BnC_hw.view(B, N, C, Hf, Wf).contiguous()
            return feats, (self.input_h, self.input_w)

        # ViT: last_hidden_state: [Bn, 1 + num_reg + HW, C]
        tok = out.last_hidden_state                      # [Bn, T, C]
        T_all = tok.shape[1]
        T_keep = T_all - (1 + self.num_reg)               # drop cls + registers
        toks_wo = tok[:, 1 + self.num_reg:, :]            # [Bn, HW, C]
        Hf = self.input_h // self.patch_size
        Wf = self.input_w // self.patch_size
        assert Hf * Wf == T_keep, f"Token/grid mismatch: HW={Hf*Wf}, tokens={T_keep}"

        feat_BnC_hw = toks_wo.transpose(1, 2).contiguous().view(Bn, toks_wo.shape[-1], Hf, Wf)
        if self.resize_feats and (Hf != self.input_h or Wf != self.input_w):
            feat_BnC_hw = F.interpolate(
                feat_BnC_hw,
                size=(self.input_h, self.input_w),
                mode="bilinear",
                align_corners=False,
            )
            Hf, Wf = self.input_h, self.input_w
        feats = feat_BnC_hw.view(B, N, -1, Hf, Wf).contiguous()
        return feats, (self.input_h, self.input_w)


# ===========================
# Depth estimator from Hugging Face (ZoeDepth)
# ===========================
ZOEDEPTH_MODEL_ID = "Intel/zoedepth-nyu-kitti"


class HFDepthEstimator(nn.Module):
    """
    HuggingFace ZoeDepth wrapper. Produces [B,N,H,W] float32 metric depth (meters).
    """
    def __init__(self,
                 device: Optional[torch.device] = None,
                 resize_depth=None):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(ZOEDEPTH_MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(ZOEDEPTH_MODEL_ID, torch_dtype=torch.float16)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        self.resize_depth = resize_depth

        print(f"[HFDepthEstimator] '{ZOEDEPTH_MODEL_ID}'")

    @torch.inference_mode()
    @torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    def forward(self, images_bnc3hw: torch.Tensor) -> torch.Tensor:
        B, N, _, H, W = images_bnc3hw.shape
        imgs_flat = images_bnc3hw.view(B * N, 3, H, W)

        imgs_list = []
        for i in range(B * N):
            x = imgs_flat[i]
            if x.max() <= 1.5:
                arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = x.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
            imgs_list.append(arr)

        inputs = self.processor(images=imgs_list, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        if self.resize_depth:
            outputs = self.processor.post_process_depth_estimation(
                outputs,
                source_sizes=[(H, W)] * (B * N)
            )
            d = torch.stack([out['predicted_depth'] for out in outputs], dim=0)  # [B*N, H, W]
        else:
            d = outputs.predicted_depth  # [B*N, H', W']
        depths = d.view(B, N, d.shape[1], d.shape[2]).to(images_bnc3hw.device)
        return depths


# ===========================
# End-to-end wrapper: images -> DINO features + depth
# ===========================
class DinoDepthExtractor(nn.Module):
    """
    Pipeline: images (B, N, 3, H, W) -> DINOv3 patch features + monocular depth map.
    """
    def __init__(self,
                 feat_backbone_id: str = DINOV3_MODEL_ID,
                 resize_feats: bool = False,   # resize DINOv3 tokens to input image size
                 resize_depth: bool = False):
        super().__init__()
        self.backbone = DINOv3PatchFeatures(model_id=feat_backbone_id, resize_feats=resize_feats)
        # Use the backbone's device to avoid free-variable 'device'
        depth_device = self.backbone.device
        self.depth_estimator = HFDepthEstimator(
            device=depth_device,
            resize_depth=resize_depth
        )

    @torch.inference_mode()
    def forward(
        self,
        images_bnc3hw: torch.Tensor,      # [B, N, 3, H, W], RGB uint8 or float in [0..1]
        depths_bnhw: torch.Tensor = None,  # [B, N, H, W], optionally provide your own depth
    ):
        feats, _ = self.backbone(images_bnc3hw)  # [B, N, C, Hf, Wf]
        if depths_bnhw is None:
            depths_bnhw = self.depth_estimator(images_bnc3hw)   # [B, N, H, W]
        return depths_bnhw, feats


# ===========================
# Image loading helper
# ===========================
def load_image_as_tensor(path: str, device) -> torch.Tensor:
    """
    Load an image from disk and convert to a [3, H, W] float32 tensor in [0..1].
    """
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return img_t.to(device)


def trans_matrix_inv(rotation, translation) -> npt.NDArray[np.float64]:
    """
    Get the inverse transformation matrix.
    :return: <np.float: 4, 4>. Inverse transformation matrix.
    """
    tm: npt.NDArray[np.float64] = np.eye(4)
    rot = Quaternion(rotation)
    rot_inv = rot.rotation_matrix.T
    tm[:3, :3] = rot_inv
    tm[:3, 3] = rot_inv.dot(np.transpose(-np.array(translation)))
    return tm


def trans_matrix(rotation, translation) -> npt.NDArray[np.float64]:
    """
    Get the transformation matrix.
    :return: <np.float: 4, 4>. Transformation matrix.
    """
    rot = Quaternion(rotation)
    tm: npt.NDArray[np.float64] = rot.transformation_matrix
    tm[:3, 3] = translation
    return tm


# ===========================
# PyTorch Dataset for nuPlan multi-cam PKL samples
# ===========================
def _safe_torch_load(p: Path):
    try:
        return torch.load(p, map_location="cpu")
    except Exception:
        with open(p, "rb") as f:
            return pickle.load(f)


def _order_cam_keys(cam_info: dict, preferred=DEFAULT_CAM_ORDER):
    keys = list(cam_info.keys())
    ordered = [k for k in preferred if k in cam_info]
    return ordered


def _extract_extrinsic(entry, ego_pose) -> np.ndarray:
    T_ego2cam = trans_matrix_inv(entry["rotation"], entry["translation"])
    T_global2ego_cam = trans_matrix_inv(entry["ego_pose"][3:], entry["ego_pose"][:3])
    T_ego2global = trans_matrix(ego_pose[1], ego_pose[0])
    T_extr = T_ego2cam @ T_global2ego_cam @ T_ego2global
    return inv(T_extr)


class NuPlanCamPKLDataset(Dataset):
    """
    Yields a dict per sample:
      {
        'images':  FloatTensor [N,3,H,W] in [0..1]
        'K':       FloatTensor [N,3,3]
        'T':       FloatTensor [N,4,4]  (extrinsics)
        'cam_names': list of str length N
        'img_hw':  (H, W)
        'meta':    {'idx': int or None, 'basename': str, 'split': str, 'pkl_path': str}
      }

    Expects pkl_root/{split}/*.pkl files, each a dict with 'cam_info' (per-camera
    filename/intrinsic/extrinsic entries) and 'ego_state_og' fields.
    """
    def __init__(
        self,
        pkl_root: str,
        split: str,
        img_root: str,
        pattern: str = "*.pkl",
        only_idx: int = None,
        only_pkl: str = None,
        cam_order: list = DEFAULT_CAM_ORDER,
        front_only: bool = False,
    ):
        super().__init__()
        self.img_root = Path(img_root)
        self.cam_order = cam_order
        self.front_only = front_only

        pkl_root = Path(pkl_root)
        if only_pkl:
            self.paths = [Path(only_pkl)]
            self.split = Path(only_pkl).parent.name
        else:
            splits = ["train", "val", "test"] if split == "all" else [split]
            self.paths = []
            for sp in splits:
                self.paths += [Path(p) for p in sorted(glob.glob(str(pkl_root / sp / pattern)))]
            self.split = split

        # Optional filtering by dataset idx
        if only_idx is not None:
            filtered = []
            for p in self.paths:
                try:
                    d = _safe_torch_load(p)
                    if int(d.get("idx", -1)) == int(only_idx):
                        filtered.append(p)
                except Exception:
                    pass
            self.paths = filtered

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No PKLs found (root={pkl_root}, split={split}, pattern={pattern}, only_idx={only_idx}, only_pkl={only_pkl})")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        pkl_path = self.paths[i]
        d = _safe_torch_load(pkl_path)
        cam_info = d["cam_info"]
        names = ['CAM_F0'] if self.front_only else _order_cam_keys(cam_info, self.cam_order)

        imgs, Ks, Ts, ok_names = [], [], [], []
        H = W = None
        for name in names:
            info = cam_info[name]
            f_abs = self.img_root / info["filename_jpg"]
            if not f_abs.exists():
                # Skip missing image; keep variable N (safe with batch_size=1)
                print(f"[WARN] missing image: {f_abs}")
                continue

            img_t = load_image_as_tensor(str(f_abs), "cpu")   # [3,H,W], 0..1
            if H is None:
                H, W = img_t.shape[-2], img_t.shape[-1]
            imgs.append(img_t)

            Ks.append(torch.from_numpy(np.array(info["intrinsic"], dtype=np.float32)))
            Ts.append(torch.from_numpy(_extract_extrinsic(info, d["ego_state_og"]).astype(np.float32)))
            ok_names.append(name)

        if len(imgs) == 0:
            raise RuntimeError(f"No cameras available for sample {pkl_path}")

        sample = {
            "images": torch.stack(imgs, dim=0),  # [N,3,H,W]
            "K": torch.stack(Ks, dim=0),         # [N,3,3]
            "T": torch.stack(Ts, dim=0),         # [N,4,4]
            "cam_names": ok_names,
            "img_hw": (H, W),
            "meta": {
                "idx": d.get("idx", None),
                "basename": pkl_path.stem,
                "split": pkl_path.parent.name,
                "pkl_path": str(pkl_path),
            },
        }
        return sample


# ===========================
# CLI / Main
# ===========================
def run_dataset_loop(
    dataset: Dataset,
    batch_size: int,
    out_root: str,
    feat_backbone_id: str,
    resize_feats: bool,
    resize_depth: bool,
    num_workers: int,
    prefetch_factor: int,
    limit: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, rank, world_size = get_rank_worldsize()

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
        print(f"[R{rank}] Using DistributedSampler with world_size={world_size}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if prefetch_factor > 0 else None,
        pin_memory=True,
        persistent_workers=True,
    )

    model = DinoDepthExtractor(
        feat_backbone_id=feat_backbone_id,
        resize_feats=resize_feats,
        resize_depth=resize_depth,
    ).to(device)

    out_root = Path(out_root)

    processed = 0
    for batch in tqdm(loader, desc=f"R{rank} {dataset.split}", ncols=100):
        filenames = batch["meta"]["basename"]
        images_bnc3hw = batch["images"].to(device, non_blocking=True)  # [B,N,3,H,W] in [0..1]
        meta = batch["meta"]

        sample_out_dir = out_root / meta['split'][0]
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        if all((sample_out_dir / f"{f}_dino_depths.npz").exists() for f in filenames):
            print(f"[SKIP] {meta['pkl_path']} (already exists)")
            continue

        with torch.inference_mode():
            depths, dino_feats = model(images_bnc3hw)
            depths = depths.detach().cpu()          # [B,N,H,W]
            dino_feats = dino_feats.detach().cpu()  # [B,N,C,Hf,Wf]

        for i, filename in enumerate(filenames):
            np.savez_compressed(
                sample_out_dir / f"{filename}_dino_depths.npz",
                dino_feats=dino_feats[i].numpy(),
                depths=depths[i].numpy()
            )
        print(f"[OK] {meta['pkl_path']} → {sample_out_dir}")
        processed += 1
        if limit and processed >= limit:
            break
        del depths, dino_feats, images_bnc3hw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()  # optional sync at end
    print(f"[DONE] Processed {processed} sample(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl-root", type=str, required=True,
                        help="Directory of per-frame nuPlan camera-info pkls, e.g. "
                             "$DATASET_ROOT/scenario_control_ae_preprocess_nuplan_3dtemp")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=8, help="Keep batch_size=1 for simplicity (variable N per sample).")
    parser.add_argument("--img-root", type=str, required=True,
                         help="Root directory of the raw nuPlan sensor_blobs (e.g. <nuplan-v1.1>/sensor_blobs)")
    parser.add_argument("--pattern", type=str, default="*_0.pkl")
    parser.add_argument("--front-only", type=bool, default=True, help="Only extract features for CAM_F0.")
    parser.add_argument("--only-idx", type=int, default=None)
    parser.add_argument("--only-pkl", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=os.path.join(os.environ.get("SCRATCH_ROOT", "."), "dino_depth_features", "nuplan"))
    parser.add_argument("--resize-feats", type=str, default="false", choices=["true", "false"], help="Resize DINO feats to the input image size (slower, finer) or keep at backbone output size (faster, coarser).")
    parser.add_argument("--resize-depth", type=str, default="true", choices=["true", "false"], help="Resize depth output to the input image size (slower, finer) or keep at backbone output size (faster, coarser).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.feat_backbone_id = DINOV3_MODEL_ID
    args.out_root = os.path.join(args.out_root, "dinov3_zoedepth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank, rank, world_size = ddp_init_if_needed()
    print(f"[proc {rank}/{world_size-1}] using device cuda:{local_rank}")

    ds = NuPlanCamPKLDataset(
        pkl_root=args.pkl_root,
        split=args.split,
        img_root=args.img_root,
        pattern=args.pattern,
        only_idx=args.only_idx,
        only_pkl=args.only_pkl,
        front_only=args.front_only,
    )

    run_dataset_loop(
        dataset=ds,
        batch_size=args.batch_size,
        out_root=args.out_root,
        feat_backbone_id=args.feat_backbone_id,
        resize_feats=args.resize_feats == "true",
        resize_depth=args.resize_depth == "true",
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        limit=args.limit,
    )
