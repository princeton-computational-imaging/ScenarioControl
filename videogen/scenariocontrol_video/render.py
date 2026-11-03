"""Wireframe control renderer: BEV rollout frames -> CAM_F0 wireframe control video.

The geometry, palette, line width, depth dimming and draw order are the ones
the released checkpoints were trained with (1280x720, width-6 anti-aliased lines,
lanes then boxes, far to near). Two changes from the training renderer: agent
centre dots are not drawn, and boxes that straddle the camera (e.g. the ego hood
box) are redrawn after near-plane clipping. Fewer than 0.3% of pixels differ.

Input: a directory of ``frame_XXXX.pkl`` files as written by the simulator's
``SequenceSaver`` (see the README "Simulation" section). Each pickle holds
``agents`` ``[N,10]`` or ``[N,1,10]`` (x, y, vx, vy, theta, length, width,
exists, z, height; ego in the last row), ``agent_types`` ``[N,5]`` one-hot,
``lanes`` ``[L,50,2]`` (all in the current ego frame), optional ``agent_active``
``[N]`` (inactive rows are dropped), optional ``ego_state_og`` (global translation,
quaternion, dims, yaw) and optional ``cam_infos``. Frames without ``cam_infos`` (or
without ``ego_state_og``) use the nuPlan CAM_F0 rig below placed at the ego pose;
this approximation ignores the real camera height and pitch.
"""

from __future__ import annotations

import pickle
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

LINE_WIDTH = 6
ALPHA_NEAR, ALPHA_FAR, ALPHA_MIN = 5.0, 80.0, 0.15
RENDER_HEIGHT = 720
COLORS = {  # RGB, the training palette
    "vehicle": (255, 0, 0),
    "pedestrian": (0, 255, 0),
    "cyclist": (255, 255, 0),
    "ego": (255, 0, 0),
    "lane": (98, 183, 249),
}
OTHER_COLOR = (1.0, 0.75, 0.8)  # agents with no known type
BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
NEAR_Z = 0.15

# nuPlan CAM_F0 rig (camera-to-ego), used when a frame carries no cam_infos.
CANONICAL_CAM_F0 = {
    "rotation": [-0.4909512705383546, 0.510777513391982, -0.5008766479792224, 0.49718785713146496],
    "translation": [1.6527723767759348, -0.007276292123089233, 1.5056675432939306],
    "intrinsic": [[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]],
    "width": 1920,
    "height": 1080,
}


# ---------------------------------------------------------------- geometry --
def _transform(points, T):
    homog = torch.hstack((points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)))
    return (T.to(points.dtype) @ homog.T).T[:, :3]


def _quat_to_rot(q):
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=q.dtype,
        device=q.device,
    )


def _trans_matrix(q, t):
    tm = torch.eye(4, dtype=q.dtype, device=q.device)
    tm[:3, :3] = _quat_to_rot(q)
    tm[:3, 3] = t
    return tm


def _trans_matrix_inv(q, t):
    R_inv = _quat_to_rot(q).T
    tm = torch.eye(4, dtype=q.dtype, device=q.device)
    tm[:3, :3] = R_inv
    tm[:3, 3] = R_inv @ (-t)
    return tm


def _box_corners(center, heading, dims, dtype):
    local = torch.tensor(
        [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
         [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]],
        dtype=dtype,
    )
    corners = torch.zeros((center.shape[0], 8, 3), dtype=dtype)
    for i in range(center.shape[0]):
        l, w, h = dims[i]
        x, y, z = center[i]
        scaled = local * torch.tensor([l, w, h], dtype=dtype)
        c, s = torch.cos(heading[i]), torch.sin(heading[i])
        R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=dtype)
        corners[i] = scaled @ R.T + torch.tensor([x, y, z], dtype=dtype)
    return corners


def _project_nodrop(P_cam, K, image_size):
    W, H = image_size
    z = P_cam[:, 2]
    eps = 1e-6
    u = K[0, 0] * (P_cam[:, 0] / torch.clamp(z, eps, None)) + K[0, 2]
    v = K[1, 1] * (P_cam[:, 1] / torch.clamp(z, eps, None)) + K[1, 2]
    mask = (z > eps) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return torch.stack([u, v], axis=1), mask


def _clip_line(x0, y0, x1, y1, width, height):
    """Liang-Barsky clip to [0,width-1]x[0,height-1]; None if fully outside."""
    dx, dy = x1 - x0, y1 - y0
    u1, u2 = 0.0, 1.0
    for pi, qi in zip([-dx, dx, -dy, dy], [x0, width - 1 - x0, y0, height - 1 - y0]):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t
    return x0 + u1 * dx, y0 + u1 * dy, x0 + u2 * dx, y0 + u2 * dy


def _heading_global(agent_heading, ego_heading):
    return (agent_heading + ego_heading + torch.pi) % (2 * torch.pi) - torch.pi


def _agent_color(i, types):
    if i == 0:
        return COLORS["ego"]
    if types[i, 0]:
        return COLORS["vehicle"]
    if types[i, 1]:
        return COLORS["pedestrian"]
    if types[i, 2]:
        return COLORS["cyclist"]
    return OTHER_COLOR


def _depth_alpha(depth):
    if depth is None or not np.isfinite(depth):
        return 1.0
    return float(np.clip(1.0 - (depth - ALPHA_NEAR) / max(ALPHA_FAR - ALPHA_NEAR, 1e-3), ALPHA_MIN, 1.0))


def _scale(rgb, alpha):
    return tuple(int(np.clip(round(c * alpha), 0, 255)) for c in rgb)


def _draw_segments(frame, segments):
    segments.sort(key=lambda s: s[0], reverse=True)  # far to near
    for _, x0, y0, x1, y1, color in segments:
        cv2.line(frame, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), color,
                 thickness=LINE_WIDTH, lineType=cv2.LINE_AA)


# ------------------------------------------------------------------ camera --
class _Camera:
    def __init__(self, cam, ego_state_og, dtype):
        f = lambda v: v.to(dtype=dtype) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=dtype)  # noqa: E731
        ego_pose = f(cam["ego_pose"])
        self.T_local2global = _trans_matrix(f(ego_state_og[1]), f(ego_state_og[0]))
        self.T_cam_tf_inv = _trans_matrix_inv(f(cam["rotation"]), f(cam["translation"]))
        self.T_cam_ego_inv = _trans_matrix_inv(ego_pose[3:], ego_pose[:3])
        intr = f(cam["intrinsic"])
        scale = RENDER_HEIGHT / cam["height"]
        self.width, self.height = int(round(cam["width"] * scale)), RENDER_HEIGHT
        if abs(scale - 1.0) >= 1e-6:
            intr = intr.clone()
            intr[0, 0] *= scale
            intr[1, 1] *= scale
            intr[0, 2] *= scale
            intr[1, 2] *= scale
        else:
            self.width, self.height = int(cam["width"]), int(cam["height"])
        self.intrinsics = intr

    def to_cam(self, pts):
        return _transform(_transform(_transform(pts, self.T_local2global), self.T_cam_ego_inv), self.T_cam_tf_inv)


# ---------------------------------------------------------------- renderer --
def render_frame(agents, agent_types, lanes, cam, ego_state_og) -> np.ndarray:
    """Render one wireframe frame (H, W, 3) uint8 RGB.

    ``agents``: [N, 9] torch (x, y, z, speed, cos, sin, length, width, height), ego first.
    ``agent_types``: [N, 3] (vehicle, pedestrian, cyclist). ``lanes``: [L, P, 2].
    ``cam``: CAM_F0 dict (rotation, translation, intrinsic, width, height, ego_pose).
    """
    dtype = agents.dtype
    cam = _Camera(cam, ego_state_og, dtype)
    w, h = cam.width, cam.height
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Lanes at z=0; an edge is kept when both endpoints are in front of the camera.
    n_lanes, n_pts, _ = lanes.shape
    lane_pts = torch.cat([lanes, torch.ones((n_lanes, n_pts, 1)) * 0.0], axis=-1).reshape(-1, 3)
    lane_cam = cam.to_cam(lane_pts)
    lane_depth = lane_cam[:, 2]
    lane_uv, _ = _project_nodrop(lane_cam, cam.intrinsics, (w, h))
    lane_uv, lane_depth_np = lane_uv.numpy(), lane_depth.numpy()
    segments = []
    for li in range(n_lanes):
        for k in range(li * n_pts, li * n_pts + n_pts - 1):
            if not (lane_depth[k] > 0 and lane_depth[k + 1] > 0):
                continue
            clipped = _clip_line(lane_uv[k][0], lane_uv[k][1], lane_uv[k + 1][0], lane_uv[k + 1][1], w, h)
            if clipped is None:
                continue
            depth = 0.5 * (lane_depth_np[k] + lane_depth_np[k + 1])
            segments.append((float(depth), *clipped, _scale(COLORS["lane"], _depth_alpha(depth))))
    if segments:
        _draw_segments(frame, segments)

    if agents.shape[0] == 0:
        return frame
    heading = _heading_global(torch.atan2(agents[:, 5], agents[:, 4]), torch.as_tensor(ego_state_og[3], dtype=dtype))
    corners_global = _box_corners(_transform(agents[:, :3], cam.T_local2global.to(dtype)), heading, agents[:, 6:], dtype)
    corners_cam = _transform(_transform(corners_global.reshape(-1, 3), cam.T_cam_ego_inv), cam.T_cam_tf_inv)
    uv, vis = _project_nodrop(corners_cam, cam.intrinsics, (w, h))
    uv, vis = uv.reshape(-1, 8, 2).numpy(), vis.reshape(-1, 8).numpy().astype(bool)
    cc = corners_cam.reshape(-1, 8, 3).numpy()
    types = agent_types.numpy()

    # Boxes: an edge is drawn when either corner projects inside the image.
    segments = []
    for i in range(uv.shape[0]):
        base = tuple(int(c * 255) if c <= 1.0 else int(c) for c in _agent_color(i, types))
        for e0, e1 in BOX_EDGES:
            if not (vis[i, e0] or vis[i, e1]):
                continue
            clipped = _clip_line(uv[i, e0][0], uv[i, e0][1], uv[i, e1][0], uv[i, e1][1], w, h)
            if clipped is None:
                continue
            depth = 0.5 * (cc[i, e0, 2] + cc[i, e1, 2])
            segments.append((float(depth), *clipped, _scale(base, _depth_alpha(depth))))
    if segments:
        _draw_segments(frame, segments)

    # Redraw boxes that straddle the camera (some corner behind the near plane),
    # clipped to z >= NEAR_Z, on top of everything (the ego hood box).
    K = cam.intrinsics.numpy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for i in range(cc.shape[0]):
        if not np.any(cc[i, :, 2] < NEAR_Z):
            continue
        base = tuple(int(round(c * 255)) if c <= 1.0 else int(round(c)) for c in _agent_color(i, types))
        for e0, e1 in BOX_EDGES:
            p0, p1 = cc[i, e0], cc[i, e1]
            z0, z1 = float(p0[2]), float(p1[2])
            if z0 < NEAR_Z and z1 < NEAR_Z:
                continue
            if z0 < NEAR_Z or z1 < NEAR_Z:
                pc = p0 + (NEAR_Z - z0) / (z1 - z0) * (p1 - p0)
                p0, p1 = (pc, p1) if z0 < NEAR_Z else (p0, pc)
            if p0[2] <= 1e-6 or p1[2] <= 1e-6:
                continue
            clipped = _clip_line(fx * p0[0] / p0[2] + cx, fy * p0[1] / p0[2] + cy,
                                 fx * p1[0] / p1[2] + cx, fy * p1[1] / p1[2] + cy, w, h)
            if clipped is None:
                continue
            depth = 0.5 * (max(z0, NEAR_Z) + max(z1, NEAR_Z))
            cv2.line(frame, (int(round(clipped[0])), int(round(clipped[1]))),
                     (int(round(clipped[2])), int(round(clipped[3]))), _scale(base, _depth_alpha(depth)),
                     thickness=LINE_WIDTH, lineType=cv2.LINE_AA)
    return frame


# ------------------------------------------------------------- BEV frames --
def _frame_inputs(raw: dict):
    """SequenceSaver frame dict -> render_frame arguments (ego moved to row 0)."""
    agents = raw["agents"]
    if agents.ndim == 3:
        agents = agents[:, 0, :]
    agents = np.vstack([agents[-1:], agents[:-1]])
    x, y, theta = agents[:, 0], agents[:, 1], agents[:, 4]
    length, width, height = agents[:, 5].copy(), agents[:, 6].copy(), agents[:, 9].copy()
    ego_state = raw.get("ego_state_og")
    if ego_state is None:
        # No global pose: render in the ego frame with the canonical rig.
        ego_state = [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.array([length[0], width[0], height[0]]), 0.0]
        raw = {k: v for k, v in raw.items() if k not in ("cam_infos", "cam_info")}
    length[0], width[0], height[0] = ego_state[2][0], ego_state[2][1], ego_state[2][2]
    vel = np.sqrt(agents[:, 2] ** 2 + agents[:, 3] ** 2)
    states = np.stack([x, y, agents[:, 8], vel, np.cos(theta), np.sin(theta), length, width, height], axis=1)

    if "agent_active" in raw:
        mask = raw["agent_active"].copy()
        mask = np.concatenate([mask[-1:], mask[:-1]])
    else:
        mask = (x >= 0) & (x < 80.0) & (np.abs(y) < 40.0)
    mask[0] = True

    t5 = raw["agent_types"]
    t5 = np.vstack([t5[-1:], t5[:-1]])
    t3 = np.zeros((len(t5), 3))
    t3[:, 0], t3[:, 1] = t5[:, 1], t5[:, 2]
    t3[:, 2] = np.clip(t5[:, 3] + t5[:, 4], 0, 1)  # cyclist + other share one colour

    lanes = raw["lanes"]
    if lanes.ndim == 2:
        lanes = lanes.reshape(lanes.shape[0] // 50, 50, 2)
    elif lanes.shape[-1] > 2:
        lanes = lanes[:, :, :2]

    cams = raw.get("cam_infos") or raw.get("cam_info") or {}
    cam = cams.get("CAM_F0") if isinstance(cams, dict) else None
    if not (isinstance(cam, dict) and cam.get("ego_pose") is not None and cam.get("rotation") is not None
            and cam.get("intrinsic") is not None and np.ravel(cam["ego_pose"]).shape[0] >= 7):
        cam = dict(CANONICAL_CAM_F0, ego_pose=list(np.ravel(ego_state[0])[:3]) + list(np.ravel(ego_state[1])[:4]))
    return torch.from_numpy(states[mask]), torch.from_numpy(t3[mask]), torch.from_numpy(lanes), cam, ego_state


def render_bev_frames(frames_dir, num_frames: int | None = None) -> list[np.ndarray]:
    """Render every ``frame_*.pkl`` in ``frames_dir`` (sorted) to wireframes."""
    pkls = sorted(Path(frames_dir).glob("frame_*.pkl"))[:num_frames]
    if not pkls:
        raise FileNotFoundError(f"no frame_*.pkl in {frames_dir}")
    out = []
    for p in pkls:
        with open(p, "rb") as f:
            out.append(render_frame(*_frame_inputs(pickle.load(f))))
    return out


def write_video(frames, path, fps: int = 10) -> Path:
    """Encode RGB uint8 frames as H.264 (crf 23, yuv420p), as in training."""
    import imageio_ffmpeg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(frames):
            cv2.imwrite(f"{tmp}/frame_{i:06d}.png", cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_RGB2BGR))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", f"{tmp}/frame_%06d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                        str(path)], check=True)
    return path


def make_control_video(frames_dir, out_path, image=None, num_frames: int | None = None) -> Path:
    """BEV frame dir -> control video. Frame 0 is ``image`` (the real first frame,
    resized to 1280x720 with INTER_AREA) when given, as in training; else a wireframe."""
    frames = render_bev_frames(frames_dir, num_frames)
    if image is not None:
        img = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        h, w = frames[0].shape[:2]
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        frames[0] = img
    return write_video(frames, out_path)
