import cv2
import os
import numpy as np
from pyquaternion import Quaternion
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import math
from io import BytesIO
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

def agent_color(a_idx, agent_types, cfg=None):
    """Get RGB color for an agent based on its type.

    Args:
        a_idx: Agent index
        agent_types: Agent type array (N, 3) for [vehicle, pedestrian, cyclist]
        cfg: Optional config with colors attribute

    Returns:
        RGB list [r, g, b] in range [0, 1]
    """
    # Use config colors if available, otherwise use defaults
    if cfg is not None and hasattr(cfg, 'colors'):
        vehicle_color = cfg.colors.vehicle
        pedestrian_color = cfg.colors.pedestrian
        cyclist_color = cfg.colors.cyclist
        ego_color = cfg.colors.ego
    else:
        # Default colors
        vehicle_color = [98, 183, 249]  # light blue
        pedestrian_color = [0, 255, 0] # green
        cyclist_color = [255, 255, 0]  # yellow
        ego_color = [255, 0, 0]  # red

    # Ego vehicle (index 0)
    if a_idx == 0:
        return ego_color

    # Determine agent type
    if agent_types[a_idx, 0]:  # vehicle
        return vehicle_color
    elif agent_types[a_idx, 1]:  # pedestrian
        return pedestrian_color
    elif agent_types[a_idx, 2]:  # cyclist
        return cyclist_color
    else:
        return [1.0, 0.75, 0.8]  # pink fallback


def create_se2_4x4_matrix(origin_translation: np.ndarray, origin_heading: float) -> np.ndarray:
    """
    Creates a 4x4 homogeneous transformation matrix from 2D SE(2) pose.

    :param origin_translation: 2D translation vector (x, y)
    :param origin_heading: heading (theta) in radians
    :return: 4x4 transformation matrix as numpy array
    """
    cos_theta = np.cos(origin_heading)
    sin_theta = np.sin(origin_heading)

    T = np.array([
        [cos_theta, -sin_theta, 0.0, origin_translation[0]],
        [sin_theta,  cos_theta, 0.0, origin_translation[1]],
        [0.0,        0.0,       1.0, origin_translation[2]],
        [0.0,        0.0,       0.0, 1.0     ]
    ], dtype=np.float64)

    return T


def load_img(image_root,filename_jpg, downsample_factor):

    image_path = os.path.join(image_root, filename_jpg)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Camera image not found: {image_path}")

    img = cv2.imread(image_path)  # BGR
    if downsample_factor != 1.0:
        new_size = (int(img.shape[1] / downsample_factor), int(img.shape[0] / downsample_factor))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    return img

def undistort_img(img, camera_intrinsics, camera_distortion, img_size):
    """
    Undistort an image using the camera intrinsics and distortion coefficients."
    img_size: (width, height) tuple representing the size of the image.
    """
    new_K, _ = cv2.getOptimalNewCameraMatrix(camera_intrinsics, camera_distortion, img_size, alpha=0)
    undistorted = cv2.undistort(img, camera_intrinsics, camera_distortion, None, new_K)
    return undistorted



def load_cam_views(cam_infos, cam_order, image_root, do_undistortion=False, downsample_factor=1.0, load_cam_img=False):
    """
    Loads and prepares camera images, ego-to-camera transforms, and intrinsics.

    Args:
        cam_infos (dict): Dictionary with camera info per view.
        cam_order (list of str): List of camera names in desired order.
        image_root (str): Root path to images.

    Returns:
        cam_img_stack (np.ndarray): [N, H, W, 3] array of normalized images.
        T_cam_tf_stack (np.ndarray): [N, 4, 4] transformation matrices from ego to each camera.
        intrinsics_stack (np.ndarray): [N, 3, 3] camera intrinsic matrices.
    """
    cam_imgs = []
    T_cam_tfs_inv = []
    T_cam_tfs = []
    intrinsics = []
    T_cam_egos_inv = []
    widths = []
    heights = []

    for cam in cam_order:
        cam_data = cam_infos.get(cam)
        if load_cam_img:
            cam_img = load_img(image_root, cam_data['filename_jpg'], downsample_factor)
            height, width = cam_img.shape[:2]
        else:
            width, height = cam_data['width'], cam_data['height']
        widths.append(width)
        heights.append(height)

        intrinsic = np.array(cam_data['intrinsic'])
        if load_cam_img:
            if do_undistortion:
                distortion = np.array(cam_data['distortion'])  # unused unless you want to undistort
                cam_img = undistort_img(cam_img, intrinsic, distortion, (width, height))
            
            cam_img = (cam_img / 255.).astype(np.float32, copy=False)
            cam_img = np.transpose(cam_img, (2, 0, 1))           # HWC -> CHW
            cam_imgs.append(cam_img)

        T_cam_tf_inv = trans_matrix_inv(
            np.array(cam_data['rotation']),
            np.array(cam_data['translation'])
        )
        T_cam_tf = trans_matrix(
            np.array(cam_data['rotation']),
            np.array(cam_data['translation'])
        )

        cam_ego_translation = np.array(cam_data['ego_pose'][:3])
        cam_ego_rotation = np.array(cam_data['ego_pose'][3:])
        T_cam_ego = trans_matrix_inv(cam_ego_rotation, cam_ego_translation)
        T_cam_tfs.append(T_cam_tf)
        T_cam_tfs_inv.append(T_cam_tf_inv)
        T_cam_egos_inv.append(T_cam_ego)
        intrinsics.append(intrinsic)

    cam_img_stack = np.stack(cam_imgs, axis=0) if load_cam_img else None        # [N, H, W, 3]
    T_cam_tf_stack = np.stack(T_cam_tfs, axis=0)          # [N, 4, 4]
    T_cam_tf_inv_stack = np.stack(T_cam_tfs_inv, axis=0)        # [N, 4, 4]
    T_cam_ego_inv_stack = np.stack(T_cam_egos_inv, axis=0)      # [N, 4, 4]
    intrinsics_stack = np.stack(intrinsics, axis=0)       # [N, 3, 3]

    return cam_img_stack, T_cam_tf_stack, T_cam_tf_inv_stack, T_cam_ego_inv_stack, intrinsics_stack, widths, heights


def trans_matrix_inv(rotation, translation) -> npt.NDArray[np.float64]:
    """
    Get the inverse transformation matrix.
    :return: <np.float: 4, 4>. Inverse transformation matrix.
    """
    tm: npt.NDArray[np.float64] = np.eye(4)
    rot = Quaternion(rotation)
    rot_inv = rot.rotation_matrix.T
    tm[:3, :3] = rot_inv
    tm[:3, 3] = rot_inv.dot(np.transpose(-translation))
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

def get_3d_box_corners(center_xyz, heading, dimensions):
    """
    center_xyz: (N, 3)
    heading: (N,)
    dimensions: (N, 3) - length, width, height

    Returns:
        corners: (N, 8, 3)
    """
    N = center_xyz.shape[0]

    local_corners = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ])  # (8, 3)

    corners = np.zeros((N, 8, 3))

    for i in range(N):
        l, w, h = dimensions[i]
        x, y, z = center_xyz[i]

        # Scale, rotate, translate
        scaled = local_corners * np.array([l, w, h])
        c, s = np.cos(heading[i]), np.sin(heading[i])
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        rotated = scaled @ R.T
        corners[i] = rotated + np.array([x, y, z])

    return corners  # (N, 8, 3)


def transform(points_ego, T_tf):
    # points_ego: (N, 3)
    N = points_ego.shape[0]
    homog_points = np.hstack((points_ego, np.ones((N, 1))))  # (N, 4)
    points_cam = (T_tf @ homog_points.T).T  # (N, 4)
    return points_cam[:, :3]


def project_cam_to_image(points_cam, intrinsic, image_size=None):
    # Intrinsic: 3x3
    # points_cam: (N, 3)
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    # Filter points in front of the camera
    mask = z > 0
    x, y, z = x[mask], y[mask], z[mask]

    points_norm = np.stack([x / z, y / z, np.ones_like(z)], axis=1).T  # (3, N)
    uv = (intrinsic @ points_norm).T  # (N, 3)

    # Step 3: Optionally filter by image size
    if image_size is not None:
        W, H = image_size
        in_bounds = (
            (uv[:, 0] >= 0) & (uv[:, 0] < W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < H)
        )
        uv = uv[in_bounds]
    return uv[:, :2], mask  # 2D image coordinates, and mask of valid points


def project_cam_to_image_nodrop(P_cam, K, image_size):
    W, H = image_size
    z = P_cam[:, 2]
    eps = 1e-6
    x = P_cam[:, 0] / np.clip(z, eps, None)
    y = P_cam[:, 1] / np.clip(z, eps, None)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    in_front = z > eps
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    mask = in_front & in_bounds

    uv = np.stack([u, v], axis=1)   # shape (M, 2) for all inputs
    return uv, mask                 # NO point dropping



def plot_projection(img, road_uv, agent_uv, cam_name, save_dir, file_prefix):
    """
    Visualize projected road and agent points on a camera image.

    Args:
        img (np.ndarray): The RGB image.
        road_uv (np.ndarray): (N, 2) array of projected road 2D points.
        agent_uv (np.ndarray): (M, 2) array of projected agent 2D points.
        cam_name (str): Name of the camera channel (e.g., 'CAM_FRONT').
        save_dir (str): Directory to save the visualization image.
        file_prefix (str): Filename prefix for saving (e.g., 'scene_001_lg_main').

    Returns:
        None
    """
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    plt.imshow(img)

    if road_uv is not None and len(road_uv) > 0:
        plt.scatter(road_uv[:, 0], road_uv[:, 1], c='cyan', s=1, label='Road')
    if agent_uv is not None and len(agent_uv) > 0:
        plt.scatter(agent_uv[:, 0], agent_uv[:, 1], c='red', s=5, label='Agents')

    plt.legend()
    plt.title(f'Projection on {cam_name}')
    plt.axis('off')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{file_prefix}_{cam_name}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_projection_single_view_with_topdown(
    cam_img, road_uv, agent_uv,
    topdown_img=None, save_dir=None, file_prefix='',
    agent_boxes_uvs=None, agent_boxes_vis=None,
    agent_types=None, figsize=(10,5),
    normalize_extent=False,
):
    import os
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import numpy as np

    img = np.transpose(cam_img, (1, 2, 0))
    h, w = img.shape[:2]

    # ------------------------------------------------------------------
    # CASE 1: NO TOPDOWN IMAGE → SINGLE IMAGE, MATCH FIGURE TO IMAGE SIZE
    # ------------------------------------------------------------------
    if topdown_img is None:

        # Ensure figure matches pixel dimensions
        dpi = 100
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])   # fill entire figure

        # main camera image
        if normalize_extent:
            ax.imshow(img, extent=[0,1,1,0], interpolation="nearest", aspect="equal")
            x_scale, y_scale = 1, 1
        else:
            ax.imshow(img, interpolation="nearest", aspect="equal")
            x_scale, y_scale = w, h

        # --- Road points ---
        if road_uv is not None and len(road_uv) > 0:
            pts = road_uv.astype(float)
            if normalize_extent:
                pts[:, 0] /= w; pts[:, 1] /= h
            ax.scatter(pts[:, 0], pts[:, 1], c="white", s=2)

        # --- Agent centers ---
        if agent_uv is not None and len(agent_uv) > 0:
            pts = agent_uv.astype(float)
            if normalize_extent:
                pts[:, 0] /= w; pts[:, 1] /= h
            ax.scatter(pts[:, 0], pts[:, 1], c="red", s=3)

        # --- 3D Boxes ---
        if agent_boxes_uvs is not None:
            box_edges = [(0,1),(1,2),(2,3),(3,0),
                         (4,5),(5,6),(6,7),(7,4),
                         (0,4),(1,5),(2,6),(3,7)]
            N = agent_boxes_uvs.shape[0]
            for a in range(N):
                colr = agent_color(a, agent_types) if agent_types is not None else 'yellow'
                if agent_types is not None:
                    colr = [c/255.0 for c in colr]
                for e0, e1 in box_edges:
                    if agent_boxes_vis[a, e0] or agent_boxes_vis[a, e1]:
                        x0,y0 = agent_boxes_uvs[a,e0]
                        x1,y1 = agent_boxes_uvs[a,e1]
                        clipped = clip_line(x0,y0,x1,y1,w,h)
                        if clipped is None: continue
                        x0c,y0c,x1c,y1c = clipped
                        if normalize_extent:
                            x0c/=w; y0c/=h; x1c/=w; y1c/=h
                        ax.plot([x0c,x1c], [y0c,y1c], lw=2.0, color=colr)

        ax.set_xlim(0, x_scale)
        ax.set_ylim(y_scale, 0)
        ax.axis("off")

        # save with NO PADDING
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{file_prefix}_single_view.png")
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return

    # ------------------------------------------------------------------
    # CASE 2: TOPDOWN IMAGE PROVIDED → USE ORIGINAL 1×2 GRID
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # ---- Camera panel ----
    ax = fig.add_subplot(gs[0, 0])

    if normalize_extent:
        ax.imshow(img, extent=[0,1,1,0], interpolation='nearest', aspect='equal')
        x_scale, y_scale = 1.0, 1.0
    else:
        ax.imshow(img, interpolation='nearest', aspect='equal')
        x_scale, y_scale = w, h

    ax.set_xlim(0, x_scale)
    ax.set_ylim(y_scale, 0)
    ax.axis('off')

    # ---- Top-down panel ----
    ax_td = fig.add_subplot(gs[0, 1])
    td_h, td_w = topdown_img.shape[:2]
    topdown_img = np.rot90(topdown_img, k=1)

    if normalize_extent:
        ax_td.imshow(topdown_img, extent=[0,1,1,0], interpolation='nearest', aspect='equal')
        ax_td.set_xlim(0, 1); ax_td.set_ylim(1, 0)
    else:
        ax_td.imshow(topdown_img, interpolation='nearest', aspect='equal')
        ax_td.set_xlim(0, td_w); ax_td.set_ylim(td_h, 0)

    ax_td.axis('off')

    # Save 1×2 layout
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_prefix}_single_view.png")

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)

    plt.close(fig)

def plot_projection_all_views(
    cam_imgs, road_uvs, agent_uvs, cam_order,
    topdown_img=None, save_dir=None, file_prefix='',
    agent_boxes_uvs=None, agent_boxes_vis=None,
    agent_types=None, figsize=(17,7),
    normalize_extent=False,  # make all panels identical in size regardless of input img WxH
):
    import os
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # --- fixed grid: always 2x5, last col reserved for top-down (blank if None) ---
    fig = plt.figure(figsize=figsize) #, constrained_layout=True
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1])

    box_edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]

    # 8 camera views occupy [0:2, 0:4]
    for i in range(len(cam_order)):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(gs[row, col])

        img = cam_imgs[i]
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        h, w = img.shape[:2]

        if normalize_extent:
            # normalize each image to unit square so axes are identical for all views
            ax.imshow(img, extent=[0,1,1,0], interpolation='nearest', aspect='equal')
            x_scale, y_scale = 1.0, 1.0
        else:
            ax.imshow(img, interpolation='nearest', aspect='equal')
            x_scale, y_scale = w, h

        # points
        if road_uvs is not None and road_uvs[i] is not None and len(road_uvs[i]) > 0:
            pts = road_uvs[i].astype(float).copy()
            if normalize_extent:
                pts[:, 0] /= w; pts[:, 1] /= h
            ax.scatter(pts[:, 0], pts[:, 1], c='cyan', s=1, label='Road')

        if agent_uvs is not None and agent_uvs[i] is not None and len(agent_uvs[i]) > 0:
            pts = agent_uvs[i].astype(float).copy()
            if normalize_extent:
                pts[:, 0] /= w; pts[:, 1] /= h
            ax.scatter(pts[:, 0], pts[:, 1], c='red', s=3, label='Centers')

        # 3D boxes
        if agent_boxes_uvs is not None and agent_boxes_uvs[i] is not None:
            uv_i = agent_boxes_uvs[i]      # (N, 8, 2)
            vis_i = agent_boxes_vis[i]     # (N, 8) bool
            N = uv_i.shape[0]
            for a in range(N):
                colr = agent_color(a, agent_types) if agent_types is not None else 'yellow'
                if agent_types is not None:
                    colr = [c/255.0 for c in colr]
                for e0, e1 in box_edges:
                    if vis_i[a, e0] or vis_i[a, e1]:
                        x0, y0 = uv_i[a, e0]
                        x1, y1 = uv_i[a, e1]
                        # clip in original pixel coords, then normalize if needed
                        clipped = clip_line(x0, y0, x1, y1, w, h)
                        if clipped is not None:
                            x0c, y0c, x1c, y1c = clipped
                            if normalize_extent:
                                x0c /= w; y0c /= h; x1c /= w; y1c /= h
                            ax.plot([x0c, x1c], [y0c, y1c], linewidth=1.0, alpha=0.9, color=colr)

        ax.set_title(cam_order[i])
        ax.set_xlim(0, x_scale)
        ax.set_ylim(y_scale, 0)
        ax.set_aspect('equal')
        ax.axis('off')

    # Top-down column (always present, blank if None)
    ax_td = fig.add_subplot(gs[:, 4])
    if topdown_img is not None:
        td_h, td_w = topdown_img.shape[:2]
        if normalize_extent:
            ax_td.imshow(topdown_img, extent=[0,1,1,0], interpolation='nearest', aspect='equal')
            ax_td.set_xlim(0, 1); ax_td.set_ylim(1, 0)
        else:
            ax_td.imshow(topdown_img, interpolation='nearest', aspect='equal')
            ax_td.set_xlim(0, td_w); ax_td.set_ylim(td_h, 0)
        ax_td.set_title("Top-down view")
    else:
        ax_td.set_title("Top-down view (n/a)")
    ax_td.axis('off')

    # Save with fixed canvas; avoid tight cropping that changes panel pixel sizes
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{file_prefix}_multi_view_projection.png')
    plt.subplots_adjust(wspace=0.02, hspace=0.02,
                    left=0.01, right=0.99,
                    top=0.95, bottom=0.01)
    fig.savefig(save_path, dpi=300)  # no bbox_inches='tight', no pad_inches
    plt.close(fig)



def clip_line(x0, y0, x1, y1, width, height):
    """Clip a line segment to the rectangle [0,width-1]x[0,height-1].
    Returns (x0c, y0c, x1c, y1c) or None if fully outside.
    """
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0, width-1 - x0, y0, height-1 - y0]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None  # parallel and outside
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
    x0c, y0c = x0 + u1*dx, y0 + u1*dy
    x1c, y1c = x0 + u2*dx, y0 + u2*dy
    return x0c, y0c, x1c, y1c


def radians_to_degrees(radians):
    return radians * (180.0 / math.pi)

def plot_topdown_lanes_and_agents(
    road_points,
    lane_types,
    edge_index_lane_to_lane,
    road_connection_types,
    agent_states,
    agent_types,
    save_path=None,
):
    colors = ['black', 'silver', 'lightcoral', 'firebrick', 'red', 'coral', 'sienna', 'darkorange',
              'gold', 'darkkhaki', 'olive', 'yellow', 'yellowgreen', 'chartreuse', 'forestgreen',
              'turquoise', 'lightcyan', 'teal', 'aqua', 'deepskyblue', 'royalblue', 'navy',
              'mediumpurple', 'indigo', 'violet', 'darkviolet', 'magenta', 'deeppink', 'pink']

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(len(road_points)):
        lane = road_points[i, :, :2]
        if len(lane) == 0:
            continue

        # Color by lane type
        if lane_types[i, 0] == 1:
            color = 'black'
        elif lane_types[i, 1] == 1:
            color = 'green'
        else:
            color = 'red'

        ax.plot(lane[:, 0], lane[:, 1], color=color, linewidth=1.5)

        label_idx = len(lane) // 2
        ax.annotate(i, (lane[label_idx, 0], lane[label_idx, 1]), zorder=5, fontsize=5)

    # Draw lane connections
    # for j in range(edge_index_lane_to_lane.shape[1]):
    #     conn_type = road_connection_types[j]
    #     src_idx = edge_index_lane_to_lane[0, j]
    #     dest_idx = edge_index_lane_to_lane[1, j]
    #     lane_src = road_points[src_idx, :, :2]
    #     lane_dest = road_points[dest_idx, :, :2]

    #     src_pos = lane_src[10, :2]
    #     dest_pos = lane_dest[10, :2]

    #     edge_color = 'black'
    #     if conn_type[1] == 1:
    #         edge_color = 'blue'
    #     elif conn_type[2] == 1:
    #         edge_color = 'purple'
    #     elif conn_type[3] == 1:
    #         edge_color = 'pink'

    #     ax.arrow(src_pos[0], src_pos[1], dest_pos[0] - src_pos[0], dest_pos[1] - src_pos[1],
    #              length_includes_head=True, head_width=1, head_length=1, zorder=10, color=edge_color)

    # Draw agents
    ax.scatter(agent_states[:, 0], agent_states[:, 1], s=10, color='black')

    x_max = 32
    x_min = -32
    y_max = 32
    y_min = -32
    alpha = 0.25
    edgecolor = 'black'

    for a in range(len(agent_states)):
        if a == 0:
            color = 'blue'
        elif agent_types[a, 0]:
            color = '#ffde8b'
        elif agent_types[a, 1]:
            color = 'purple'
        elif agent_types[a, 2]:
            color = 'green'
        else:
            color = 'pink'

        # Draw bounding boxes
        length = agent_states[a, 5]
        width = agent_states[a, 6]
        bbox_x_min = agent_states[a, 0] - width / 2
        bbox_y_min = agent_states[a, 1] - length / 2
        lw = 0.35 / ((x_max - x_min) / 140)
        rectangle = mpatches.FancyBboxPatch(
            (bbox_x_min, bbox_y_min), width, length,
            ec=edgecolor, fc=color, linewidth=lw, alpha=alpha,
            boxstyle=mpatches.BoxStyle("Round", pad=0.3)
        )

        cos_theta = agent_states[a, 3]
        sin_theta = agent_states[a, 4]
        theta = np.arctan2(sin_theta, cos_theta)
        tr = transforms.Affine2D().rotate_deg_around(agent_states[a, 0], agent_states[a, 1], radians_to_degrees(theta) - 90) + ax.transData
        rectangle.set_transform(tr)
        ax.set_aspect('equal', adjustable='box')
        ax.add_patch(rectangle)

        # Heading line
        heading_length = length / 2 + 1.5
        line_end_x = agent_states[a, 0] + heading_length * math.cos(theta)
        line_end_y = agent_states[a, 1] + heading_length * math.sin(theta)
        ax.plot([agent_states[a, 0], line_end_x], [agent_states[a, 1], line_end_y],
                color='black', alpha=0.25, linewidth=0.25 / ((x_max - x_min) / 140))

    # Final setup
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('equal')
    ax.axis('off')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close(fig)
    else:
        # Save figure to memory (not disk)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free memory
        buf.seek(0)

        # Read image from buffer and convert to RGB NumPy array
        image = Image.open(buf).convert('RGB')
        return np.array(image)





def plot_cameras_in_ego(
    T_cam_tfs,
    cam_names=None,
    axis_length=0.5,
    ego_dim=None,
    save_dir=None,
    file_prefix=None,
):
    """
    Plot camera positions and orientations relative to ego vehicle frame.

    Args:
        T_cam_tfs (list[np.ndarray] or np.ndarray): List of 4x4 matrices [N] or stacked [N,4,4].
        cam_names (list of str, optional): Names of cameras for labeling. If None, uses indices.
        axis_length (float): Length of the orientation axes to draw.
        ego_dim (np.ndarray, optional): [length, width, height] of ego vehicle (drawn as a box).
        save_dir (str, optional): If provided, saves the plot to this path.
        show (bool): Whether to display the plot interactively.
    """
    # Ensure array of shape [N,4,4]
    T_cam_tfs = np.array(T_cam_tfs)
    assert T_cam_tfs.ndim == 3 and T_cam_tfs.shape[1:] == (4, 4), \
        f"Expected [N,4,4], got {T_cam_tfs.shape}"

    N = T_cam_tfs.shape[0]
    if cam_names is None:
        cam_names = [f"Cam{i}" for i in range(N)]

    fig = plt.figure(figsize=(14, 7))

    # ========== 3D VIEW ==========
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.scatter(0, 0, 0, c='k', marker='o', label="Ego origin")

    # Draw ego vehicle (3D box)
    if ego_dim is not None:
        L, W, H = ego_dim
        x = np.array([ L/2,  L/2, -L/2, -L/2,  L/2,  L/2, -L/2, -L/2])
        y = np.array([ W/2, -W/2, -W/2,  W/2,  W/2, -W/2, -W/2,  W/2])
        z = np.array([ 0,    0,    0,    0,    H,    H,    H,    H   ])
        edges = [[0,1],[1,2],[2,3],[3,0],
                 [4,5],[5,6],[6,7],[7,4],
                 [0,4],[1,5],[2,6],[3,7]]
        for e in edges:
            ax3d.plot([x[e[0]], x[e[1]]],
                      [y[e[0]], y[e[1]]],
                      [z[e[0]], z[e[1]]], 'k-')

    # Plot cameras in 3D
    for i in range(N):
        T = T_cam_tfs[i]
        t = T[:3, 3]
        R = T[:3, :3]

        ax3d.scatter(t[0], t[1], t[2], marker='^', s=50)
        x_axis, y_axis, z_axis = R[:,0]*axis_length, R[:,1]*axis_length, R[:,2]*axis_length
        ax3d.quiver(t[0], t[1], t[2], *x_axis, color='r')
        ax3d.quiver(t[0], t[1], t[2], *y_axis, color='g')
        ax3d.quiver(t[0], t[1], t[2], *z_axis, color='b')
        ax3d.text(t[0], t[1], t[2], cam_names[i], fontsize=9)

    ax3d.set_xlabel("X (forward)")
    ax3d.set_ylabel("Y (left)")
    ax3d.set_zlabel("Z (up)")
    ax3d.set_title("3D Camera Poses")

    # ========== TOP-DOWN VIEW ==========
    ax2d = fig.add_subplot(122)
    ax2d.scatter(0, 0, c='k', marker='o', label="Ego origin")

    # Draw ego rectangle (2D footprint)
    if ego_dim is not None:
        L, W, _ = ego_dim
        rect_x = [ L/2,  L/2, -L/2, -L/2,  L/2]
        rect_y = [ W/2, -W/2, -W/2,  W/2,  W/2]
        ax2d.plot(rect_x, rect_y, 'k-')

    # Plot cameras in 2D (XY plane)
    # Plot cameras in 2D (XY plane)
    for i in range(N):
        T = T_cam_tfs[i]
        t = T[:3, 3]
        R = T[:3, :3]

        ax2d.scatter(t[0], t[1], marker='^', s=50)

        # Camera up (z-axis) projected to XY
        up = R[:, 2] * axis_length
        ax2d.arrow(t[0], t[1], up[0], up[1],
                   head_width=0.1, head_length=0.1, fc='b', ec='b')

        ax2d.text(t[0], t[1], cam_names[i], fontsize=9)

    ax2d.set_xlabel("X (forward)")
    ax2d.set_ylabel("Y (left)")
    ax2d.axis("equal")
    ax2d.set_title("Top-Down Camera Poses")

    fig.suptitle("Camera Poses Relative to Ego Vehicle Frame", fontsize=14)

    # Save or show
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{file_prefix}_cam_calib.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def transform_box_upright(
    x_c, y_c, z_c,          # centre in source frame
    l, w, h,                # extents
    yaw_deg_src,            # yaw about +Z in source frame  (degrees)
    R_src_to_tgt,           # 3×3 rotation matrix (pose)
    t_src_to_tgt            # 3‑vector translation  (pose)
):
    """
    Rigidly transform an upright 3‑D box from *src* to *tgt* frame.

    Parameters
    ----------
    R_src_to_tgt : (3,3) ndarray
        Rotation that takes vectors **from source to target frame**.
    t_src_to_tgt : (3,)  ndarray
        Translation of *source‐origin* expressed in target frame; the same
        `position` you add to the rotated corners.
    Returns
    -------
    (x_t, y_t, z_t, l, w, h, yaw_deg_tgt)
    """
    # 1) centre
    centre_src = np.array([x_c, y_c, z_c])
    centre_tgt = R_src_to_tgt @ centre_src + t_src_to_tgt

    # 2) yaw  → simply add the frame‑to‑frame yaw component
    # extract pose yaw (rotation about +Z) from R
    yaw_pose = np.arctan2(R_src_to_tgt[1, 0],
                                     R_src_to_tgt[0, 0])
    yaw_tgt = yaw_deg_src + yaw_pose

    # normalise yaw to (−180, 180]
    yaw_tgt = (yaw_tgt + np.pi) % (2*np.pi) - np.pi

    out = dict(x=float(centre_tgt[0]),
            y=float(centre_tgt[1]),
            z=float(centre_tgt[2]),
            l=float(l), w=float(w), h=float(h),
            yaw=yaw_tgt)
    return out



def transform_heading(
    agent_heading,          # heading in ego frame
    ego_heading,            # heading of ego vehicle
):

    yaw_tgt = agent_heading + ego_heading
    # normalise yaw to (−180, 180]
    yaw_tgt = (yaw_tgt + np.pi) % (2*np.pi) - np.pi

    return yaw_tgt