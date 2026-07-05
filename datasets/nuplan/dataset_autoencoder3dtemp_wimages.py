import os
import glob
import hydra
import torch
import pickle
import random
import copy
import gzip
from tqdm import tqdm
from typing import Any, Dict, Optional
import warnings

from torch_geometric.data import Dataset
torch.set_printoptions(threshold=100000)
import numpy as np
import sys
np.set_printoptions(suppress=True, threshold=sys.maxsize)
from cfgs.config import CONFIG_PATH, NUPLAN_VEHICLE, NUPLAN_PEDESTRIAN, NUPLAN_STATIC_OBJECT, PARTITIONED

from utils.data_container import ScenarioDreamerData
from utils.lane_graph_helpers import resample_polyline, adjacency_matrix_to_adjacency_list
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from utils.torch_helpers import from_numpy
from utils.data_helpers import get_lane_connection_type_onehot_nuplan, get_object_type_onehot_nuplan, get_lane_type_onehot_nuplan, modify_agent_states, normalize_scene_3d, randomize_indices
from utils.cam_img_utils import trans_matrix, transform, project_cam_to_image, project_cam_to_image_nodrop, plot_projection_all_views, plot_topdown_lanes_and_agents, load_cam_views, get_3d_box_corners, transform_heading


class NuplanDatasetAutoEncoder3DTemp(Dataset):
    """A Torch-Geometric ``Dataset`` wrapping NuPlan scenes for auto-encoding.

    The dataset performs processing of the extracted
    NuPlan Dataset pickles (obtained from a separate SLEDGE data extraction script), including agent / lane-graph extraction,
    and partitioning for in-painting. If preprocess=True, loads directly from preprocessed files
    for efficient autoencoder training. If preprocess=False, saves preprocessed data to disk.
    """
    def __init__(self, cfg: Any, split_name: str = "train", mode: str = "train", lg_type: Optional[int] = None, map_id: Optional[int] = None) -> None:
        """Instantiate a :class:`NuplanDatasetAutoEncoder`.

        Parameters
        ----------
        cfg
            Hydra configuration object containing dataset configs (cfg.dataset in global config)
        split_name
            One of ``{"train", "val", "test"}`` selecting which split
            to load from ``cfg.dataset.dataset_path``.
        mode
            "train" or "eval" - affects shuffling/randomisation inside
            :meth:`get_data`.
        """
        super(NuplanDatasetAutoEncoder3DTemp, self).__init__()
        self.cfg = cfg
        self.data_root_raw = self.cfg.sledge_raw_dataset_path
        self.data_root_map_id = self.cfg.map_id_dataset_path
        self.split_name = split_name
        self.mode = mode
        self.lg_type = lg_type
        self.map_id = map_id
        self.load_images = cfg.load_images
        self.preprocess = self.cfg.preprocess
        self.preprocessed_dir = os.path.join(self.cfg.preprocess_dir, f"{self.split_name}")
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir, exist_ok=True)

        if not self.preprocess:
            self.files = sorted(self._collect_frame_paths_from_sequences())
        else:
            # For preprocessed data, still look for individual files
            self.files = sorted(self._collect_filtered_frame_paths())

        self.image_root = f"{self.cfg.nuplan_data_root}/sensor_blobs"  # this should be the base path before filename_jpg
        self.cam_order = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']

        self.dset_len = len(self.files)

        # Build map_id index for efficient iteration
        self.map_id_index = {}
        if self.preprocess:
            self._build_map_id_index()

        print(f"[NuplanDatasetAutoEncoder3DTemp] Using preprocessed data from: {self.preprocessed_dir}")

    def _collect_frame_paths_from_sequences(self):
        """Collect frame file paths from temporal sequence directories."""
        frame_paths = []
        split_path = os.path.join(self.data_root_raw, self.split_name)

        if not os.path.exists(split_path):
            return frame_paths

        for sequence_id in os.listdir(split_path):
            sequence_path = os.path.join(split_path, sequence_id)
            if os.path.isdir(sequence_path):
                frame_files = sorted(glob.glob(os.path.join(sequence_path, "sledge_raw_frame_*.gz")))
                # if self.load_images:
                #     frame_files = frame_files[::4]  # subsample
                frame_paths.extend(frame_files)

        return frame_paths

    def _collect_filtered_frame_paths(self):
        all_files = glob.glob(self.preprocessed_dir + "/*.pkl")
        if self.lg_type is None and self.map_id is None:
            return all_files

        filtered_files = []
        for path in all_files:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if self.lg_type is not None and data['lg_type'] != self.lg_type:
                continue
            if self.map_id is not None and data['map_id'] != self.map_id:
                continue
            filtered_files.append(path)

        return filtered_files

    def _build_map_id_index(self):
        """Build an index mapping map_id to list of file paths for efficient iteration."""
        if not self.preprocess:
            return

        for path in self.files:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            map_id = data.get('map_id')
            if map_id is not None:
                # Convert to Python int if it's a numpy array or other type
                if isinstance(map_id, np.ndarray):
                    map_id = int(map_id.item())
                elif not isinstance(map_id, int):
                    map_id = int(map_id)

                if map_id not in self.map_id_index:
                    self.map_id_index[map_id] = []
                self.map_id_index[map_id].append(path)

    def get_all_map_ids(self):
        """Return sorted list of all unique map IDs in the dataset."""
        return sorted(list(self.map_id_index.keys()))

    def _load_camera_image(self, data, camera_name: str = 'CAM_F0'):
        """Load camera image from disk given the data sample.

        Args:
            data: Data dictionary containing cam_info with filename_jpg
            camera_name: Name of the camera (default: CAM_F0)

        Returns:
            numpy array of shape (H, W, 3) in RGB format, or None if image not found
        """
        import cv2

        if 'cam_info' not in data or camera_name not in data['cam_info']:
            return None

        cam_info = data['cam_info'][camera_name]
        filename_jpg = cam_info.get('filename_jpg')

        if filename_jpg is None:
            return None

        image_path = os.path.join(self.image_root, filename_jpg)

        if not os.path.exists(image_path):
            warnings.warn(f"Camera image not found: {image_path}")
            return None

        img = cv2.imread(image_path)
        if img is None:
            warnings.warn(f"Failed to read image: {image_path}")
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_lane_graph_within_fov(self, lane_graph: Dict[str, Any], front_only: bool = False, front_preprocess: bool = False) -> Dict[str, Any]:
        """Return only those lanes that intersect the square *field-of-view*.

        The coordinate frame is converted to an ego-centred frame
        earlier in the pipeline, so the autonomous vehicle (AV) is at
        the origin.  A lane point is considered *in view* when both its
        absolute X *and* Y coordinates are strictly smaller than
        ``cfg_dataset.fov / 2``.  Each retained lane is then resampled
        to a fixed number of points.

        Parameters
        ----------
        lane_graph : Dict[str, Any]
            A *compact* or *partitioned* lane-graph with the standard
            keys ``{"lanes", "lane_types", "pre_pairs", "suc_pairs"}``.
            All coordinates must already be expressed
            in the AV-centric frame.

        Returns
        -------
        lane_graph_within_fov: Dict[str, Any]
            A new lane-graph containing only lanes that intersect the
            configured field-of-view.  Connection dictionaries are
            pruned so they reference *in-FOV* lanes exclusively, and each
            lane polyline has exactly
            ``cfg_dataset.upsample_lane_num_points`` points.
        """
        lane_ids = lane_graph['lanes'].keys()
        pre_pairs = lane_graph['pre_pairs']
        suc_pairs = lane_graph['suc_pairs']

        # ── Identify lanes that intersect the square FOV ──────────────
        lane_ids_within_fov = []
        valid_pts = {}
        for lane_id in lane_ids:
            lane = lane_graph['lanes'][lane_id]
            if front_only:
                points_in_fov_x =  (lane[:, 0] < self.cfg.fov) & (lane[:, 0] >= 0)
            elif front_preprocess:
                points_in_fov_x =  np.abs(lane[:, 0]) < self.cfg.fov

            else:
                points_in_fov_x = np.abs(lane[:, 0]) < (self.cfg.fov / 2)

            points_in_fov_y = np.abs(lane[:, 1]) < (self.cfg.fov / 2)
            points_in_fov = points_in_fov_x * points_in_fov_y

            if np.any(points_in_fov):
                lane_ids_within_fov.append(lane_id)
                valid_pts[lane_id] = points_in_fov

        lanes_within_fov = {}
        lane_types_within_fov = {}
        pre_pairs_within_fov = {}
        suc_pairs_within_fov = {}

        # ── Prune connection dictionaries and resample polylines ─────────────────────────────
        for lane_id in lane_ids_within_fov:
            if lane_id in lane_ids:
                lane = lane_graph['lanes'][lane_id][valid_pts[lane_id]]
                resampled_lane = resample_polyline(lane, num_points=self.cfg.upsample_lane_num_points)
                lanes_within_fov[lane_id] = resampled_lane
                lane_types_within_fov[lane_id] = lane_graph['lane_types'][lane_id]

            if lane_id in pre_pairs:
                pre_pairs_within_fov[lane_id] = [l for l in pre_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                pre_pairs_within_fov[lane_id] = []

            if lane_id in suc_pairs:
                suc_pairs_within_fov[lane_id] = [l for l in suc_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                suc_pairs_within_fov[lane_id] = []

        lane_graph_within_fov = {
            'lanes': lanes_within_fov,
            'lane_types': lane_types_within_fov,
            'pre_pairs': pre_pairs_within_fov,
            'suc_pairs': suc_pairs_within_fov
        }

        return lane_graph_within_fov


    def partition_compact_lane_graph(self, compact_lane_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Split lanes that cross the scene's y-axis (``x = 0``).
        NOTE: Waymo splits on ``x = 0``, but NuPlan splits on ``y = 0``. This is to stay consistent
        with how SLEDGE scenes are oriented.

        The coordinate frame places the ego at ``(0, 0)``.
        To simplify conditional generation (in-painting), we partition
        any merged *compact* lane that crosses ``y = 0`` into multiple
        *sub-lanes* so that the origin acts as a semantic divider.

        Parameters
        ----------
        compact_lane_graph
            The *compact* lane graph returned by
            :meth:`extract_lane_graph`.

        Returns
        -------
        partitioned_lane_graph
            A deep-copy of *compact_lane_graph* where lanes have been
            further split and edge dictionaries updated so that no lane
            segment itself crosses ``y = 0``.
        """
        max_lane_id = max(list(compact_lane_graph['lanes'].keys()))
        next_lane_id = max_lane_id + 1

        lane_ids = list(compact_lane_graph['lanes'].keys())
        for lane_id in lane_ids:
            lane = compact_lane_graph['lanes'][lane_id]

            # Get x-values of the lane and find where it crosses or is near x = 0
            x_values = lane[:, 0]  # Assuming lane is [x, y] points
            sign_diff = np.insert(np.diff(np.signbit(x_values)), 0, 0)
            zero_crossings = np.where(sign_diff)[0]  # Indices where lane crosses x = 0

            if len(zero_crossings) == 0:  # If no crossings, skip this lane
                continue

            # Add artificial partitions at x = 0 crossings
            new_lanes = {}
            start_index = 0
            for crossing in zero_crossings:
                end_index = crossing + 1  # Create a partition from start to crossing
                new_lanes[next_lane_id] = lane[start_index:end_index]
                start_index = crossing  # Update start index for the next partition
                next_lane_id += 1

            # Handle the remaining part of the lane after the last crossing
            if zero_crossings[-1] < len(x_values) - 1:
                new_lanes[next_lane_id] = lane[start_index:]
                next_lane_id += 1

            # Update the compact_lane_graph with new lanes
            num_new_lanes = len(new_lanes)
            if num_new_lanes == 1:
                continue

            for j, new_lane_id in enumerate(new_lanes.keys()):
                compact_lane_graph['lanes'][new_lane_id] = new_lanes[new_lane_id]
                compact_lane_graph['lane_types'][new_lane_id] = compact_lane_graph['lane_types'][lane_id]
                if j == 0:
                    compact_lane_graph['pre_pairs'][new_lane_id] = compact_lane_graph['pre_pairs'][lane_id]
                    # leveraging bijection between suc/pre
                    # replace successors of other lanes with new lane
                    for other_lane_id in compact_lane_graph['pre_pairs'][lane_id]:
                        if other_lane_id is not None:
                            compact_lane_graph['suc_pairs'][other_lane_id].remove(lane_id)
                            compact_lane_graph['suc_pairs'][other_lane_id].append(new_lane_id)
                    compact_lane_graph['suc_pairs'][new_lane_id] = [new_lane_id + 1] # by way we defined new lane ids

                elif j == num_new_lanes - 1:
                    compact_lane_graph['suc_pairs'][new_lane_id] = compact_lane_graph['suc_pairs'][lane_id]
                    # leveraging bijection between suc/pre
                    # replace predecessors of other lanes with new lane
                    for other_lane_id in compact_lane_graph['suc_pairs'][lane_id]:
                        if other_lane_id is not None:
                            compact_lane_graph['pre_pairs'][other_lane_id].remove(lane_id)
                            compact_lane_graph['pre_pairs'][other_lane_id].append(new_lane_id)
                    compact_lane_graph['pre_pairs'][new_lane_id] = [new_lane_id - 1] # by way we define new lane ids

                else:
                    compact_lane_graph['pre_pairs'][new_lane_id] = [new_lane_id - 1]
                    compact_lane_graph['suc_pairs'][new_lane_id] = [new_lane_id + 1]

            # remove old (now partitioned) lane from lane graph
            del compact_lane_graph['lanes'][lane_id]
            del compact_lane_graph['pre_pairs'][lane_id]
            del compact_lane_graph['suc_pairs'][lane_id]

        return compact_lane_graph


    def extract_lane_graph(
            self,
            G,
            lines,
            green_lights,
            red_lights,
            map_id):
        """ Extracts lane graph from SLEDGE cache data format. Outputs similar format to Waymo
        but additional lane_type attribute distinguishes lane/green light/red light."""

        lanes = {}
        lane_types = {}
        ct = 0

        lane_graph_adj = G['states']

        # remove lanes with only one point
        indices_to_remove = []
        for i, (line_states, line_mask) in enumerate(zip(lines['states'], lines['mask'])):
            line_in_mask = line_states[line_mask, :]  # (n, 3)
            if len(line_in_mask) < 2:
                indices_to_remove.append(i)
                continue

            lanes[ct] = line_in_mask
            lane_types[ct] = get_lane_type_onehot_nuplan("lane")

            ct += 1

        if len(indices_to_remove) > 0:
            lane_graph_adj = np.delete(lane_graph_adj, indices_to_remove, axis=0)
            lane_graph_adj = np.delete(lane_graph_adj, indices_to_remove, axis=1)

        # add green lights to lanes
        for green_light_states, green_light_mask in zip(green_lights['states'], green_lights['mask']):
            green_light = green_light_states[green_light_mask, :]
            if len(green_light) < 2:
                continue

            lanes[ct] = green_light
            lane_types[ct] = get_lane_type_onehot_nuplan("green_light")

            ct += 1

        # add red lights to lanes
        for red_light_states, red_light_mask in zip(red_lights['states'], red_lights['mask']):
            red_light = red_light_states[red_light_mask, :]
            if len(red_light) < 2:
                continue

            lanes[ct] = red_light
            lane_types[ct] = get_lane_type_onehot_nuplan("red_light")

            ct += 1

        # adjacency list only defined over lanes, not red/green lights
        pre_pairs, suc_pairs = adjacency_matrix_to_adjacency_list(lane_graph_adj)

        lane_graph = {
            'lanes': lanes,
            'lane_types': lane_types,
            'pre_pairs': pre_pairs,
            'suc_pairs': suc_pairs,
            'map_id': map_id
        }

        return lane_graph


    def extract_agents(self, ego, vehicles, pedestrians, static_objects):
        """ Extracts agent features from SLEDGE cache data format.
            Output format is the same as the Waymo dataset, but instead of modeling
            vehicle/pedestrian/bicycle we model vehicle/pedestrian/static_object."""
        processed_agent_states = []
        agent_types = []
        ground_heights = []
        """
        `ego` indices:
        0: vel_x
        1: vel_y
        2: accel_x
        3: accel_y
        """
        ego_states = ego['states']
        ego_x = 0.
        ego_y = 0.
        ego_z = 0.
        ego_vel_x = ego_states[0]
        ego_vel_y = ego_states[1]
        ego_heading = 0.
        ego_length = self.cfg.ego_length
        ego_width = self.cfg.ego_width

        ego_og_x = ego_states[4]
        ego_og_y = ego_states[5]
        ego_og_z = ego_states[7]
        ego_og_heading = ego_states[6]
        ego_height = ego_states[8]
        ego_rotation = np.array([ego_states[9], ego_states[10], ego_states[11], ego_states[12]])
        ego_translation = np.array([ego_og_x, ego_og_y, ego_og_z])
        ego_dim = np.array([ego_length, ego_width, ego_height])
        ego_state_og = [ego_translation, ego_rotation, ego_dim, ego_og_heading]

        ego_state = np.array([ego_x, ego_y, ego_vel_x, ego_vel_y, ego_heading, ego_length, ego_width, ego_z, ego_height])
        processed_agent_states.append(ego_state)
        agent_types.append(get_object_type_onehot_nuplan("vehicle"))

        vehicle_states = vehicles['states']
        vehicle_mask = ~vehicles['mask']
        vehicle_states = vehicle_states[vehicle_mask]

        """
        `vehicles`, `pedestrians`, and `static_objects` indices:
        0: x
        1: y
        2: heading
        3: width
        4: length
        5: velocity (speed)
        """
        for v in range(len(vehicle_states)):
            x = vehicle_states[v, 0]
            y = vehicle_states[v, 1]
            z = vehicle_states[v, 6]
            heading = vehicle_states[v, 2]
            speed = vehicle_states[v, 5]
            vel_x = speed * np.cos(heading)
            vel_y = speed * np.sin(heading)
            length = vehicle_states[v, 4]
            width = vehicle_states[v, 3]
            height = vehicle_states[v, 7]
            veh_state = np.array([x, y, vel_x, vel_y, heading, length, width, z, height])
            processed_agent_states.append(veh_state)
            agent_types.append(get_object_type_onehot_nuplan("vehicle"))
            ground_heights.append(z+height/2)

        pedestrian_states = pedestrians['states']
        pedestrian_mask = ~pedestrians['mask']
        pedestrian_states = pedestrian_states[pedestrian_mask]

        for v in range(len(pedestrian_states)):
            x = pedestrian_states[v, 0]
            y = pedestrian_states[v, 1]
            z = pedestrian_states[v, 6]
            heading = pedestrian_states[v, 2]
            speed = pedestrian_states[v, 5]
            vel_x = speed * np.cos(heading)
            vel_y = speed * np.sin(heading)
            length = pedestrian_states[v, 4]
            width = pedestrian_states[v, 3]
            height = pedestrian_states[v, 7]
            veh_state = np.array([x, y, vel_x, vel_y, heading, length, width, z, height])
            processed_agent_states.append(veh_state)
            agent_types.append(get_object_type_onehot_nuplan("pedestrian"))
            ground_heights.append(z+height/2)

        static_object_states = static_objects['states']
        static_object_mask = ~static_objects['mask']
        static_object_states = static_object_states[static_object_mask]

        for v in range(len(static_object_states)):
            x = static_object_states[v, 0]
            y = static_object_states[v, 1]
            z = static_object_states[v, 5]
            heading = static_object_states[v, 2]
            vel_x = 0.
            vel_y = 0.
            length = static_object_states[v, 4]
            width = static_object_states[v, 3]
            height = static_object_states[v, 6]
            veh_state = np.array([x, y, vel_x, vel_y, heading, length, width, z, height])
            processed_agent_states.append(veh_state)
            agent_types.append(get_object_type_onehot_nuplan("static_object"))
            ground_heights.append(z+height/2)

        processed_agent_states = np.array(processed_agent_states)
        agent_types = np.array(agent_types)
        ground_heights = np.array(ground_heights)

        return processed_agent_states, agent_types, ego_state_og, ground_heights


    def get_agents_within_fov(self, agent_states, agent_types, front_only=False):
        """ Filters agents that are within the field of view (fov) and returns the closest agents
        to the origin, up to the specific max number of vehicles, pedestrians, and static objects."""

        # filter agents that are within the field of view (fov)
        agents_in_fov_x = np.abs(agent_states[:, 0]) < (self.cfg.fov / 2) if not front_only else (agent_states[:, 0] < self.cfg.fov) & (agent_states[:, 0] >= 0)
        agents_in_fov_y = np.abs(agent_states[:, 1]) < (self.cfg.fov / 2)
        agents_in_fov_mask = agents_in_fov_x * agents_in_fov_y
        valid_agents = np.where(agents_in_fov_mask > 0)[0]
        valid_vehicles = np.array(list(set(valid_agents).intersection(set(np.where(agent_types[:, NUPLAN_VEHICLE] == 1)[0]))))
        valid_pedestrians = np.array(list(set(valid_agents).intersection(set(np.where(agent_types[:, NUPLAN_PEDESTRIAN] == 1)[0]))))
        valid_static_objects = np.array(list(set(valid_agents).intersection(set(np.where(agent_types[:, NUPLAN_STATIC_OBJECT] == 1)[0]))))

        # find closest agents to the origin that are within the field of view, up to the specific max number
        dist_to_origin = np.linalg.norm(agent_states[:, :2], axis=-1)
        closest_ag_ids = np.argsort(dist_to_origin)
        closest_veh_ids = closest_ag_ids[np.in1d(closest_ag_ids, valid_vehicles)]
        closest_veh_ids = closest_veh_ids[:self.cfg.max_num_vehicles]
        closest_ped_ids = closest_ag_ids[np.in1d(closest_ag_ids, valid_pedestrians)]
        closest_ped_ids = closest_ped_ids[:self.cfg.max_num_pedestrians]
        closest_static_obj_ids = closest_ag_ids[np.in1d(closest_ag_ids, valid_static_objects)]
        closest_static_obj_ids = closest_static_obj_ids[:self.cfg.max_num_static_objects]
        closest_ag_ids = np.concatenate([closest_veh_ids, closest_ped_ids, closest_static_obj_ids], axis=0)

        return agent_states[closest_ag_ids], agent_types[closest_ag_ids]


    def get_road_points_adj(self, compact_lane_graph):
        """This helper converts the *sparse*, dictionary-based lane graph
        representation that comes out of
        :meth:`get_compact_lane_graph` / :meth:`partition_compact_lane_graph`
        into adjacency matrices and resamples lanes to num_points_per_lane points."""

        # ── Step 1: resample every lane to fixed P points ──────────────
        resampled_lanes = []
        lane_types = []
        idx_to_id = {}
        id_to_idx = {}
        i = 0
        for lane_id in compact_lane_graph['lanes']:
            lane = compact_lane_graph['lanes'][lane_id]
            lane_type = compact_lane_graph['lane_types'][lane_id]
            resampled_lane = resample_polyline(lane, num_points=self.cfg.num_points_per_lane)
            resampled_lanes.append(resampled_lane)
            lane_types.append(lane_type)
            idx_to_id[i] = lane_id
            id_to_idx[lane_id] = i

            i += 1

        # ── Step 2: keep the max_num_lanes closest to the origin ───────
        resampled_lanes = np.array(resampled_lanes)
        lane_types = np.array(lane_types)
        num_lanes = min(len(resampled_lanes), self.cfg.max_num_lanes)
        dist_to_origin = np.linalg.norm(resampled_lanes, axis=-1).min(1)
        closest_lane_ids = np.argsort(dist_to_origin)[:num_lanes]
        resampled_lanes = resampled_lanes[closest_lane_ids]
        lane_types = lane_types[closest_lane_ids]

        # mapping from old idx to new index after ordering by distance
        idx_to_new_idx = {}
        new_idx_to_idx = {}
        for i, j in enumerate(closest_lane_ids):
            idx_to_new_idx[j] = i
            new_idx_to_idx[i] = j

        # Pre‑allocate adjacency matrices (no left/right connections in nuplan) --------
        pre_road_adj = np.zeros((num_lanes, num_lanes))
        suc_road_adj = np.zeros((num_lanes, num_lanes))
        for new_idx_i in range(num_lanes):
            for id_j in compact_lane_graph['pre_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    pre_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1

            for id_j in compact_lane_graph['suc_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    suc_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1

        return resampled_lanes, lane_types, pre_road_adj, suc_road_adj, num_lanes


    def get_partitioned_masks(self, agents, lanes, a2a_edge_index, l2l_edge_index, l2a_edge_index):
        """Create boolean masks that *hide* edges crossing the Y-axis partition."""
        a2a_edge_index = a2a_edge_index.numpy()
        l2l_edge_index = l2l_edge_index.numpy()
        l2a_edge_index = l2a_edge_index.numpy()

        num_agents = len(agents)
        num_lanes = len(lanes)

        agents_x = agents[:, 0]
        lanes_x = lanes[:, 9, 0]
        agents_after_origin = np.where(agents_x > 0)[0]
        lanes_after_origin = np.where(lanes_x > 0)[0]

        # sum only equals 1 if two agents on opposite sides of partition
        a2a_mask = np.isin(a2a_edge_index, agents_after_origin).sum(0) != 1
        l2l_mask = np.isin(l2l_edge_index, lanes_after_origin).sum(0) != 1

        lane_l2a_mask = np.isin(l2a_edge_index[0], lanes_after_origin)[None, :]
        agent_l2a_mask = np.isin(l2a_edge_index[1], agents_after_origin)[None, :]
        l2a_mask = np.concatenate([lane_l2a_mask, agent_l2a_mask], axis=0).sum(0) != 1

        return torch.from_numpy(a2a_mask), torch.from_numpy(l2l_mask), torch.from_numpy(l2a_mask), lanes_x <= 0


    def get_data(self, data, idx):
        """Process **one** Nuplan scenario.

        if preprocess=True: read from cached preprocessed pickle and return ScenarioDreamerData object for autoencoder training
        if preprocess=False: cache processed data as pickle file to disk to reduce data processing overhead during autoencoder training."""

        # ───────────────────────────────────────────────────────────────
        # FAST PATH: already pre-processed tensors on disk
        # ───────────────────────────────────────────────────────────────
        if self.preprocess:
            road_points = data['road_points']
            agent_states = data['agent_states']
            edge_index_lane_to_lane = data['edge_index_lane_to_lane']
            edge_index_lane_to_agent = data['edge_index_lane_to_agent']
            edge_index_agent_to_agent = data['edge_index_agent_to_agent']
            road_connection_types = data['road_connection_types']
            num_lanes = data['num_lanes']
            num_agents = data['num_agents']
            agent_types = data['agent_types']
            lane_types = data['lane_types']
            lg_type = data['lg_type']
            map_id = data['map_id']
            cam_infos = {}
            if self.load_images:
                cam_infos = data['cam_info']
                cam_infos = {k: v for k, v in cam_infos.items() if k in 'CAM_F0'}
            ego_state_og = data['ego_state_og']  # [ego_translation, ego_rotation, ego_dim, ego_heading]


        # ───────────────────────────────────────────────────────────────
        # SLOW PATH: raw Nuplan pickle → preprocess and cache to disk
        # ───────────────────────────────────────────────────────────────
        else:

            # cache the processed dict to disk so subsequent runs take the fast path
            raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
            sequence_id = os.path.basename(os.path.dirname(self.files[idx]))

            # elements of scene already normalized to ego by SLEDGE preprocessing and agents off driveable area have already been removed
            compact_lane_graph_scene = self.extract_lane_graph(
                data['G'],
                data['lines'],
                data['green_lights'],
                data['red_lights'],
                data['id'])
            agent_states, agent_types, ego_state_og, ground_heights = self.extract_agents(
                data['ego'],
                data['vehicles'],
                data['pedestrians'],
                data['static_objects'])

            cam_infos = {}
            valid = True
            if self.load_images:
                cam_infos = data['cam_infos']
                if not 'CAM_F0' in cam_infos:
                    valid = False
                    warnings.warn(f"Skipping sequence {sequence_id} for {raw_file_name} due to missing front camera information.")
                    
                if self.cfg.debug_vis:
                    cam_img_stack, T_cam_tf_stack, T_cam_tf_inv_stack, T_cam_ego_inv_stack, intrinsics_stack, widths, heights  = load_cam_views(
                        cam_infos=cam_infos,
                        cam_order=self.cam_order,
                        image_root=self.image_root,
                        do_undistortion=False,
                        load_cam_img=True
                    )

                # ego_dim = ego_state_og[2]
                # if self.cfg.debug_vis:
                #     plot_cameras_in_ego(T_cam_tf_stack, cam_names=self.cam_order, save_dir=self.cfg.debug_vis_dir,
                #         file_prefix='{}_scene_{}'.format(raw_file_name, idx))

            # statistics here
            normalize_statistics = {}

            compact_lane_graph_scene_regular = self.get_lane_graph_within_fov(compact_lane_graph_scene)
            if len(compact_lane_graph_scene_regular['lanes']) == 0:
                d = {
                'normalize_statistics': None,
                'valid_scene': False
                }
                return d

            # partitioned lane graph enables explicit training to inpaint
            compact_lane_graph_inpainting = self.partition_compact_lane_graph(copy.deepcopy(compact_lane_graph_scene_regular))
            agent_states_regular, agent_types_regular = self.get_agents_within_fov(agent_states, agent_types)
            agent_states_regular = modify_agent_states(agent_states_regular)
            num_agents = len(agent_states_regular)

            # first filter to (-fov, fov)
            compact_lane_graph_front_preprocess = self.get_lane_graph_within_fov(compact_lane_graph_scene, front_preprocess=True)
            # partition to get lanes at cross x=0
            compact_lane_graph_front_preprocess_part = self.partition_compact_lane_graph(copy.deepcopy(compact_lane_graph_front_preprocess))
            # then filter to (0, fov)
            compact_lane_graph_scene_front_only = self.get_lane_graph_within_fov(compact_lane_graph_front_preprocess_part, front_only=True)
            agent_states_front_only, agent_types_front_only = self.get_agents_within_fov(agent_states, agent_types, front_only=True)
            agent_states_front_only = modify_agent_states(agent_states_front_only)

            if self.cfg.debug_vis and self.load_images:
                # ego_state_og =  [ego_translation, ego_rotation, ego_dim]
                # Get transformation matrix
                #T_local2global = create_se2_4x4_matrix(ego_state_og[0], ego_state_og[3])
                T_local2global = trans_matrix(ego_state_og[1], ego_state_og[0])

                # Precompute extrinsics for each camera
                T_extrinsics = [
                    T_cam_tf_inv_stack[i] @ T_cam_ego_inv_stack[i] @ T_local2global
                    for i in range(len(self.cam_order))
                ]

                z_agent = agent_states_front_only[:, -2:-1]
                agent_3d = np.concatenate([agent_states_front_only[:, :2], z_agent], axis=1)
                agent_dim = np.concatenate([agent_states_front_only[:, 5:7], agent_states_front_only[:, 8:9]], axis=1)
                agent_heading = np.arctan2(agent_states_front_only[:, 4], agent_states_front_only[:, 3])  # heading in radians

                agent_global = transform(agent_3d, T_local2global)  # shape (num_agents, 3)
                agent_uv_list = []
                for i in range(len(self.cam_order)):
                    agent_ego = transform(agent_global, T_cam_ego_inv_stack[i])
                    agent_cam = transform(agent_ego, T_cam_tf_inv_stack[i])
                    # alternative
                    # agent_cam = transform(agent_3d, T_extrinsics[i])
                    agent_uv, agent_mask = project_cam_to_image(agent_cam, intrinsics_stack[i], (widths[i], heights[i])) #
                    agent_uv_list.append(agent_uv)
                # boxes
                agent_corners_ego = get_3d_box_corners(agent_3d, agent_heading, agent_dim)  # (N, 8, 3)
                corners_global = transform(agent_corners_ego.reshape(-1, 3), T_local2global).reshape(-1, 8, 3)

                ego_heading = ego_state_og[3]
                agent_heading_global = transform_heading(agent_heading, ego_heading)
                corners_global = get_3d_box_corners(agent_global, agent_heading_global, agent_dim)  # (N, 8, 3)
                # alternative
                # corners_3d = get_3d_box_corners(agent_3d, agent_heading, agent_dim)  # (N, 8, 3)
                agent_boxes_uv_list = []   # list length = num cams; each item: (N, 8, 2)
                agent_boxes_vis_list = []  # list length = num cams; each item: (N, 8) boolean mask
                for i in range(len(self.cam_order)):
                    corners_ego = transform(corners_global.reshape(-1, 3), T_cam_ego_inv_stack[i]).reshape(-1, 8, 3)
                    corners_cam = transform(corners_ego.reshape(-1, 3), T_cam_tf_inv_stack[i]).reshape(-1, 8, 3)
                    # alternative
                    #corners_cam = transform(corners_3d.reshape(-1, 3), T_extrinsics[i]).reshape(-1, 8, 3)

                    # Project all corners; mask marks points behind camera / outside image
                    uv, mask = project_cam_to_image_nodrop(
                        corners_cam.reshape(-1, 3),
                        intrinsics_stack[i],
                        (widths[i], heights[i])
                    )
                    uv = uv.reshape(-1, 8, 2)
                    mask = mask.reshape(-1, 8)

                    agent_boxes_uv_list.append(uv)
                    agent_boxes_vis_list.append(mask)

            if num_agents == 0:
                d = {
                'normalize_statistics': None,
                'valid_scene': False
                }
                return d

            # Process *both* regular & partitioned lane graphs
            lg_dict = {
                'regular': compact_lane_graph_scene_regular,
                'partitioned': compact_lane_graph_inpainting, ## uncomment for debugging
                'front_only': compact_lane_graph_scene_front_only
            }
            for lg_type in lg_dict.keys():
                lg = lg_dict[lg_type]

                if lg_type == 'regular':
                    lg_type_num = 0
                    agent_states = agent_states_regular
                    agent_types = agent_types_regular
                elif lg_type == 'partitioned':
                    lg_type_num = 1
                    agent_states = agent_states_regular
                    agent_types = agent_types_regular
                else:
                    lg_type_num = 2
                    num_agents = len(agent_states_front_only)
                    agent_states = agent_states_front_only
                    agent_types = agent_types_front_only

                    if num_agents == 0 or len(lg['lanes']) == 0:
                        d = {
                        'normalize_statistics': None,
                        'valid_scene': False
                        }
                        return d

                road_points, lane_types, pre_road_adj, suc_road_adj, num_lanes = self.get_road_points_adj(lg)

                # get edge information
                edge_index_lane_to_lane = get_edge_index_complete_graph(num_lanes)
                edge_index_agent_to_agent = get_edge_index_complete_graph(num_agents)
                edge_index_lane_to_agent = get_edge_index_bipartite(num_lanes, num_agents)

                road_connection_types = []
                for i in range(edge_index_lane_to_lane.shape[1]):
                    pre_conn_indicator = pre_road_adj[edge_index_lane_to_lane[1, i], edge_index_lane_to_lane[0, i]]
                    suc_conn_indicator = suc_road_adj[edge_index_lane_to_lane[1, i], edge_index_lane_to_lane[0, i]]
                    if edge_index_lane_to_lane[1, i] == edge_index_lane_to_lane[0, i]:
                        road_connection_types.append(get_lane_connection_type_onehot_nuplan('self'))
                    elif pre_conn_indicator:
                        road_connection_types.append(get_lane_connection_type_onehot_nuplan('pred'))
                    elif suc_conn_indicator:
                        road_connection_types.append(get_lane_connection_type_onehot_nuplan('succ'))
                    else:
                        road_connection_types.append(get_lane_connection_type_onehot_nuplan('none'))
                road_connection_types = np.array(road_connection_types)

                if self.cfg.debug_vis:
                    topdown_img = plot_topdown_lanes_and_agents(
                            road_points=road_points,
                            lane_types=lane_types,
                            edge_index_lane_to_lane=edge_index_lane_to_lane,
                            road_connection_types=road_connection_types,
                            agent_states=agent_states,
                            agent_types=agent_types,
                            save_path=os.path.join(self.cfg.debug_vis_dir,
                            f'{idx}_{sequence_id}_{raw_file_name}_{lg_type_num}.png')
                        )

                if self.load_images and self.cfg.debug_vis:
                    ## Get 3d points
                    N_lanes, N_points, _ = road_points.shape
                    height_estimate = 0.0
                    z_road = np.ones((N_lanes, N_points, 1))*height_estimate
                    road_points_3d = np.concatenate([road_points, z_road], axis=-1)  # shape (7, 20, 3)
                    road_points_3d = road_points_3d.reshape(-1, 3)# shape (num_lanes * lane_length, 3)
                    road_global = transform(road_points_3d, T_local2global)
                    # Transform to camera frame
                    road_uv_list = []

                    for i in range(len(self.cam_order)):
                        road_ego = transform(road_global, T_cam_ego_inv_stack[i])  # shape (num_lanes * lane_length, 3)
                        road_cam = transform(road_ego, T_cam_tf_inv_stack[i])
                        # alternative
                        #road_cam = transform(road_points_3d, T_extrinsics[i])
                        road_uv, road_mask = project_cam_to_image(road_cam, intrinsics_stack[i], (widths[i], heights[i]))
                        road_uv_list.append(road_uv)

                    topdown_img = plot_topdown_lanes_and_agents(
                        road_points=road_points,
                        lane_types=lane_types,
                        edge_index_lane_to_lane=edge_index_lane_to_lane,
                        road_connection_types=road_connection_types,
                        agent_states=agent_states,
                        agent_types=agent_types,
                        #save_path=os.path.join(self.cfg.debug_vis_dir,
                        #    f'{raw_file_name}_scene_{idx}_lg_{lg_type_num}.png')
                    )

                    plot_projection_all_views(
                        cam_imgs=cam_img_stack,
                        road_uvs=road_uv_list,
                        agent_uvs=agent_uv_list,
                        cam_order=self.cam_order,
                        topdown_img=topdown_img,
                        save_dir=self.cfg.debug_vis_dir,
                        file_prefix='{}_{}_scene_{}_lg_{}'.format(sequence_id, raw_file_name, idx, lg_type_num),
                        agent_boxes_uvs=agent_boxes_uv_list,
                        agent_boxes_vis=agent_boxes_vis_list,
                        agent_types=agent_types
                    )

                to_pickle = dict()
                to_pickle['idx'] = idx
                to_pickle['lg_type'] = lg_type_num
                to_pickle['num_agents'] = num_agents
                to_pickle['num_lanes'] = num_lanes
                to_pickle['road_points'] = road_points
                to_pickle['lane_types'] = lane_types
                to_pickle['agent_states'] = agent_states
                to_pickle['agent_types'] = agent_types # only vehicle, pedestrian, static_object
                to_pickle['edge_index_lane_to_lane'] = edge_index_lane_to_lane
                to_pickle['edge_index_agent_to_agent'] = edge_index_agent_to_agent
                to_pickle['edge_index_lane_to_agent'] = edge_index_lane_to_agent
                to_pickle['road_connection_types'] = road_connection_types
                to_pickle['map_id'] = data['id']
                to_pickle['cam_info'] = cam_infos
                to_pickle['ego_state_og'] = ego_state_og # [ego_translation, ego_rotation, ego_dim, ego_heading]
                # save preprocessed file
                if valid:
                    with open(os.path.join(self.preprocessed_dir, f'{sequence_id}_{raw_file_name}_{to_pickle["lg_type"]}.pkl'), 'wb') as f:
                        pickle.dump(to_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)

                if lg_type == 'regular':
                    normalize_statistics['num_agents'] = num_agents
                    normalize_statistics['num_lanes'] = num_lanes
                    normalize_statistics['max_speed'] = agent_states[:, 2].max()
                    normalize_statistics['min_length'] = agent_states[:, 5].min()
                    normalize_statistics['max_length'] = agent_states[:, 5].max()
                    normalize_statistics['min_width'] = agent_states[:, 6].min()
                    normalize_statistics['max_width'] = agent_states[:, 6].max()
                    normalize_statistics['min_lane_x'] = road_points[:, 0].min()
                    normalize_statistics['min_lane_y'] = road_points[:, 1].min()
                    normalize_statistics['max_lane_x'] = road_points[:, 0].max()
                    normalize_statistics['max_lane_y'] = road_points[:, 1].max()

            d = {
                'normalize_statistics': normalize_statistics,
                'valid_scene': True
            }

            return d

        mode = "positive" if lg_type == 2 else "centered"
        if self.mode != 'vis':
            agent_states, road_points = normalize_scene_3d(
                agent_states,
                road_points,
                fov=self.cfg.fov,
                min_speed=self.cfg.min_speed,
                max_speed=self.cfg.max_speed,
                min_length=self.cfg.min_length,
                max_length=self.cfg.max_length,
                min_width=self.cfg.min_width,
                max_width=self.cfg.max_width,
                min_lane_x=self.cfg.min_lane_x,
                min_lane_y=self.cfg.min_lane_y,
                max_lane_x=self.cfg.max_lane_x,
                max_lane_y=self.cfg.max_lane_y,
                min_height=self.cfg.min_height,
                max_height=self.cfg.max_height,
                fov_z=self.cfg.fov_z,
                mode=mode)

        # randomize order of indices except for ego (which is always index 0)
        if self.mode == 'train':
            agent_states, agent_types, road_points, lane_types, edge_index_lane_to_lane = randomize_indices(
                agent_states,
                agent_types,
                road_points,
                edge_index_lane_to_lane,
                lane_types)
            edge_index_lane_to_lane = torch.from_numpy(edge_index_lane_to_lane)

        if lg_type == PARTITIONED:
            a2a_mask, l2l_mask, l2a_mask, lane_partition_mask = self.get_partitioned_masks(
                agent_states,
                road_points,
                edge_index_agent_to_agent,
                edge_index_lane_to_lane,
                edge_index_lane_to_agent)

            agents_x = agent_states[:, 0]
            lanes_x = road_points[:, 9, 0]
            num_agents_after_origin = len(np.where(agents_x > 0)[0])
            num_lanes_after_origin = len(np.where(lanes_x > 0)[0])
        else:
            a2a_mask = torch.ones(edge_index_agent_to_agent.shape[1]).bool()
            l2l_mask = torch.ones(edge_index_lane_to_lane.shape[1]).bool()
            l2a_mask = torch.ones(edge_index_lane_to_agent.shape[1]).bool()
            lane_partition_mask = np.zeros(num_lanes).astype(bool)
            num_agents_after_origin = 0
            num_lanes_after_origin = 0

        assert a2a_mask.shape[0] == edge_index_agent_to_agent.shape[1]
        assert l2l_mask.shape[0] == edge_index_lane_to_lane.shape[1]
        assert l2a_mask.shape[0] == edge_index_lane_to_agent.shape[1]
        assert lane_partition_mask.shape[0] == num_lanes

        # --------------------------------------------------------------
        # ️Assemble final PyG heterogeneous graph ------------------
        # --------------------------------------------------------------
        # reformat agent_states
        # from [x, y, vel, cos(heading), sin(heading), length, width, z, height] to
        # [x, y, z, vel, cos(heading), sin(heading), length, width, height]
        agent_states = np.concatenate([
            agent_states[:, :2],  # x, y
            agent_states[:, 7:8],  # z
            agent_states[:, 2:3],  # vel
            agent_states[:, 3:5],  # cos(heading), sin(heading)
            agent_states[:, 5:7],  # length, width
            agent_states[:, -1:]  # height
        ], axis=1)
        d = ScenarioDreamerData()
        d['idx'] = idx
        d['num_lanes'] = num_lanes
        d['num_agents'] = num_agents
        d['lg_type'] = lg_type
        d['map_id'] = int(map_id)
        d['agent'].x = from_numpy(agent_states)
        d['agent'].type = from_numpy(agent_types)
        d['lane'].x = from_numpy(road_points)
        d['lane'].type = from_numpy(lane_types)
        d['lane'].partition_mask = from_numpy(lane_partition_mask)
        d['num_agents_after_origin'] = num_agents_after_origin
        d['num_lanes_after_origin'] = num_lanes_after_origin

        # edge indices required for pyg
        d['lane', 'to', 'lane'].edge_index = edge_index_lane_to_lane
        d['lane', 'to', 'lane'].type = torch.from_numpy(road_connection_types)
        d['agent', 'to', 'agent'].edge_index = edge_index_agent_to_agent
        d['lane', 'to', 'agent'].edge_index = edge_index_lane_to_agent
        d['lane', 'to', 'lane'].encoder_mask = l2l_mask
        d['lane', 'to', 'agent'].encoder_mask = l2a_mask
        d['agent', 'to', 'agent'].encoder_mask = a2a_mask
        d['cam_info'] = cam_infos
        d['ego_state_og'] = ego_state_og

        return d


    def get(self, idx: int):
        # cache processed data to disk as pickle file
        if not self.cfg.preprocess:
            raw_file_path = self.files[idx]
            # Construct map_id file path by replacing sledge_raw with map_id in the base path and pointing to the single map_id.gz file
            map_id_file_path = raw_file_path.replace('/sledge_raw/', '/map_id/').replace(os.path.basename(raw_file_path), 'map_id.gz')
            with gzip.open(raw_file_path, 'rb') as f:
                data = pickle.load(f)
            with gzip.open(map_id_file_path, 'rb') as f:
                map_id_data = pickle.load(f)
            data['id'] = map_id_data['id']

            d = self.get_data(data, idx)

        # return ScenarioDreamerData object for autoencoder training
        else:
            raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
            raw_path = os.path.join(self.preprocessed_dir, f'{raw_file_name}.pkl')
            with open(raw_path, 'rb') as f:
                data = pickle.load(f)
            d = self.get_data(data, idx)

        return d

    def len(self):
        return self.dset_len

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config3dtemp")
def main(cfg):
    cfg.ae.dataset.preprocess = False
    dset = NuplanDatasetAutoEncoder3DTemp(cfg.ae.dataset, split_name='test', mode='eval')
    print(len(dset))
    np.random.seed(10)
    random.seed(10)
    torch.manual_seed(10)

    for idx in tqdm(range(len(dset))):
        d = dset.get(idx)

        if idx == 100:
            break


if __name__ == '__main__':
    main()