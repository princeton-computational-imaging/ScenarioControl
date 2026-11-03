import os
import pickle
import copy
import numpy as np


def update_camera_extrinsics(cam_infos, ego_state_og, initial_ego_state_og, initial_cam_infos):
    """ Move each camera's global ego_pose rigidly with the ego.

    cam_info['ego_pose'] sits ~1.5 m ahead of ego_state_og (camera mount vs vehicle centre),
    so we apply the ego's motion since t=0 to the initial camera pose instead of overwriting it.

    Args:
        cam_infos: {cam_name: {..., 'ego_pose': [x, y, z, qw, qx, qy, qz]}} (global frame)
        ego_state_og: current global ego pose [translation (3,), quat (4,), dims (3,), heading]
        initial_ego_state_og: ego_state_og at t=0
        initial_cam_infos: cam_infos at t=0
    Returns:
        copy of cam_infos with updated 'ego_pose' per camera
    """
    updated_cam_infos = copy.deepcopy(cam_infos)

    ego_translation = np.array(ego_state_og[0]).reshape(-1)[:3]
    # headings may be stored as shape-(1,) arrays
    current_heading = float(np.asarray(ego_state_og[3]).reshape(-1)[0])
    initial_ego_translation = np.array(initial_ego_state_og[0]).reshape(-1)[:3]
    initial_heading = float(np.asarray(initial_ego_state_og[3]).reshape(-1)[0])
    delta_heading = current_heading - initial_heading
    cos_dh = np.cos(delta_heading)
    sin_dh = np.sin(delta_heading)
    # yaw-only rotation quaternion [qw, 0, 0, qz]
    delta_qw = np.cos(delta_heading / 2)
    delta_qz = np.sin(delta_heading / 2)

    for cam_name in updated_cam_infos:
        initial_cam_ego_pose = np.array(initial_cam_infos[cam_name]['ego_pose'])
        offset_from_ego = initial_cam_ego_pose[:3] - initial_ego_translation
        rotated_offset = np.array([
            offset_from_ego[0] * cos_dh - offset_from_ego[1] * sin_dh,
            offset_from_ego[0] * sin_dh + offset_from_ego[1] * cos_dh,
            offset_from_ego[2]
        ])
        new_cam_xyz = ego_translation + rotated_offset

        # q_new = q_delta * q_init
        init_qw, init_qx, init_qy, init_qz = initial_cam_ego_pose[3:7]
        new_qw = delta_qw * init_qw - delta_qz * init_qz
        new_qx = delta_qw * init_qx + delta_qz * init_qy
        new_qy = delta_qw * init_qy - delta_qz * init_qx
        new_qz = delta_qw * init_qz + delta_qz * init_qw

        updated_cam_infos[cam_name]['ego_pose'] = [
            float(new_cam_xyz[0]), float(new_cam_xyz[1]), float(new_cam_xyz[2]),
            float(new_qw), float(new_qx), float(new_qy), float(new_qz)
        ]

    return updated_cam_infos


class SequenceSaver:
    """ Writes one simulation rollout as <output_dir>/<sequence_id>/frame_XXXX.pkl (schema: README, Simulation)."""

    def __init__(self, output_dir, sequence_id, initial_scene_dict):
        self.sequence_dir = os.path.join(output_dir, sequence_id)
        os.makedirs(self.sequence_dir, exist_ok=True)

        # static data from the initial scene
        self.lane_graph = initial_scene_dict['lane_graph']
        self.route = initial_scene_dict['route']
        self.route_lane_indices = initial_scene_dict.get('route_lane_indices', None)
        self.num_lanes = initial_scene_dict['num_lanes']
        self.initial_cam_infos = copy.deepcopy(initial_scene_dict.get('cam_infos', None))
        self.initial_ego_state_og = initial_scene_dict.get('ego_state_og', None)
        self.additional_keys = {k: initial_scene_dict[k]
                                for k in ['ego_state_og', 'route_completed', 'tile_occupancy']
                                if k in initial_scene_dict}

    def save_frame(self, frame_idx, state):
        """ Save one frame given Simulator.get_serializable_state() output.

        Agents/lanes are in the CURRENT ego frame; ego_state_og / cam_infos stay global.
        The ego is appended as the last agent.
        """
        agent_types = state['agent_types']
        agents = np.vstack([state['agents_ego_centered'], state['ego_state_ego_centered'][None, :]])[:, None, :]
        agent_types = np.vstack([agent_types, np.array([[0, 1, 0, 0, 0]], dtype=agent_types.dtype)])

        frame_data = {
            'agents': agents,  # [N+1, 1, 10]
            'agent_types': agent_types,  # [N+1, 5]
            'lanes': state['lanes_ego_centered'],
            'lane_graph': self.lane_graph,
            'route': self.route,
            'route_lane_indices': self.route_lane_indices,
            'num_agents': len(agents),
            'num_lanes': self.num_lanes,
        }

        cam_infos = state.get('cam_infos', None)
        ego_state_og = state.get('ego_state_og', None)
        if cam_infos is not None and ego_state_og is not None:
            cam_infos = update_camera_extrinsics(
                cam_infos, ego_state_og, self.initial_ego_state_og, self.initial_cam_infos)
        if cam_infos is not None:
            frame_data['cam_infos'] = cam_infos

        if state.get('agent_active', None) is not None:
            frame_data['agent_active'] = np.append(state['agent_active'], True)  # ego always active

        additional_keys = dict(self.additional_keys)
        if ego_state_og is not None:
            additional_keys['ego_state_og'] = ego_state_og
        for key, value in additional_keys.items():
            frame_data.setdefault(key, value)

        with open(os.path.join(self.sequence_dir, f"frame_{frame_idx:04d}.pkl"), 'wb') as f:
            pickle.dump(frame_data, f)
