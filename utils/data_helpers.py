import numpy as np
import torch
np.set_printoptions(suppress=True)
from utils.data_container import get_batches, get_features
from typing import Any, Dict, Optional, Tuple, Any, Dict, List
from cfgs.config import CONFIG_PATH, NUPLAN_VEHICLE, NUPLAN_PEDESTRIAN, NUPLAN_STATIC_OBJECT, PARTITIONED
import os
import pickle
import json, re

def extract_raw_waymo_data(agents_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the list-of-dict agent format from Waymo to flat arrays.

    Parameters
    ----------
    agents_data
        List where each element corresponds to a single agent and
        replicates Waymo's *per-time-step* trajectory dictionaries.

    Returns
    -------
    agent_data
        Array with shape ``(num_agents, T, 8)`` containing position
        ``(x, y)``, velocity ``(vx, vy)``, heading *(rad)*, length,
        width and existence mask for each time-step ``T``.
    agent_types
        One-hot encoded array of shape ``(num_agents, 5)`` for
        ``{"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}``.
    """
    
    # Get indices of non-parked cars and cars that exist for the entire episode
    agent_data = []
    agent_types = []

    for n in range(len(agents_data)):
        # Position ---------------------------------------------------
        ag_position = agents_data[n]['position']
        x_values = [entry['x'] for entry in ag_position]
        y_values = [entry['y'] for entry in ag_position]
        ag_position = np.column_stack((x_values, y_values))
        
        # Heading (unwrap to (‑pi, pi]) ------------------------------
        ag_heading = np.radians(np.array(agents_data[n]['heading']).reshape((-1, 1)))
        ag_heading = np.mod(ag_heading + np.pi, 2 * np.pi) - np.pi
        
        # Velocity ---------------------------------------------------
        ag_velocity = agents_data[n]['velocity']
        x_values = [entry['x'] for entry in ag_velocity]
        y_values = [entry['y'] for entry in ag_velocity]
        ag_velocity = np.column_stack((x_values, y_values))
        
        # Existence & size -----------------------------------------
        ag_existence = np.array(agents_data[n]['valid']).reshape((-1, 1))
        ag_length = np.ones((len(ag_position), 1)) * agents_data[n]['length']
        ag_width = np.ones((len(ag_position), 1)) * agents_data[n]['width']
        
        # Pack -------------------------------------------------------
        agent_type = get_object_type_onehot_waymo(agents_data[n]['type'])
        ag_state = np.concatenate((ag_position, ag_velocity, ag_heading, ag_length, ag_width, ag_existence), axis=-1)
        agent_data.append(ag_state)
        agent_types.append(agent_type)
    
    # convert to numpy array
    agent_data = np.array(agent_data)
    agent_types = np.array(agent_types)
    
    return agent_data, agent_types

def add_batch_dim(arr):
    return np.expand_dims(arr, axis=0)

def extract_raw_waymo_data(agents_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the list-of-dict agent format from Waymo to flat arrays.

    Parameters
    ----------
    agents_data
        List where each element corresponds to a single agent and
        replicates Waymo's *per-time-step* trajectory dictionaries.

    Returns
    -------
    agent_data
        Array with shape ``(num_agents, T, 8)`` containing position
        ``(x, y)``, velocity ``(vx, vy)``, heading *(rad)*, length,
        width and existence mask for each time-step ``T``.
    agent_types
        One-hot encoded array of shape ``(num_agents, 5)`` for
        ``{"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}``.
    """
    
    # Get indices of non-parked cars and cars that exist for the entire episode
    agent_data = []
    agent_types = []

    for n in range(len(agents_data)):
        # Position ---------------------------------------------------
        ag_position = agents_data[n]['position']
        x_values = [entry['x'] for entry in ag_position]
        y_values = [entry['y'] for entry in ag_position]
        ag_position = np.column_stack((x_values, y_values))
        
        # Heading (unwrap to (‑pi, pi]) ------------------------------
        ag_heading = np.radians(np.array(agents_data[n]['heading']).reshape((-1, 1)))
        ag_heading = np.mod(ag_heading + np.pi, 2 * np.pi) - np.pi
        
        # Velocity ---------------------------------------------------
        ag_velocity = agents_data[n]['velocity']
        x_values = [entry['x'] for entry in ag_velocity]
        y_values = [entry['y'] for entry in ag_velocity]
        ag_velocity = np.column_stack((x_values, y_values))
        
        # Existence & size -----------------------------------------
        ag_existence = np.array(agents_data[n]['valid']).reshape((-1, 1))
        ag_length = np.ones((len(ag_position), 1)) * agents_data[n]['length']
        ag_width = np.ones((len(ag_position), 1)) * agents_data[n]['width']
        
        # Pack -------------------------------------------------------
        agent_type = get_object_type_onehot_waymo(agents_data[n]['type'])
        ag_state = np.concatenate((ag_position, ag_velocity, ag_heading, ag_length, ag_width, ag_existence), axis=-1)
        agent_data.append(ag_state)
        agent_types.append(agent_type)
    
    # convert to numpy array
    agent_data = np.array(agent_data)
    agent_types = np.array(agent_types)
    
    return agent_data, agent_types

def add_batch_dim(arr):
    return np.expand_dims(arr, axis=0)

def get_object_type_onehot_waymo(agent_type):
    """Return the one-hot NumPy vector encoding of an agent type."""
    agent_types = {"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}
    return np.eye(len(agent_types))[agent_types[agent_type]]

def get_lane_connection_type_onehot_waymo(lane_connection_type):
    """Return the one-hot NumPy vector encoding of a lane-connection type."""
    lane_connection_types = {"none": 0, "pred": 1, "succ": 2, "left": 3, "right": 4, "self": 5}
    return np.eye(len(lane_connection_types))[lane_connection_types[lane_connection_type]]

def get_lane_connection_type_onehot_nuplan(lane_connection_type):
    """Converts a lane connection type to a one-hot encoded vector."""
    lane_connection_types = {"none": 0, "pred": 1, "succ": 2, "self": 3}
    return np.eye(len(lane_connection_types))[lane_connection_types[lane_connection_type]]

def get_lane_type_onehot_nuplan(lane_type):
    """Converts a lane type to a one-hot encoded vector."""
    lane_types = {"lane": 0, "green_light": 1, "red_light": 2}
    return np.eye(len(lane_types))[lane_types[lane_type]]

def get_object_type_onehot_nuplan(agent_type):
    """Converts an agent type to a one-hot encoded vector."""
    agent_types = {"vehicle": 0, "pedestrian": 1, "static_object": 2}
    return np.eye(len(agent_types))[agent_types[agent_type]]

def compute_fov_mask_from_normalized_xy(x, y, fov_deg=70):
    # Condition: must be in front of the camera
    front_mask = x > 0

    # horizontal FOV angle
    half_fov = np.deg2rad(fov_deg / 2)
    angles = np.arctan2(y, x)

    fov_mask = np.abs(angles) <= half_fov

    # both must be satisfied
    return front_mask & fov_mask



def reorder_indices(
        agent_mu: np.ndarray,
        agent_log_var: np.ndarray,
        lane_mu: np.ndarray,
        lane_log_var: np.ndarray,
        edge_index_lane_to_lane: np.ndarray,
        agent_states: np.ndarray,
        road_points: np.ndarray,
        lg_type: int,
        tolerance: float = 0.5 / 32,
        dataset: str = 'waymo',
        return_loss_mask: bool = False
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
    """Reorder agents and lanes to ensure deterministic ordering. This makes the positional
    encodings more meaningful.

    The routine performs a **hierarchical sort** on non-ego agents and on road
    lanes over the following metrics in the prescribed order: [min_y, min_x, max_y, max_x]

    A *tolerance* is applied so that small positional differences do not change the order.  After sorting, all latent
    representations, state tensors, and graph indices are permuted
    consistently.  The ego agent (index 0) is **never moved**.

    Parameters
    ----------
    agent_mu : np.ndarray
        Mean of the Gaussian latent variables for each agent with shape
        ``(N_agents, latent_dim)``.
    agent_log_var : np.ndarray
        Log-variance of the Gaussian latent variables for each agent with the
        same shape as *agent_mu*.
    lane_mu : np.ndarray
        Mean of the Gaussian latent variables for each lane with shape
        ``(N_lanes, latent_dim)``.
    lane_log_var : np.ndarray
        Log-variance of the Gaussian latent variables for each lane with the
        same shape as *lane_mu*.
    edge_index_lane_to_lane : np.ndarray
        Edge list of the lane-to-lane graph in COO format with shape
        ``(2, E)`` or ``(E, 2)``.  Indices are updated to reflect the new lane
        order.
    agent_states : np.ndarray
        Full state tensor for agents used to derive the sort keys.
    road_points : np.ndarray
        Sampled poly-line points for each lane with shape
        ``(N_lanes, N_points, 2)``.
    lg_type : int
        Scene layout type.  If ``lg_type == 1`` the function marks agents and
        lanes that are south of the horizontal partition (``y <= 0``).
        Otherwise no partitioning mask is applied.
    tolerance : float, optional
        Numerical tolerance (in the same units as coordinates) within which
        metric differences are considered equal.  Defaults to ``0.5 / 32``
        (≈0.0156).
    dataset : str, optional
        Either waymo or nuplan, which determines orientation of scene and therefore recursive ordering

    Returns
    -------
    Tuple[np.ndarray, ...]
        A 7-tuple containing:

        1. **agent_mu_sorted** - permuted *agent_mu* with ego agent first.
        2. **agent_log_var_sorted** - permuted *agent_log_var*.
        3. **lane_mu_sorted** - permuted *lane_mu*.
        4. **lane_log_var_sorted** - permuted *lane_log_var*.
        5. **edge_index_lane_to_lane_new** - updated edge indices.
        6. **agent_partition_mask** - boolean mask of shape ``(N_agents,)``
        indicating agents below the ``y=0`` partition when
        ``lg_type == 1``.
        7. **lane_partition_mask** - boolean mask of shape ``(N_lanes,)``
        indicating lanes below the partition when ``lg_type == 1``.

    Notes
    -----
    • The sorting of agents excludes the ego agent (index ``0``), which is
    re-inserted at the head of every returned tensor.

    • When *road_points* is empty (``shape[0] == 0``) the lane-related outputs
    are returned unchanged.
    """
    
    def hierarchical_sort(values, metrics, tolerance):
        """
        Recursively sorts indices based on a list of metrics and a tolerance.
        """
        indices = np.arange(len(values[metrics[0]]))
        
        def sort_recursive(indices, metric_idx):
            if len(indices) == 0:
                return indices  # Return empty array if no indices to sort
            if metric_idx >= len(metrics):
                return indices
            
            metric = metrics[metric_idx]
            values_metric = values[metric][indices]
            sorted_order = np.argsort(values_metric)
            indices = indices[sorted_order]
            values_metric_sorted = values_metric[sorted_order]
            
            # Group indices where the difference is less than tolerance
            groups = []
            current_group = [indices[0]]
            for i in range(1, len(indices)):
                diff = values_metric_sorted[i] - values_metric_sorted[i - 1]
                if diff < tolerance:
                    current_group.append(indices[i])
                else:
                    # Recursively sort the current group if needed
                    if len(current_group) > 1:
                        current_group = sort_recursive(np.array(current_group), metric_idx + 1).tolist()
                    groups.extend(current_group)
                    current_group = [indices[i]]
            # Handle the last group
            if len(current_group) > 1:
                current_group = sort_recursive(np.array(current_group), metric_idx + 1).tolist()
            groups.extend(current_group)
            return np.array(groups)
        
        return sort_recursive(indices, 0)
    
    if dataset == 'waymo':
        PARTITION_IDX = 1  # y-axis partition for Waymo 
    else:
        PARTITION_IDX = 0 # x-axis partition for Nuplan
    
    # Process Agents (ego is first index)
    non_ego_agent_mu = agent_mu[1:]
    non_ego_agent_log_var = agent_log_var[1:]
    non_ego_agent_states = agent_states[1:]
    
    if non_ego_agent_states.shape[0] > 0:
        # Calculate metrics for agents
        agent_min_y = non_ego_agent_states[:, 1]
        agent_min_x = non_ego_agent_states[:, 0]
        agent_max_y = non_ego_agent_states[:, 1]
        agent_max_x = non_ego_agent_states[:, 0]
        
        agent_values = {
            'min_y': agent_min_y,
            'min_x': agent_min_x,
            'max_y': agent_max_y,
            'max_x': agent_max_x
        }
        
        if dataset == 'waymo':
            agent_metrics = ['min_y', 'min_x', 'max_y', 'max_x']
        else:
            agent_metrics = ['min_x', 'min_y', 'max_x', 'max_y']
        perm = hierarchical_sort(agent_values, agent_metrics, tolerance)
        
        # Reorder non-ego agents
        non_ego_agent_mu = non_ego_agent_mu[perm]
        non_ego_agent_log_var = non_ego_agent_log_var[perm]
        non_ego_agent_states = non_ego_agent_states[perm]
    
    # Concatenate ego agent back
    agent_mu = np.concatenate([agent_mu[:1], non_ego_agent_mu], axis=0)
    agent_log_var = np.concatenate([agent_log_var[:1], non_ego_agent_log_var], axis=0)
    agent_states_sorted = np.concatenate([agent_states[:1], non_ego_agent_states], axis=0)

    # which agents are below the partition
    if lg_type == PARTITIONED:
        agent_partition_mask = agent_states_sorted[:, PARTITION_IDX] <= 0
    else:
        agent_partition_mask = np.zeros_like(agent_states_sorted[:, PARTITION_IDX] <= 0)
    
    if road_points.shape[0] > 0:
        lane_min_y = np.min(road_points[:, :, 1], axis=1)
        lane_min_x = np.min(road_points[:, :, 0], axis=1)
        lane_max_y = np.max(road_points[:, :, 1], axis=1)
        lane_max_x = np.max(road_points[:, :, 0], axis=1)
        
        lane_values = {
            'min_y': lane_min_y,
            'min_x': lane_min_x,
            'max_y': lane_max_y,
            'max_x': lane_max_x
        }
        
        if dataset == 'waymo':
            lane_metrics = ['min_y', 'min_x', 'max_y', 'max_x']
        else:
            lane_metrics = ['min_x', 'min_y', 'max_x', 'max_y']
        lane_perm = hierarchical_sort(lane_values, lane_metrics, tolerance)
        
        # Reorder lanes
        lane_mu = lane_mu[lane_perm]
        lane_log_var = lane_log_var[lane_perm]

        road_points_sorted = road_points[lane_perm]
        # which roads are below the partition
        if lg_type == PARTITIONED:
            lane_partition_mask = road_points_sorted[:, 9, PARTITION_IDX] <= 0
        else:
            lane_partition_mask = np.zeros_like(road_points_sorted[:, 9, PARTITION_IDX] <= 0)
        
        # Update edge indices
        old_index_positions = np.argsort(lane_perm)
        edge_index_lane_to_lane_new = old_index_positions[edge_index_lane_to_lane]
    else:
        edge_index_lane_to_lane_new = edge_index_lane_to_lane  # No change if no lanes
        # no lanes
        lane_partition_mask = road_points[:, 9, PARTITION_IDX] <= 0
    
    # which agents are below the partition
    if return_loss_mask:
        agent_loss_mask = agent_states_sorted[:, PARTITION_IDX] >= 0
        lane_loss_mask =  road_points_sorted[:, 9, PARTITION_IDX] >= 0
        return agent_mu, agent_log_var, lane_mu, lane_log_var, edge_index_lane_to_lane_new, agent_partition_mask, lane_partition_mask, agent_loss_mask, lane_loss_mask

    # ============================================================
    # For lg_type == 2: compute camera FOV masks (using normalized xy)
    # ============================================================
    if lg_type == 2:
        # --- Agents ---
        agent_x = agent_states_sorted[:, 0]
        agent_y = agent_states_sorted[:, 1]
        ego_x, ego_y = agent_x[0], agent_y[0]  # camera position
        agent_x = agent_x - ego_x
        agent_y = agent_y - ego_y

        def compute_fov_mask_from_normalized_xy(x, y, fov_deg=100):
            front = x > 0
            half_fov = np.deg2rad(fov_deg / 2)
            angles = np.arctan2(y, x)
            inside_fov = np.abs(angles) <= half_fov
            return front & inside_fov

        agent_fov_mask = compute_fov_mask_from_normalized_xy(agent_x, agent_y)
        agent_fov_mask[0] = True
        # --- Lanes ---
        if road_points.shape[0] > 0:
            lane_x = road_points_sorted[:, :, 0]
            lane_y = road_points_sorted[:, :, 1]
            lane_fov_mask = compute_fov_mask_from_normalized_xy(lane_x, lane_y)
            lane_fov_mask = lane_fov_mask.any(axis=1)
        else:
            lane_fov_mask = np.zeros(0, dtype=bool)

    else:
        agent_fov_mask = np.ones(agent_mu.shape[0], dtype=bool)
        lane_fov_mask = np.ones(lane_mu.shape[0], dtype=bool)

    return agent_mu, agent_log_var, lane_mu, lane_log_var, edge_index_lane_to_lane_new, agent_partition_mask, lane_partition_mask, agent_fov_mask, lane_fov_mask, agent_states_sorted, road_points_sorted


def modify_agent_states(agent_states):
    """Canonicalise velocity & heading for neural consumption. All remaining trailing columns (if any) are copied verbatim.

    Parameters
    ----------
    agent_states : np.ndarray [x, y, vel_x, vel_y, heading, length, width, z, height]
        Float32 array of shape ``(N, D)`` where columns ``2-4`` are
        ``vx``, ``vy``, and ``yaw`` respectively.

    Returns
    -------
    new_agent_states : np.ndarray [x, y, vel, cos(heading), sin(heading), length, width, z, height]
        Array with the *same* shape ``(N, D)`` where columns ``2-4``
        have been replaced by ``speed``, ``cosθ``, ``sinθ``.
    """
    new_agent_states = np.zeros_like(agent_states)
    new_agent_states[:, :2] = agent_states[:, :2]
    new_agent_states[:, 5:] = agent_states[:, 5:]
    new_agent_states[:, 2] = np.sqrt(agent_states[:, 2] ** 2 + agent_states[:, 3] ** 2)
    new_agent_states[:, 3] = np.cos(agent_states[:, 4])
    new_agent_states[:, 4] = np.sin(agent_states[:, 4])

    return new_agent_states


def normalize_scene(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max normalise agent and lane features into **[-1, 1]**."""
    
    # pos_x
    agent_states[:, 0] = 2 * ((agent_states[:, 0] - (-1 * fov/2))
                            / fov) - 1
    # pos_y
    agent_states[:, 1] = 2 * ((agent_states[:, 1] - (-1 * fov/2))
                            / fov) - 1
    # speed
    agent_states[:, 2] = 2 * ((agent_states[:, 2] - (min_speed))
                            / (max_speed - min_speed)) - 1
    # length
    agent_states[:, 5] = 2 * ((agent_states[:, 5] - (min_length))
                            / (max_length - min_length)) - 1
    # width
    agent_states[:, 6] = 2 * ((agent_states[:, 6] - (min_width))
                            / (max_width - min_width)) - 1
    
    # road pos_x
    road_points[:, :, 0] = 2 * ((road_points[:, :, 0] - (min_lane_x))
                            / (max_lane_x - min_lane_x)) - 1
    road_points[:, :, 1] = 2 * ((road_points[:, :, 1] - (min_lane_y))
                            / (max_lane_y - min_lane_y)) - 1

    return agent_states, road_points


def normalize_scene_2d(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float,
        mode: str = "centered"   # "centered" or "positive"
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max normalise agent and lane features into **[-1, 1]**."""
    
    if mode == "centered":
        # symmetric FOV: [-fov/2, +fov/2]
        x_min, x_max = -fov / 2, fov / 2
        y_min, y_max = -fov / 2, fov / 2
    elif mode == "positive":
        # positive range: [0, fov]
        x_min, x_max = 0, fov
        y_min, y_max = -fov / 2, fov / 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # position
    agent_states[:, 0] = 2 * ((agent_states[:, 0] - x_min) / (x_max - x_min)) - 1
    agent_states[:, 1] = 2 * ((agent_states[:, 1] - y_min) / (y_max - y_min)) - 1
    # speed
    agent_states[:, 2] = 2 * ((agent_states[:, 2] - (min_speed))
                            / (max_speed - min_speed)) - 1
    # length
    agent_states[:, 5] = 2 * ((agent_states[:, 5] - (min_length))
                            / (max_length - min_length)) - 1
    # width
    agent_states[:, 6] = 2 * ((agent_states[:, 6] - (min_width))
                            / (max_width - min_width)) - 1
    
    # road pos_x
    road_points[:, :, 0] = 2 * ((road_points[:, :, 0] - (x_min))
                            / (x_max - x_min)) - 1
    road_points[:, :, 1] = 2 * ((road_points[:, :, 1] - (y_min))
                            / (y_max - y_min)) - 1

    return agent_states, road_points


def normalize_scene_3d_post_process(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float,
        min_height: float,
        max_height: float,
        fov_z: float,
        mode: str = "centered"   # "centered" or "positive"
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max normalise agent and lane features into **[-1, 1]**."""
    # state format: [x, y, z, vel, cos(heading), sin(heading), length, width, height]
    if mode == "centered":
        # symmetric FOV: [-fov/2, +fov/2]
        x_min, x_max = -fov / 2, fov / 2
        y_min, y_max = -fov / 2, fov / 2
    elif mode == "positive":
        # positive range: [0, fov]
        x_min, x_max = 0, fov
        y_min, y_max = -fov / 2, fov / 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # position
    agent_states[:, 0] = 2 * ((agent_states[:, 0] - x_min) / (x_max - x_min)) - 1
    agent_states[:, 1] = 2 * ((agent_states[:, 1] - y_min) / (y_max - y_min)) - 1
    agent_states[:, 2] = 2 * ((agent_states[:, 2] - (-fov_z/2)) / fov_z) - 1


    # speed
    agent_states[:, 3] = 2 * ((agent_states[:, 3] - (min_speed))
                            / (max_speed - min_speed)) - 1
    # length
    agent_states[:, 6] = 2 * ((agent_states[:, 6] - (min_length))
                            / (max_length - min_length)) - 1
    # width
    agent_states[:, 7] = 2 * ((agent_states[:, 7] - (min_width))
                            / (max_width - min_width)) - 1
    # height
    agent_states[:, 8] = 2 * ((agent_states[:, 8] - (min_height))
                            / (max_height - min_height)) - 1

    # road pos_x
    road_points[:, :, 0] = 2 * ((road_points[:, :, 0] - (x_min))
                            / (x_max - x_min)) - 1
    road_points[:, :, 1] = 2 * ((road_points[:, :, 1] - (y_min))
                            / (y_max - y_min)) - 1

    return agent_states, road_points

def normalize_scene_3d(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float,
        min_height: float,
        max_height: float,
        fov_z: float,
        mode: str = "centered"   # "centered" or "positive"
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max normalise agent and lane features into **[-1, 1]**."""
    # state format: [x, y, vel, cos(heading), sin(heading), length, width, z, height]
    if mode == "centered":
        # symmetric FOV: [-fov/2, +fov/2]
        x_min, x_max = -fov / 2, fov / 2
        y_min, y_max = -fov / 2, fov / 2
    elif mode == "positive":
        # positive range: [0, fov]
        x_min, x_max = 0, fov
        y_min, y_max = -fov / 2, fov / 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # position
    agent_states[:, 0] = 2 * ((agent_states[:, 0] - x_min) / (x_max - x_min)) - 1
    agent_states[:, 1] = 2 * ((agent_states[:, 1] - y_min) / (y_max - y_min)) - 1
    agent_states[:, 7] = 2 * ((agent_states[:, 7] - (-fov_z/2)) / fov_z) - 1


    # speed
    agent_states[:, 2] = 2 * ((agent_states[:, 2] - (min_speed))
                            / (max_speed - min_speed)) - 1
    # length
    agent_states[:, 5] = 2 * ((agent_states[:, 5] - (min_length))
                            / (max_length - min_length)) - 1
    # width
    agent_states[:, 6] = 2 * ((agent_states[:, 6] - (min_width))
                            / (max_width - min_width)) - 1
    # height
    agent_states[:, 8] = 2 * ((agent_states[:, 8] - (min_height))
                            / (max_height - min_height)) - 1

    # road pos_x
    road_points[:, :, 0] = 2 * ((road_points[:, :, 0] - (x_min))
                            / (x_max - x_min)) - 1
    road_points[:, :, 1] = 2 * ((road_points[:, :, 1] - (y_min))
                            / (y_max - y_min)) - 1

    return agent_states, road_points


def unnormalize_scene_mixed_modes(data, agent_samples, lane_samples, cfg_dataset, dim="3d"):
    """
    Unnormalize agent and lane samples with different modes based on lg_type:
      - lg_type in {0, 1} → 'centered' mode
      - lg_type == 2 → 'positive' mode

    Handles missing splits safely and restores the original batch order.
    """
    lg_types = data['lg_type']  # shape: [num_batches]
    agent_batch = data['agent'].batch
    lane_batch = data['lane'].batch

    # Identify batches by lg_type
    centered_batches = (lg_types == 0) | (lg_types == 1)
    positive_batches = (lg_types == 2)

    # Masks for selecting corresponding agents/lanes
    agent_centered_mask = torch.isin(agent_batch, torch.nonzero(centered_batches, as_tuple=True)[0])
    agent_positive_mask = torch.isin(agent_batch, torch.nonzero(positive_batches, as_tuple=True)[0])

    lane_centered_mask = torch.isin(lane_batch, torch.nonzero(centered_batches, as_tuple=True)[0])
    lane_positive_mask = torch.isin(lane_batch, torch.nonzero(positive_batches, as_tuple=True)[0])

    # Split data by type
    agent_centered = agent_samples[agent_centered_mask]
    lane_centered = lane_samples[lane_centered_mask]
    agent_positive = agent_samples[agent_positive_mask]
    lane_positive = lane_samples[lane_positive_mask]

    # Prepare output placeholders
    agent_samples_unnorm = torch.zeros_like(agent_samples)
    lane_samples_unnorm = torch.zeros_like(lane_samples)

    # Helper function for unnormalizing safely
    def safe_unnorm(agent_split, lane_split, mode):
        if agent_split.numel() == 0 or lane_split.numel() == 0:
            return agent_split, lane_split
        
        if dim == "3d":
            return unnormalize_scene_3d(
                agent_split, lane_split,
                fov=cfg_dataset.fov,
                min_speed=cfg_dataset.min_speed,
                max_speed=cfg_dataset.max_speed,
                min_length=cfg_dataset.min_length,
                max_length=cfg_dataset.max_length,
                min_width=cfg_dataset.min_width,
                max_width=cfg_dataset.max_width,
                min_lane_x=cfg_dataset.min_lane_x,
                min_lane_y=cfg_dataset.min_lane_y,
                max_lane_x=cfg_dataset.max_lane_x,
                max_lane_y=cfg_dataset.max_lane_y,
                min_height=cfg_dataset.min_height,
                max_height=cfg_dataset.max_height,
                fov_z=cfg_dataset.fov_z,
                mode=mode
            )
        elif dim == "2d":
            return unnormalize_scene_2d(
                agent_split, lane_split,
                fov=cfg_dataset.fov,
                min_speed=cfg_dataset.min_speed,
                max_speed=cfg_dataset.max_speed,
                min_length=cfg_dataset.min_length,
                max_length=cfg_dataset.max_length,
                min_width=cfg_dataset.min_width,
                max_width=cfg_dataset.max_width,
                min_lane_x=cfg_dataset.min_lane_x,
                min_lane_y=cfg_dataset.min_lane_y,
                max_lane_x=cfg_dataset.max_lane_x,
                max_lane_y=cfg_dataset.max_lane_y,
                mode=mode
            )
        else:
            raise ValueError(f"Unknown dimension setting: {dim}")

    # Unnormalize for both modes
    agent_centered, lane_centered = safe_unnorm(agent_centered, lane_centered, mode="centered")
    agent_positive, lane_positive = safe_unnorm(agent_positive, lane_positive, mode="positive")

    # Reassemble results in original batch order
    agent_samples_unnorm[agent_centered_mask] = agent_centered
    agent_samples_unnorm[agent_positive_mask] = agent_positive
    lane_samples_unnorm[lane_centered_mask] = lane_centered
    lane_samples_unnorm[lane_positive_mask] = lane_positive

    return agent_samples_unnorm, lane_samples_unnorm


def unnormalize_scene(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Unnormalize the agent states and lane points from a range of [-1, 1] to their original scale based on the dataset configuration."""
    # pos_x
    agent_states[:, 0] = ((torch.clip(agent_states[:, 0], -1, 1) + 1) / 2) * fov + (-1 * fov/2)
    # pos_y
    agent_states[:, 1] = ((torch.clip(agent_states[:, 1], -1, 1) + 1) / 2) * fov + (-1 * fov/2)
    # speed
    agent_states[:, 2] = ((torch.clip(agent_states[:, 2], -1, 1) + 1) / 2) * (max_speed - min_speed) + min_speed
    # cos_theta
    agent_states[:, 3] = torch.clip(agent_states[:, 3], -1, 1)
    # sin_theta
    agent_states[:, 4] = torch.clip(agent_states[:, 4], -1, 1)
    # length
    agent_states[:, 5] = ((torch.clip(agent_states[:, 5], -1, 1) + 1) / 2) * (max_length - min_length) + min_length
    # width
    agent_states[:, 6] = ((torch.clip(agent_states[:, 6], -1, 1) + 1) / 2) * (max_width - min_width) + min_width

    lower_clip = -1000
    upper_clip = 1000
    
    # lane pos_x
    road_points[:, :, 0] = ((torch.clip(road_points[:, :, 0], lower_clip, upper_clip) + 1) / 2) * (max_lane_x - min_lane_x) + min_lane_x
    # lane pos_y
    road_points[:, :, 1] = ((torch.clip(road_points[:, :, 1], lower_clip, upper_clip) + 1) / 2) * (max_lane_y - min_lane_y) + min_lane_y

    return agent_states, road_points


def unnormalize_scene_2d(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float,
        mode: str = "centered"
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Unnormalize the agent states and lane points from a range of [-1, 1] to their original scale based on the dataset configuration."""
    if mode == "centered":
        x_min, x_max = -fov / 2, fov / 2
        y_min, y_max = -fov / 2, fov / 2
    elif mode == "positive":
        x_min, x_max = 0, fov
        y_min, y_max = -fov / 2, fov / 2
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # pos_x, pos_y
    agent_states[:, 0] = ((torch.clip(agent_states[:, 0], -1, 1) + 1) / 2) * (x_max - x_min) + x_min
    agent_states[:, 1] = ((torch.clip(agent_states[:, 1], -1, 1) + 1) / 2) * (y_max - y_min) + y_min
    # speed
    agent_states[:, 2] = ((torch.clip(agent_states[:, 2], -1, 1) + 1) / 2) * (max_speed - min_speed) + min_speed
    # cos_theta
    agent_states[:, 3] = torch.clip(agent_states[:, 3], -1, 1)
    # sin_theta
    agent_states[:, 4] = torch.clip(agent_states[:, 4], -1, 1)
    # length
    agent_states[:, 5] = ((torch.clip(agent_states[:, 5], -1, 1) + 1) / 2) * (max_length - min_length) + min_length
    # width
    agent_states[:, 6] = ((torch.clip(agent_states[:, 6], -1, 1) + 1) / 2) * (max_width - min_width) + min_width

    lower_clip = -1000
    upper_clip = 1000
    
    # lane pos_x
    road_points[:, :, 0] = ((torch.clip(road_points[:, :, 0], lower_clip, upper_clip) + 1) / 2) * (x_max - x_min) + x_min
    # lane pos_y
    road_points[:, :, 1] = ((torch.clip(road_points[:, :, 1], lower_clip, upper_clip) + 1) / 2) * (y_max - y_min) + y_min

    return agent_states, road_points

def unnormalize_scene_3d(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float,
        min_height: float,
        max_height: float,
        fov_z: float,
        mode: str = "centered"
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Unnormalize the agent states and lane points from a range of [-1, 1] to their original scale based on the dataset configuration."""
    # agent_state format: [x, y, z, vel, cos(heading), sin(heading), length, width, height]
    if mode == "centered":
        x_min, x_max = -fov / 2, fov / 2
        y_min, y_max = -fov / 2, fov / 2
    elif mode == "positive":
        x_min, x_max = 0, fov
        y_min, y_max = -fov / 2, fov / 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # positions
    agent_states[:, 0] = ((torch.clip(agent_states[:, 0], -1, 1) + 1) / 2) * (x_max - x_min) + x_min
    agent_states[:, 1] = ((torch.clip(agent_states[:, 1], -1, 1) + 1) / 2) * (y_max - y_min) + y_min
    agent_states[:, 2] = ((torch.clip(agent_states[:, 2], -1, 1) + 1) / 2) * fov_z + (-fov_z / 2)

    # speed
    agent_states[:, 3] = ((torch.clip(agent_states[:, 3], -1, 1) + 1) / 2) * (max_speed - min_speed) + min_speed
    # cos_theta
    agent_states[:, 4] = torch.clip(agent_states[:, 4], -1, 1)
    # sin_theta
    agent_states[:, 5] = torch.clip(agent_states[:, 5], -1, 1)
    # length
    agent_states[:, 6] = ((torch.clip(agent_states[:, 6], -1, 1) + 1) / 2) * (max_length - min_length) + min_length
    # width
    agent_states[:, 7] = ((torch.clip(agent_states[:, 7], -1, 1) + 1) / 2) * (max_width - min_width) + min_width
    # height
    agent_states[:, 8] = ((torch.clip(agent_states[:, 8], -1, 1) + 1) / 2) * (max_height - min_height) + min_height

    lower_clip = -1000
    upper_clip = 1000
    
    # lane pos_x
    road_points[:, :, 0] = ((torch.clip(road_points[:, :, 0], lower_clip, upper_clip) + 1) / 2) * (x_max - x_min) + x_min
    # lane pos_y
    road_points[:, :, 1] = ((torch.clip(road_points[:, :, 1], lower_clip, upper_clip) + 1) / 2) * (y_max - y_min) + y_min

    return agent_states, road_points


def randomize_indices(
    agent_states,
    agent_types,
    road_points,
    edge_index_lane_to_lane,
    lane_types=None):
    """Randomly permute non-ego agents and lane order during training."""
    non_ego_agent_states = agent_states[1:]
    non_ego_agent_types = agent_types[1:]

    num_non_ego_agents = len(non_ego_agent_states)
    perm = np.arange(num_non_ego_agents)
    np.random.shuffle(perm)
    non_ego_agent_states = non_ego_agent_states[perm]
    non_ego_agent_types = non_ego_agent_types[perm]

    agent_states = np.concatenate([agent_states[:1], non_ego_agent_states], axis=0)
    agent_types = np.concatenate([agent_types[:1], non_ego_agent_types], axis=0)

    lane_perm = np.arange(len(road_points))
    np.random.shuffle(lane_perm)
    road_points = road_points[lane_perm]
    if lane_types is not None:
        lane_types = lane_types[lane_perm]
    
    old_index_positions = np.argsort(lane_perm)
    edge_index_lane_to_lane_new = old_index_positions[edge_index_lane_to_lane]

    if lane_types is not None:
        return agent_states, agent_types, road_points, lane_types, edge_index_lane_to_lane_new
    else:
        return agent_states, agent_types, road_points, edge_index_lane_to_lane_new
    

def normalize_latents(
        agent_latents, 
        lane_latents,
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std):
    """ Normalize the agent and lane latents using the mean and std from the config."""
    agent_latents = (agent_latents - agent_latents_mean) / agent_latents_std
    lane_latents = (lane_latents - lane_latents_mean) / lane_latents_std

    return agent_latents, lane_latents


def unnormalize_latents(
        agent_latents, 
        lane_latents,
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std):
    """ Unnormalize the agent and lane latents using the mean and std from the config."""
    agent_latents = agent_latents * agent_latents_std + agent_latents_mean
    lane_latents = lane_latents * lane_latents_std + lane_latents_mean

    return agent_latents, lane_latents


def reparameterize(mu, log_var):
    """ Reparameterization trick to sample from a Gaussian distribution
    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution.
        log_var (torch.Tensor): Log variance of the Gaussian distribution.
    Returns:
        torch.Tensor: Sampled latent variable.
    """
    assert mu.shape == log_var.shape
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def sample_latents(
        data, 
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std,
        normalize=True):
    """ Sample latents from the agent and lane data, and (optionally) normalize them."""
    agent_mu = data['agent'].x
    agent_log_var = data['agent'].log_var 
    agent_latents = reparameterize(agent_mu, agent_log_var)

    lane_mu = data['lane'].x 
    lane_log_var = data['lane'].log_var 
    lane_latents = reparameterize(lane_mu, lane_log_var)

    if normalize:
        agent_latents, lane_latents = normalize_latents(
            agent_latents, 
            lane_latents,
            agent_latents_mean,
            agent_latents_std,
            lane_latents_mean,
            lane_latents_std)
    
    return agent_latents, lane_latents


def convert_batch_to_scenarios(data, batch_size, batch_idx, cache_dir, conditioning_filenames=None, cache_samples=False, cache_lane_types=False, mode='initial_scene', raw_file_names=None):
    """ Converts batch output into individual scenarios. Optionally saves scenarios to disk."""
    if cache_samples and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    num_samples_in_batch = data.batch_size
    agent_batch, lane_batch, lane_conn_batch = get_batches(data)
    x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)
    if mode == 'inpainting':
        x_lane_mask = data['lane'].mask.float() # mask indicating lanes before partition
        x_agent_mask = data['agent'].mask.float() # mask indicating agents before partition
        x_lane_ids = data['lane'].ids # ids of the lanes before partition
    
    # move to cpu
    lg_type = data['lg_type'].cpu().numpy()
    map_ids = data['map_id'].cpu().numpy()
    x_agent_states = x_agent_states.cpu().numpy()
    x_agent_types = x_agent_types.cpu().numpy()
    if cache_lane_types:
        x_lane_types = x_lane_types.cpu().numpy()
    x_lane_states = x_lane_states.cpu().numpy()
    x_lane_conn = x_lane_conn.cpu().numpy()
    agent_batch = agent_batch.cpu().numpy()
    lane_batch = lane_batch.cpu().numpy()
    lane_conn_batch = lane_conn_batch.cpu().numpy()
    cam_infos = data.get('cam_infos', None)
    ego_state_og_list = data.get('ego_state_og', None)
    if mode == 'inpainting':
        x_lane_mask = x_lane_mask.cpu().numpy()
        x_agent_mask = x_agent_mask.cpu().numpy()
        x_lane_ids = x_lane_ids.cpu().numpy()

    batch_of_scenarios = {}
    for i in range(num_samples_in_batch):
        map_id_i = map_ids[i]  
        scene_i_agents = x_agent_states[agent_batch == i]
        scene_i_lanes = x_lane_states[lane_batch == i]
        scene_i_agent_types = x_agent_types[agent_batch == i]
        if cache_lane_types:
            scene_i_lane_types = x_lane_types[lane_batch == i]
        scene_i_lane_conns = x_lane_conn[lane_conn_batch == i]
        lg_type_i = lg_type[i]
        if mode == 'inpainting':
            scene_i_lane_mask = x_lane_mask[lane_batch == i]
            scene_i_agent_mask = x_agent_mask[agent_batch == i]
            scene_i_lane_ids = x_lane_ids[lane_batch == i]
        
        data = {
            'num_agents': len(scene_i_agents),
            'num_lanes': len(scene_i_lanes),
            'map_id': map_id_i,
            'lg_type': lg_type_i,
            'agent_states': scene_i_agents,
            'road_points': scene_i_lanes,
            'agent_types': scene_i_agent_types,
            'road_connection_types': scene_i_lane_conns
        }
        if cam_infos is not None:
            cam_info_i = {}
            for k, v in cam_infos['CAM_F0'].items():
                if torch.is_tensor(v):
                    cam_info_i[k] = v[i].cpu().numpy()
                else:
                    cam_info_i[k] = v[i]   # keep lists/strings as they are
            data['cam_infos'] = {'CAM_F0': cam_info_i}
        if ego_state_og_list is not None:
            ego_dims = [3, 4, 3, 1]  # per-sample sizes for each list entry

            ego_i_list = []
            for t, d in zip(ego_state_og_list, ego_dims):
                s, e = i * d, (i + 1) * d
                ego_i = t[s:e].detach().cpu().numpy() if torch.is_tensor(t[s:e]) else t[s:e]
                ego_i_list.append(ego_i)

            data['ego_state_og'] = ego_i_list
        if cache_lane_types:
            data['lane_types'] = scene_i_lane_types
        if mode == 'inpainting':
            data['lane_mask'] = scene_i_lane_mask
            data['agent_mask'] = scene_i_agent_mask
            data['lane_ids'] = scene_i_lane_ids

        if mode != 'inpainting':
            scenario_id = f"{i}_{batch_idx}" if raw_file_names is None else raw_file_names[batch_idx]
        else:
            scenario_id = conditioning_filenames[int(batch_idx * batch_size + i)]
        filename = f"{scenario_id}.pkl"

        batch_of_scenarios[scenario_id] = data
        
        if cache_samples:
            with open(os.path.join(cache_dir, filename), 'wb') as f:
                pickle.dump(data, f)

    return batch_of_scenarios


def _try_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _extract_codefence(text: str) -> Optional[str]:
    """
    Return the JSON string inside ```json ... ``` or ``` ... ``` fences, if present.
    """
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    return m.group(1) if m else None

def _extract_braced_json(text: str) -> Optional[str]:
    """
    Fallback: grab the largest {...} block.
    """
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None

def parse_caption(raw: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Robustly parse a caption file that may be:
      - plain text,
      - a JSON dict with keys like {"per_view": ..., "fused": "```json {...} ```"},
      - just a JSON object,
      - or a string containing a fenced JSON blob.

    Returns: (parsed_dict_or_None, text_for_T5)
      - parsed_dict_or_None: structured dict if we found one; else None
      - text_for_T5: a clean text string to feed your T5
    """
    # If file was already loaded as dict
    if isinstance(raw, dict):
        fused = raw.get("fused")
        if isinstance(fused, dict):
            parsed = fused
        elif isinstance(fused, str):
            inner = _extract_codefence(fused) or _extract_braced_json(fused) or fused
            parsed = _try_json_load(inner)
        else:
            parsed = raw  # already a dict with fields
    else:
        # raw is a string
        s = str(raw)

        # 1) try whole string as JSON
        parsed = _try_json_load(s)
        if isinstance(parsed, dict) and "fused" in parsed:
            fused = parsed.get("fused")
            if isinstance(fused, dict):
                parsed = fused
            elif isinstance(fused, str):
                inner = _extract_codefence(fused) or _extract_braced_json(fused) or fused
                parsed2 = _try_json_load(inner)
                parsed = parsed2 or parsed
        elif parsed is None:
            # 2) try fenced JSON
            inner = _extract_codefence(s) or _extract_braced_json(s)
            parsed = _try_json_load(inner) if inner else None

    # Build a clean text for T5 (fallback to the raw string if nothing parsed)
    if parsed is not None:
        # Prefer a concise, stable summary if present; else synthesize one.
        parts = []
        sd = parsed.get("scene_description")
        if isinstance(sd, str) and sd.strip():
            parts.append(sd.strip())

        scen = parsed.get("scenario_type")
        if scen: parts.append(f"scenario: {scen}")

        rl = parsed.get("road_layout")
        if isinstance(rl, dict):
            t = rl.get("type"); curv = rl.get("curvature")
            if t or curv:
                parts.append(f"road: {t or ''} {curv or ''}".strip())

        w = parsed.get("weather")
        if w: parts.append(f"weather: {w}")

        tod = parsed.get("time_of_day")
        if tod: parts.append(f"time: {tod}")

        vis = parsed.get("visibility")
        if vis: parts.append(f"visibility: {vis}")

        # traffic signs/lights (optional, concise)
        ts = parsed.get("traffic_signs")
        if isinstance(ts, list) and ts:
            parts.append(f"signs: {', '.join(map(str, ts))[:80]}")

        tl = parsed.get("traffic_lights")
        if isinstance(tl, dict) and tl:
            parts.append(f"lights: {', '.join(k for k,v in tl.items() if v)[:80]}")

        text_for_t5 = ". ".join(parts) if parts else json.dumps(parsed, ensure_ascii=False)
    else:
        text_for_t5 = raw if isinstance(raw, str) else str(raw)

    return parsed, text_for_t5

def get_agents_within_fov(agent_states, agent_types, cfg, front_only=False):
    """ Filters agents that are within the field of view (fov) and returns the closest agents
    to the origin, up to the specific max number of vehicles, pedestrians, and static objects."""

    # filter agents that are within the field of view (fov)
    agents_in_fov_x = np.abs(agent_states[:, 0]) < (cfg.fov / 2) if not front_only else (agent_states[:, 0] < cfg.fov) & (agent_states[:, 0] >= 0)
    agents_in_fov_y = np.abs(agent_states[:, 1]) < (cfg.fov / 2)
    agents_in_fov_mask = agents_in_fov_x * agents_in_fov_y
    valid_agents = np.where(agents_in_fov_mask > 0)[0]
    valid_vehicles = np.array(list(set(valid_agents).intersection(set(np.where(agent_types[:, NUPLAN_VEHICLE] == 1)[0]))))
    valid_pedestrians = np.array(list(set(valid_agents).intersection(set(np.where(agent_types[:, NUPLAN_PEDESTRIAN] == 1)[0]))))
    valid_static_objects = np.array(list(set(valid_agents).intersection(set(np.where(agent_types[:, NUPLAN_STATIC_OBJECT] == 1)[0]))))

    # find closest agents to the origin that are within the field of view, up to the specific max number
    dist_to_origin = np.linalg.norm(agent_states[:, :2], axis=-1)
    closest_ag_ids = np.argsort(dist_to_origin)
    
    closest_veh_ids = closest_ag_ids[np.in1d(closest_ag_ids, valid_vehicles)]
    closest_veh_ids = closest_veh_ids[:cfg.max_num_vehicles]

    closest_ped_ids = closest_ag_ids[np.in1d(closest_ag_ids, valid_pedestrians)]
    closest_ped_ids = closest_ped_ids[:cfg.max_num_pedestrians]
    
    closest_static_obj_ids = closest_ag_ids[np.in1d(closest_ag_ids, valid_static_objects)]
    closest_static_obj_ids = closest_static_obj_ids[:cfg.max_num_static_objects]
    
    closest_ag_ids = np.concatenate([closest_veh_ids, closest_ped_ids, closest_static_obj_ids], axis=0)
    return agent_states[closest_ag_ids], agent_types[closest_ag_ids]

def extract_agents(ego, vehicles, pedestrians, static_objects, ego_length, ego_width):
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



def diff_unnorm_scene_by_mode(data, agent_norm, lane_norm, cfg, dim="3d"):
    """
    Differentiable unnormalization for agent & lane tensors based on lg_type mode.

    lg_type:
      0 or 1 → 'centered' mode  (scene centered at ego)
      2      → 'positive' mode  (scene shifted to positive x)

    Args:
        data: batch dict containing 'lg_type', 'agent'.batch, 'lane'.batch
        agent_norm, lane_norm: normalized tensors
        cfg: dataset config with min/max bounds
        dim: "2d" or "3d"
    Returns:
        agent_unnorm, lane_unnorm (same shapes as input)
    """
    lg_types = data["lg_type"]
    agent_batch = data["agent"].batch
    lane_batch = data["lane"].batch

    centered_batches = (lg_types == 0) | (lg_types == 1)
    positive_batches = (lg_types == 2)

    agent_centered_mask = torch.isin(agent_batch, torch.nonzero(centered_batches, as_tuple=True)[0])
    agent_positive_mask = torch.isin(agent_batch, torch.nonzero(positive_batches, as_tuple=True)[0])
    lane_centered_mask  = torch.isin(lane_batch,  torch.nonzero(centered_batches, as_tuple=True)[0])
    lane_positive_mask  = torch.isin(lane_batch,  torch.nonzero(positive_batches, as_tuple=True)[0])

    agent_centered = agent_norm[agent_centered_mask]
    lane_centered  = lane_norm[lane_centered_mask]
    agent_positive = agent_norm[agent_positive_mask]
    lane_positive  = lane_norm[lane_positive_mask]

    # Pick proper version
    if dim == "3d":
        unnorm_fn = diff_unnorm_scene3d
    elif dim == "2d":
        unnorm_fn = diff_unnorm_scene2d
    else:
        raise ValueError(f"Unknown dim={dim}")

    # Apply
    if agent_centered.numel() > 0:
        agent_centered, lane_centered = unnorm_fn(agent_centered, lane_centered, cfg, mode="centered")
    if agent_positive.numel() > 0:
        agent_positive, lane_positive = unnorm_fn(agent_positive, lane_positive, cfg, mode="positive")

    # Merge back
    agent_out = torch.zeros_like(agent_norm)
    lane_out  = torch.zeros_like(lane_norm)
    agent_out[agent_centered_mask] = agent_centered
    agent_out[agent_positive_mask] = agent_positive
    lane_out[lane_centered_mask]   = lane_centered
    lane_out[lane_positive_mask]   = lane_positive

    return agent_out, lane_out


def diff_unnorm_scene2d(agent_norm, lane_norm, cfg, mode="centered"):
    """
    Differentiable 2D unnormalization: [-1,1] → metric coordinates.
    Keeps gradients intact.
    """
    fov = cfg.fov
    if mode == "centered":
        x_min, x_max = -fov / 2, fov / 2
        y_min, y_max = -fov / 2, fov / 2
    elif mode == "positive":
        x_min, x_max = 0, fov
        y_min, y_max = -fov / 2, fov / 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    def scale(t, tmin, tmax):
        return ((t.clamp(-1, 1) + 1) / 2) * (tmax - tmin) + tmin

    x = scale(agent_norm[:, 0], x_min, x_max)
    y = scale(agent_norm[:, 1], y_min, y_max)
    vel = scale(agent_norm[:, 2], cfg.min_speed, cfg.max_speed)
    cos_h = agent_norm[:, 3].clamp(-1, 1)
    sin_h = agent_norm[:, 4].clamp(-1, 1)
    length = scale(agent_norm[:, 5], cfg.min_length, cfg.max_length)
    width  = scale(agent_norm[:, 6], cfg.min_width,  cfg.max_width)

    parts = [x, y, vel, cos_h, sin_h, length, width]
    if agent_norm.shape[1] > 7:
        parts += [agent_norm[:, 7:]]
    agent_out = torch.cat([p.unsqueeze(-1) if p.ndim == 1 else p for p in parts], dim=-1)

    lane_x = scale(lane_norm[..., 0], x_min, x_max)
    lane_y = scale(lane_norm[..., 1], y_min, y_max)
    lane_out = torch.stack([lane_x, lane_y], dim=-1)

    return agent_out, lane_out


def diff_unnorm_scene3d(agent_norm, lane_norm, cfg, mode="centered"):
    """
    Differentiable 3D unnormalization: [-1,1] → metric coordinates.
    For agent states [x, y, z, vel, cos, sin, length, width, height].
    """
    fov = cfg.fov
    fov_z = cfg.fov_z
    if mode == "centered":
        x_min, x_max = -fov / 2, fov / 2
        y_min, y_max = -fov / 2, fov / 2
    elif mode == "positive":
        x_min, x_max = 0, fov
        y_min, y_max = -fov / 2, fov / 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    def scale(t, tmin, tmax):
        return ((t.clamp(-1, 1) + 1) / 2) * (tmax - tmin) + tmin

    x = scale(agent_norm[:, 0], x_min, x_max)
    y = scale(agent_norm[:, 1], y_min, y_max)
    z = scale(agent_norm[:, 2], -fov_z / 2, fov_z / 2)
    vel = scale(agent_norm[:, 3], cfg.min_speed, cfg.max_speed)
    cos_h = agent_norm[:, 4].clamp(-1, 1)
    sin_h = agent_norm[:, 5].clamp(-1, 1)
    length = scale(agent_norm[:, 6], cfg.min_length, cfg.max_length)
    width  = scale(agent_norm[:, 7], cfg.min_width,  cfg.max_width)
    height = scale(agent_norm[:, 8], cfg.min_height, cfg.max_height)

    parts = [x, y, z, vel, cos_h, sin_h, length, width, height]
    if agent_norm.shape[1] > 9:
        parts += [agent_norm[:, 9:]]
    agent_out = torch.cat([p.unsqueeze(-1) if p.ndim == 1 else p for p in parts], dim=-1)

    lane_x = scale(lane_norm[..., 0], x_min, x_max)
    lane_y = scale(lane_norm[..., 1], y_min, y_max)
    lane_out = torch.stack([lane_x, lane_y], dim=-1)

    return agent_out, lane_out



