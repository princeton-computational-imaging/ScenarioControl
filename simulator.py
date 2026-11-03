# Adapted from Scenario Dreamer (https://github.com/princeton-computational-imaging/scenario-dreamer).
# Changes: CtRL-Sim agents only (no RL planner or log replay), agents carry z and height,
# optional agent-removal flags, and per-frame state export for utils/sequence_saver.py.
import glob
import os
import pickle
import copy
import numpy as np
import torch
import torch.nn.functional as F

from datasets.waymo.dataset_ctrl_sim import CtRLSimDataset
from utils.sim_helpers import (
    ego_completed_route,
    ego_collided,
    ego_off_route,
    ego_progress,
    normalize_route
)
from utils.geometry import normalize_agents, apply_se2_transform
from utils.lane_graph_helpers import resample_lanes_with_mask
from utils.k_disks_helpers import inverse_k_disks, forward_k_disks
from utils.collision_helpers import compute_collision_states_one_scene
from utils.torch_helpers import from_numpy
from utils.data_container import CtRLSimData
from utils.data_helpers import add_batch_dim, modify_agent_states
from utils.viz import render_state
from models.ctrl_sim import CtRLSim

MAX_RTG_VAL = 349


class Simulator:
    """ Simple closed-loop simulator: a planner drives the ego (last agent) through a
    simulation environment (.pkl) while CtRL-Sim controls the other agents.
    """
    def __init__(self, cfg):
        """ Initialize simulator."""
        self.cfg = cfg
        self.steps = self.cfg.sim.steps
        self.dt = self.cfg.sim.dt
        self.dataset_path = self.cfg.sim.dataset_path
        self.test_files = sorted(glob.glob(os.path.join(self.dataset_path, '*.pkl')))
        self.num_test_scenarios = len(self.test_files)

        self.ctrl_sim_dset = CtRLSimDataset(self.cfg.ctrl_sim.dataset, split_name='val')
        self.behaviour_model = CtRLSimBehaviourModel(
            model_path=self.cfg.sim.behaviour_model.model_path,
            model=CtRLSim.load_from_checkpoint(self.cfg.sim.behaviour_model.model_path).to('cuda'),
            dset=self.ctrl_sim_dset,
            use_rtg=self.cfg.sim.behaviour_model.use_rtg,
            predict_rtgs=self.cfg.sim.behaviour_model.predict_rtgs,
            action_temperature=self.cfg.sim.behaviour_model.action_temperature,
            tilt=self.cfg.sim.behaviour_model.tilt,
            steps=self.steps,
            offroad_threshold=self.cfg.sim.agent_offroad_threshold,
            remove_colliding=self.cfg.sim.remove_colliding_agents,
            remove_offroad=self.cfg.sim.remove_offroad_agents
        )
        # tracks state of all objects during simulation
        self.data_dict = {}


    def load_initial_scene(self, i):
        """ Load initial configurations of scenario (map and initial state) given index."""
        with open(self.test_files[i], 'rb') as f:
            scenario_dict = pickle.load(f)
        # 8-column agents (no z, height) -> pad to [x, y, vx, vy, theta, l, w, exists, z, height]
        agents = scenario_dict['agents']
        if agents.shape[-1] == 8:
            scenario_dict['agents'] = np.concatenate([agents, np.zeros((*agents.shape[:-1], 2))], axis=-1)
        return scenario_dict
    

    def _find_invalid_new_agents(
            self, 
            next_states, 
            newly_added_agent_mask, 
            still_existing_agent_mask,
            dist_gap_s=5.0,
            heading_threshold=np.pi/6,
            dist_threshold=2.0):
        """ Find newly added agents that are invalid due to 
        being at edge of FOV and heading outwards. Such agents
        would immediately leave the scene again, so we remove them.
        Also remove newly added agents that violate time gap."""
        normalized_next_states = normalize_agents(
            next_states[:, None], 
            self.local_frame
        )
        lanes, lanes_mask = self.ctrl_sim_dset.get_normalized_lanes_in_fov(
            self.data_dict['lanes'], 
            self.local_frame
        )
        lanes_resampled = resample_lanes_with_mask(
            lanes, 
            lanes_mask, 
            num_points=100
        )
        dist_to_lanes = np.linalg.norm(
            normalized_next_states[:, None, :, :2] 
            - lanes_resampled[None], axis=-1
        ).min(2)
        closest_lane_idxs = np.argmin(dist_to_lanes, axis=-1)

        new_agent_idxs_to_remove = []
        newly_added_agent_idxs = np.where(newly_added_agent_mask)[0]
        for new_agent_idx in newly_added_agent_idxs:
            heading = normalized_next_states[new_agent_idx, 0, 4]
            if (np.abs(heading - np.pi/2) < heading_threshold
                and (normalized_next_states[new_agent_idx, 0, 1]
                     - self.cfg.ctrl_sim.dataset.fov) < dist_threshold):
                new_agent_idxs_to_remove.append(new_agent_idx)
                continue
            
            closest_lane = closest_lane_idxs[new_agent_idx]
            closest_lane_mask = closest_lane_idxs == closest_lane
            agent_in_same_lane_mask = np.logical_and(
                closest_lane_mask,
                still_existing_agent_mask
            )
            if not agent_in_same_lane_mask.sum():
                continue

            dist_to_agent_in_same_lane = np.linalg.norm(
                normalized_next_states[new_agent_idx, :, :2] 
                - normalized_next_states[agent_in_same_lane_mask][:, 0, :2], 
                axis=-1)
            closest_agent_idx = np.where(
                agent_in_same_lane_mask
            )[0][np.argmin(dist_to_agent_in_same_lane)]
            dist_gap = np.linalg.norm(
                normalized_next_states[closest_agent_idx, 0, 2:4]) * dist_gap_s
            dist_to_closest_agent = np.linalg.norm(
                normalized_next_states[new_agent_idx, 0, :2] 
                - normalized_next_states[closest_agent_idx, 0, :2])

            if dist_to_closest_agent < dist_gap:
                new_agent_idxs_to_remove.append(new_agent_idx)
        return new_agent_idxs_to_remove
    

    def step(self, action):
        """ Advance the simulation by one step given the ego action [next_x, next_y, next_theta, next_speed]."""
        self.t += 1

        old_ego_state = copy.deepcopy(self.ego_state)
        next_x, next_y, next_theta, next_speed = action[0], action[1], action[2], action[3]
        self.ego_state = np.array(
            [next_x,
             next_y,
             next_speed * np.cos(next_theta),
             next_speed * np.sin(next_theta),
             next_theta,
             self.ego_state[5],
             self.ego_state[6],
             self.ego_state[7],
             self.ego_state[8],
             self.ego_state[9]],
            dtype=np.float64
        )

        self.local_frame = {
            'center': self.ego_state[:2].copy(),
            'yaw': self.ego_state[4].copy()
        }

        # used by ctrl_sim: find ego action closest to the GT deltas
        inverse_ego_action = inverse_k_disks(old_ego_state, self.ego_state, self.ctrl_sim_dset.V)
        
        self.data_dict['ego_action'].append(inverse_ego_action)
        # always set ego rtg to highest possible value (as that's what done during training)
        self.data_dict['ego_rtg'].append(np.array([MAX_RTG_VAL], dtype=np.float64)[None, :])
        self.data_dict = self.behaviour_model.step(self.data_dict)

        # apply forward model to get the next states (only for active agents)
        next_states = forward_k_disks(
            states=self.data_dict['agent'][-1], 
            actions=self.data_dict['agent_next_action'], 
            vocab=self.ctrl_sim_dset.V, 
            delta_t=self.dt, 
            exists=self.agent_active
        )
        
        # update last active positions for active agents
        self.last_active_agent_position[self.agent_active] = next_states[self.agent_active]
        # for the non-active agents, next state is set to most recent active state
        next_states[~self.agent_active] = self.last_active_agent_position[~self.agent_active]
        
        agent_mask = self.ctrl_sim_dset.get_agent_mask(
            copy.deepcopy(next_states[:, None, :self.ctrl_sim_dset.HEAD_IDX+1]),
            self.local_frame)[:, 0]
        if self.cfg.sim.keep_agents_leaving_fov:
            # scene membership only: the CtRL-Sim context still uses the trained FOV
            agent_mask = np.logical_or(agent_mask, self.agent_active)

        # newly added agents:
        # not active previously (self.agent_active set to 0) 
        # in the simulation radius (agent_mask set to 1)
        # have not yet previously left scene (once left, cannot re-enter)
        newly_added_agent_mask = np.logical_and(
            np.logical_and(
                ~self.agent_active, 
                agent_mask
            ),
            ~self.left_scene
        )
        still_existing_agent_mask = np.logical_and(
            self.agent_active,
            agent_mask
        )

        if newly_added_agent_mask.sum():
            new_agent_idxs_to_remove = self._find_invalid_new_agents(
                next_states, 
                newly_added_agent_mask, 
                still_existing_agent_mask,
            )
            # remove new vehicle from scene if it doesn't respect time gap
            for agent_idx in new_agent_idxs_to_remove:
                self.left_scene[agent_idx] = True
        
        self.left_scene = np.logical_or(
            self.left_scene,
            (self.agent_active.astype(int) - agent_mask.astype(int)) == 1
        )
        
        # activated agents are those 
        # - in the FOV 
        # - have not previously left the scene
        self.agent_active = agent_mask * ~self.left_scene

        # update the data dictionary agent information
        self.data_dict['agent_active'] = copy.deepcopy(self.agent_active)
        self.data_dict['agent'].append(next_states)
        self.data_dict['agent_action'].append(self.data_dict['agent_next_action'])
        self.data_dict['agent_rtg'].append(self.data_dict['agent_next_rtg'])
        
        # update the data dictionary ego information
        self.data_dict['ego'].append(self.ego_state[None, :])
        
        terminated = False
        completed_route = ego_completed_route(
            self.local_frame['center'], 
            self.scenario_dict['route']
        )
        collided = ego_collided(
            self.ego_state, 
            self.data_dict['agent'][-1][self.agent_active]
        ) 
        off_route = ego_off_route(
            self.local_frame['center'], 
            self.scenario_dict['route'],
        )
        
        if (collided or off_route or completed_route 
            or self.t == self.cfg.sim.steps):
            # handle case where off route simply 
            # because you went past the endpoint of the route
            if completed_route:
                off_route = False 
                collided = False
            
            progress = ego_progress(
                self.local_frame['center'], 
                self.scenario_dict['route']
            )
            terminated = True
            info = {
                'collision': collided,
                'off_route': off_route,
                'completed': completed_route,
                'progress': progress
            }
        else:
            info = {}

        # remove offroad / collided agents from scene
        invalid_agents = self.behaviour_model.update_running_statistics(
            self.data_dict, 
            terminated
        )
        invalid_agent_idxs = np.where(invalid_agents)[0]
        if len(invalid_agent_idxs):
            for idx in invalid_agent_idxs:
                self.left_scene[idx] = True 
                self.agent_active[idx] = True
            self.data_dict['agent_active'] = copy.deepcopy(self.agent_active)

        self.current_state = self._get_observation()
        self._update_viz_state()
        
        return self.current_state, terminated, info
    

    def _get_observation(self):
        """ Get agent observation array (agent states + active flag, ego last) for current time step."""
        current_agent_states = np.concatenate(
            [self.data_dict['agent'][-1],
            np.expand_dims(copy.deepcopy(
                self.agent_active
            ), axis=1)
            ], axis=1
        )
        ego_state = np.concatenate(
            [self.ego_state,
             np.ones(1, dtype=np.float64)])

        return np.concatenate([
            current_agent_states,
            np.expand_dims(
                ego_state,
                axis=0)
        ])


    def _update_viz_state(self, num_route_points=30):
        """ Update visualization state for current time step."""
        current_agent_states = self.data_dict['agent'][-1]
        current_agent_types = self.data_dict['agent_type'][0]
        agent_active_mask = self.agent_active
        current_agent_states_rel = normalize_agents(
            current_agent_states[:, None], 
            normalize_dict=self.local_frame
        )[:, 0]
        
        lanes, lanes_mask = self.ctrl_sim_dset.get_normalized_lanes_in_fov(
            self.scenario_dict['lanes'], 
            normalize_dict=self.local_frame
        )
        lanes[~lanes_mask] = 0.0

        route = normalize_route(
            self.scenario_dict['route'], 
            normalize_dict=self.local_frame
        )
        dist_to_route = np.linalg.norm(route, axis=-1)
        route_start = np.argmin(dist_to_route)
        route = route[route_start:route_start+num_route_points]

        self.viz_state = {
            'route': route,
            'agent_states': current_agent_states_rel,
            'agent_types': current_agent_types,
            'agent_active': agent_active_mask,
            'lanes': lanes,
            'lanes_mask': lanes_mask
        }


    def initialize_data_dict(self):
        """ Initialize data dictionary for simulation."""
        data_dict = {}

        ego = self.ego_state[None, :]
        ego_type = np.zeros((1,5), dtype=np.float64)
        ego_type[0, 1] = 1

        agents = self.scenario_dict['agents'][:, 0]
        agent_types = self.scenario_dict['agent_types']

        data_dict['agent'] = [agents]
        data_dict['agent_type'] = [agent_types]
        data_dict['agent_action'] = []
        data_dict['agent_rtg'] = []
        data_dict['agent_next_action'] = []
        data_dict['agent_next_rtg'] = []

        data_dict['ego'] = [ego]
        data_dict['ego_type'] = [ego_type]
        data_dict['ego_action'] = []
        data_dict['ego_rtg'] = []
        # no ego next action because behaviour model does not predict that
        data_dict['ego_next_rtg'] = []

        # as ctrl-sim needs to process the lanes
        data_dict['lanes'] = self.scenario_dict['lanes']
        # which agents are actively being simulated at the current timestep
        data_dict['agent_active'] = copy.deepcopy(self.agent_active)

        self.data_dict = data_dict

        invalid_agents = self.behaviour_model.update_running_statistics(self.data_dict)
        invalid_agent_idxs = np.where(invalid_agents)[0]
        if len(invalid_agent_idxs):
            for idx in invalid_agent_idxs:
                self.left_scene[idx] = True 
                self.agent_active[idx] = True
            self.data_dict['agent_active'] = copy.deepcopy(self.agent_active)


    def reset(self, i):
        """ Reset the environment for a new scenario given index."""
        self.t = 0
        self.scenario_dict = self.load_initial_scene(i)

        self.ego_trajectory = self.scenario_dict['agents'][-1]
        # current state of the ego
        self.ego_state = self.ego_trajectory[0]

        # Store camera info and initial ego_state_og for sequence saving
        self.cam_infos = self.scenario_dict.get('cam_infos', None)
        self.ego_state_og_initial = self.scenario_dict.get('ego_state_og', None)

        # non-ego agents
        self.scenario_dict['agents'] = self.scenario_dict['agents'][:-1]
        self.scenario_dict['agent_types'] = self.scenario_dict['agent_types'][:-1]

        # initialize CtRL-Sim behaviour model
        self.behaviour_model.reset(
            len(self.scenario_dict['agents']) + 1) # +1 to account for the ego

        # ego-centric simulation
        self.local_frame = {
            'center': self.ego_trajectory[0, :2].copy(),
            'yaw': self.ego_trajectory[0, 4].copy()
        }

        # Find agents in FOV
        agent_mask = self.ctrl_sim_dset.get_agent_mask(
            copy.deepcopy(self.scenario_dict['agents'][:, :, :self.ctrl_sim_dset.HEAD_IDX+1]), 
            self.local_frame
        )
        # tells which of the non-ego agents are active
        # and thus get added to context + rendered in visualization
        self.agent_active = agent_mask[:, 0]
        self.left_scene = np.zeros_like(self.agent_active).astype(bool)

        # we initialize all agents to be most "recently activated" at the first timestep
        self.last_active_agent_position = self.scenario_dict['agents'][:, 0]

        # Initialize data dictionary to track simulation state
        self.initialize_data_dict()

        self.current_state = self._get_observation()
        self._update_viz_state()

        return self.current_state
    

    def render_state(self, name, movie_path):
        """ Render the current state of the simulation."""
        agent_states = (
            self.viz_state['agent_states']
            [self.viz_state['agent_active']])
        
        ego_state = normalize_agents(
            self.ego_state[None, None, :], 
            normalize_dict=self.local_frame
        )[:, 0]
        states = np.concatenate(
            [agent_states, ego_state]
            , axis=0)

        agent_types = (
            self.viz_state['agent_types']
            [self.viz_state['agent_active']])
        agent_types = np.concatenate(
            [agent_types, 
             np.array(
                 [0,1,0,0,0], dtype=int
             )[None, :]
            ], axis=0)

        route = self.viz_state['route']
        lanes = self.viz_state['lanes']
        lanes_mask = self.viz_state['lanes_mask']
        
        render_state(
            states,
            agent_types,
            route,
            lanes,
            lanes_mask,
            self.t,
            name,
            movie_path,
            lightweight=self.cfg.sim.lightweight
        )

    def get_current_ego_state_og(self):
        """
        Compute current ego_state_og from current ego_state.

        ego_state_og format: [translation, rotation_quat, dimensions, heading]
        - translation: numpy array (3,) - [x_global, y_global, z_global]
        - rotation_quat: numpy array (4,) - [qw, qx, qy, qz] quaternion
        - dimensions: numpy array (3,) - [length, width, height]
        - heading: float - heading angle in radians (global frame)

        Returns:
            list: [translation, rotation_quat, dimensions, heading] or None if not available
        """
        if self.ego_state_og_initial is None:
            return None

        # ego_state format: [x, y, vx, vy, theta, length, width, exists, ...]
        # NOTE: ego_state positions are in LOCAL/ego-centric coordinates (start at 0,0)
        # We need to transform to GLOBAL coordinates
        x_local, y_local = self.ego_state[0], self.ego_state[1]
        theta = self.ego_state[4]  # heading in radians
        length, width = self.ego_state[5], self.ego_state[6]

        # Get initial global position and orientation
        x_initial_global = self.ego_state_og_initial[0][0]
        y_initial_global = self.ego_state_og_initial[0][1]
        z = self.ego_state_og_initial[0][2] if len(self.ego_state_og_initial[0]) > 2 else 0.0
        height = self.ego_state_og_initial[2][2] if len(self.ego_state_og_initial[2]) > 2 else width

        # Use initial GLOBAL heading for position transformation
        # This rotates local coordinates (where ego starts at origin facing +X)
        # to global coordinates (where ego faces its true global direction)
        # NOTE: some scenarios store heading as a shape-(1,) array; coerce to a
        # Python float so x_global/y_global stay scalars and the np.array below
        # (ego_translation) is homogeneous.
        initial_theta = float(np.asarray(self.ego_state_og_initial[3]).reshape(-1)[0])

        # Transform local position to global using initial global heading, then translate
        cos_theta_init = np.cos(initial_theta)
        sin_theta_init = np.sin(initial_theta)
        x_global = x_initial_global + (x_local * cos_theta_init - y_local * sin_theta_init)
        y_global = y_initial_global + (x_local * sin_theta_init + y_local * cos_theta_init)

        # Create updated translation in global coordinates
        ego_translation = np.array([x_global, y_global, z], dtype=np.float64)

        # For rotation: agents are in local frame with local headings
        # The initial global heading is stored in ego_state_og_initial[3]
        # The initial local heading (at t=0) is stored in local_frame['yaw'] (from agents[-1,0,4])
        # Current local heading is theta (from ego_state[4])
        # Change in heading = current_local - initial_local
        initial_local_heading = self.ego_trajectory[0, 4]  # Initial local heading from agents
        delta_heading = theta - initial_local_heading

        # Global heading = initial global heading + change
        initial_global_heading = float(np.asarray(self.ego_state_og_initial[3]).reshape(-1)[0])
        global_theta = initial_global_heading + delta_heading

        # Convert global heading to quaternion (rotation around z-axis)
        # Note: This is simplified - ideally we'd compose quaternions properly
        qw = np.cos(global_theta / 2)
        qx = 0.0
        qy = 0.0
        qz = np.sin(global_theta / 2)
        ego_rotation = np.array([qw, qx, qy, qz], dtype=np.float64)

        # Dimensions (from current state, though they rarely change)
        ego_dim = np.array([length, width, height], dtype=np.float64)

        return [ego_translation, ego_rotation, ego_dim, global_theta]

    def get_serializable_state(self):
        """
        Get the current simulation state in a format suitable for saving as .pkl file.

        All agent and lane data is transformed to CURRENT-EGO-CENTERED coordinates
        (ego at origin facing +x), matching the per-frame format of the nuPlan training data.

        Returns:
            dict: Contains:
                - agents_ego_centered: numpy array (N_agents, 10) - all agents in ego-centered frame
                - agent_types: numpy array (N_agents, 5) - agent type encodings
                - ego_state_ego_centered: numpy array (10,) - ego state in ego-centered frame (at origin)
                - agent_active: numpy array (N_agents,) - boolean mask for active agents
                - cam_infos: dict - camera information with extrinsics (if available)
                - ego_state_og: list - GLOBAL ego pose [translation, rotation, dims, heading]
                - lanes_ego_centered: numpy array - lanes in ego-centered frame
        """
        # Get current agent states in scenario frame
        agents_local = self.data_dict['agent'][-1]  # (N_agents, 10)

        # Agents in data_dict['agent'] are in the initial scenario frame, as is the current ego
        # (self.ego_state). normalize_agents subtracts 'center' and rotates by -'yaw'.
        offset = 0
        ego_centered_frame = {
            'center': self.ego_state[:2],  # Current ego position in scenario frame
            'yaw': self.ego_state[4]       # Current ego heading in scenario frame
        }

        # Transform agents to CURRENT ego-centered frame
        agents_ego_centered = normalize_agents(
            agents_local[:, None],  # add time dim: (N, 1, 10)
            normalize_dict=ego_centered_frame,
            offset=offset
        )[:, 0]  # remove time dim: (N, 10)

        # Transform lanes to current ego-centered frame
        lanes_local = self.scenario_dict['lanes'].copy()
        yaw = ego_centered_frame['yaw']
        translation = ego_centered_frame['center'][np.newaxis, np.newaxis, :]
        angle_of_rotation = offset + np.sign(-yaw) * np.abs(yaw)
        lanes_ego_centered = apply_se2_transform(
            coordinates=lanes_local[:, :, :2],
            translation=translation,
            yaw=angle_of_rotation
        )

        # Get agent types (static throughout simulation)
        agent_types = self.data_dict['agent_type'][0]  # (N_agents, 5)

        # Get current ego_state_og in GLOBAL coordinates (needed for dimensions)
        ego_state_og = self.get_current_ego_state_og()

        # Ego state in ego-centered frame: at origin with heading=0
        # Format: [x, y, vx, vy, heading, length, width, exists, z, height]
        ego_state_ego_centered = np.zeros(10, dtype=np.float64)
        ego_state_ego_centered[5] = self.ego_state[5]  # length
        ego_state_ego_centered[6] = self.ego_state[6]  # width
        ego_state_ego_centered[7] = 1.0  # exists flag
        # Get z and height from ego_state_og (these are constant for ego)
        if ego_state_og is not None and len(ego_state_og) >= 3:
            ego_state_ego_centered[8] = 0.0  # z position (ego at origin in ego frame)
            ego_state_ego_centered[9] = ego_state_og[2][2]  # height from ego_state_og dimensions
        # x, y = 0 (at origin)
        # vx, vy = 0 (ego doesn't move relative to itself in its own frame)
        # heading = 0 (ego faces forward in its own frame)

        # Get agent active mask (indicates which agents should be rendered)
        agent_active = self.agent_active  # (N_agents,) boolean array

        return {
            'agents_ego_centered': agents_ego_centered,
            'agent_types': agent_types,
            'ego_state_ego_centered': ego_state_ego_centered,
            'agent_active': agent_active,
            'cam_infos': self.cam_infos,
            'ego_state_og': ego_state_og,
            'lanes_ego_centered': lanes_ego_centered
        }


class CtRLSimBehaviourModel:
    # simulator agents carry [.., existence, z, height]; the model only sees the first 8 columns
    NUM_MODEL_STATES = 8  # [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]
    NUM_AGENT_TYPES = 5  # [is_unset, is_vehicle, is_pedestrian, is_cyclist, is_other]
    
    """ Behaviour model wrapper for Ctrl-Sim model used in simulation."""
    def __init__(self, 
                 model_path,
                 model,
                 dset,
                 use_rtg,
                 predict_rtgs,
                 action_temperature,
                 tilt,
                 steps,
                 offroad_threshold=3.0,
                 remove_colliding=True,
                 remove_offroad=True):

        self.model_path = model_path 
        self.model = model 
        self.model.eval()
        self.dset = dset
        self.cfg_model = model.cfg.model
        self.cfg_dataset = model.cfg.dataset
        
        self.steps = steps
        self.use_rtg = use_rtg 
        self.predict_rtgs = predict_rtgs
        self.action_temperature = action_temperature 
        self.tilt = tilt
        self.offroad_threshold = offroad_threshold
        self.remove_colliding = remove_colliding
        self.remove_offroad = remove_offroad
        self.t = 0

        # for aggregating metrics
        self.collision_rate_scenario = []
        self.offroad_rate_scenario = []

        self.has_collided = None
        self.has_offroad = None
        # which agents (since beginning of trajectory) has been activated. Used for computing metrics.
        self.has_activated = None
        self.has_activated_vehicle = None
    
    def update_running_statistics(
            self, 
            data_dict,
            scene_complete=False
        ):
        """ Update collision / offroad statistics and return the agents to remove from the scene."""
        # data_dict: agent: [self.t, A, 10]: [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence, z, height]

        is_vehicle = data_dict['agent_type'][0][:, 1] == 1
        invalid_agents = np.zeros(
            data_dict['agent_active'].shape[0]
        ).astype(bool)
        
        if self.t == 0:
            self.has_collided = np.zeros(
                data_dict['agent_active'].shape[0]
            ).astype(bool)
            self.has_offroad = np.zeros(
                data_dict['agent_active'].shape[0]
            ).astype(bool)
            self.has_activated = data_dict['agent_active']
            self.has_activated_vehicle = np.logical_and(
                data_dict['agent_active'], is_vehicle)
        
        else:
            self.has_activated = np.logical_or(
                self.has_activated,
                data_dict['agent_active']
            )
            
            active_vehicles = np.logical_and(
                data_dict['agent_active'], 
                is_vehicle
            )
            self.has_activated_vehicle = np.logical_or(
                self.has_activated_vehicle,
                active_vehicles
            )
        
        agent_active = data_dict['agent_active']

        # determine which agents (of those currently activated are colliding)
        sim_agents = np.array(data_dict['agent'])[self.t, agent_active]
        if sim_agents.shape[0] > 1:
            agents_colliding = compute_collision_states_one_scene(
                modify_agent_states(sim_agents)
            )
            
            active_agent_idxs = np.where(agent_active == 1)[0]
            colliding_all = np.zeros(len(agent_active)).astype(bool)
            for active_agent_idx, agent_colliding in zip(
                active_agent_idxs, agents_colliding):
                colliding_all[active_agent_idx] = agent_colliding

            # compute the offroad rate for vehicles
            normalize_dict = {  
                'center': data_dict['ego'][self.t][0, :2].copy(),
                'yaw': data_dict['ego'][self.t][0, 4].copy()
            }
            lanes, lanes_mask = self.dset.get_normalized_lanes_in_fov(
                data_dict['lanes'], 
                normalize_dict
            )
            lanes_resampled = resample_lanes_with_mask(
                lanes, 
                lanes_mask, 
                num_points=100
            )
            
            agents_normalized = normalize_agents(
                data_dict['agent'][self.t][:, None], 
                normalize_dict
            )
            min_dist_to_lane = np.linalg.norm(
                lanes_resampled.reshape(-1, 2)[None, :] - 
                agents_normalized[:, :, :2], axis=-1).min(1)
            agents_offroad = min_dist_to_lane > self.offroad_threshold
            agents_offroad[~agent_active] = False
            agents_offroad[~is_vehicle] = False
            offroad_all = agents_offroad

            # remove agents that are colliding
            if self.remove_colliding:
                invalid_agents = np.logical_or(
                    invalid_agents,
                    colliding_all
                )
            # remove agents that are offroad
            if self.remove_offroad:
                invalid_agents = np.logical_or(
                    invalid_agents,
                    offroad_all
                )
            
            self.has_collided = np.logical_or(
                self.has_collided,
                colliding_all
            )
            self.has_offroad = np.logical_or(
                self.has_offroad,
                offroad_all
            )

        if scene_complete:
            if np.sum(self.has_activated) > 0:
                collision_rate = (np.sum(self.has_collided) 
                                  / np.sum(self.has_activated))
            else:
                collision_rate = 0.

            if np.sum(self.has_activated_vehicle) > 0:
                offroad_rate = (np.sum(self.has_offroad) 
                                / np.sum(self.has_activated_vehicle))
            else:
                offroad_rate = 0.
            
            self.collision_rate_scenario.append(collision_rate)  
            self.offroad_rate_scenario.append(offroad_rate)

        return invalid_agents

    
    def compute_metrics(self):
        """ Compute behaviour model metrics after all scenarios have been run."""
        metrics_dict = {
            'collision_rate': np.array(
                self.collision_rate_scenario).mean(),
            'offroad_rate': np.array(
                self.offroad_rate_scenario).mean()
        }
        return metrics_dict, ["{}: {:.6f}".format(k,v) for (k,v) in metrics_dict.items()]

    
    def reset(self, num_agents):
        """ Reset the behaviour model state for a new scenario."""
        self.t = 0
        # Only store 8 elements that the model was trained on
        self.states = np.zeros((num_agents, self.steps, self.NUM_MODEL_STATES), dtype=np.float64)
        self.types = np.zeros((num_agents, self.NUM_AGENT_TYPES), dtype=np.float64)
        self.actions = np.zeros((num_agents, self.steps), dtype=np.float64)
        self.rtgs = np.ones((num_agents, self.steps, self.cfg_model.num_reward_components), dtype=np.float64) * MAX_RTG_VAL


    def update_state(self, data_dict):
        """ Update the internal state of the behaviour model with new data (model sees columns 0-7 only)."""
        # now, EGO is the first index
        self.states[:1, self.t, :] = data_dict['ego'][self.t][:, :self.NUM_MODEL_STATES]
        self.states[1:, self.t, :] = data_dict['agent'][self.t][:, :self.NUM_MODEL_STATES]

        if self.t == 0:
            self.types[:1] = data_dict['ego_type'][0]
            self.types[1:] = data_dict['agent_type'][0]

        # for ego, we use the action from the planner
        # for the other agents, that is what ctrl-sim is for
        self.actions[:1, self.t] = data_dict['ego_action'][self.t]
        self.rtgs[:1, self.t, :] = data_dict['ego_rtg'][self.t]

        # Update previous timestep actions and rtgs for non-ego agents.
        if self.t > 0:
            self.actions[1:, self.t-1] = data_dict['agent_action'][self.t-1]
            if self.predict_rtgs:
                self.rtgs[1:, self.t-1, 0] = data_dict['agent_rtg'][self.t-1]

        # clear out cache for all non-existing agents
        self.states[1:][~data_dict['agent_active']] = 0
    
    
    def get_motion_data(self, data_dict):
        """ Prepare inputs to CtRL-Sim model for forward pass."""
        timesteps = np.arange(
            self.cfg_dataset.train_context_length
        ).astype(int)

        # retrieve relevant context
        if self.t < self.cfg_dataset.train_context_length:
            ag_states = self.states[:, :self.cfg_dataset.train_context_length].copy()
            ag_types = self.types.copy()
            actions = self.actions[:, :self.cfg_dataset.train_context_length].copy()
            rtgs = self.rtgs[:, :self.cfg_dataset.train_context_length, 0].copy()
            rtg_mask = ag_states[:, :, -1]
            timestep_buffer = np.repeat(
                timesteps[np.newaxis, :, np.newaxis], 
                self.cfg_dataset.max_num_agents, 
                0
            )
            normalize_timestep = self.t
        else:
            ag_states = self.states[:,self.t-(
                self.cfg_dataset.train_context_length - 1
                ):self.t+1].copy()
            ag_types = self.types.copy()
            actions = self.actions[:, self.t-(
                self.cfg_dataset.train_context_length - 1
                ):self.t+1].copy()
            rtgs = self.rtgs[:, self.t-(
                self.cfg_dataset.train_context_length - 1
                ):self.t+1, 0].copy()
            rtg_mask = ag_states[:, :, -1]
            timestep_buffer = np.repeat(
                timesteps[np.newaxis, :, np.newaxis], 
                self.cfg_dataset.max_num_agents, 
                0)
            normalize_timestep = self.cfg_dataset.train_context_length - 1

        # ego is index 0 now
        normalize_dict = {
            'center': ag_states[0, normalize_timestep, :2].copy(),
            'yaw': ag_states[0, normalize_timestep, 4].copy()
        }
        
        # filters out observations that are not within the FOV at the normalize_timestep
        agent_mask = self.dset.get_agent_mask(
            copy.deepcopy(ag_states[:, :, :self.dset.HEAD_IDX+1]
        ), normalize_dict)

        # we don't filter out non-moving agents
        moving_agent_mask = np.ones(
            ag_states.shape[0]
        ).astype(bool)
        
        motion_datas = {}
        correspondences = {}
        motion_data_id = 0 
        # vehicle ids in the FOV (ie, that need to be predicted)
        # that have not yet been added to a data buffer for prediction.
        unaccounted_veh_ids = np.where(data_dict['agent_active'] == 1)[0]
        
        while len(unaccounted_veh_ids) > 0:
            (state_buffer, 
             agent_type_buffer, 
             agent_mask_buffer, 
             action_buffer, 
             rtg_buffer, 
             rtg_mask_buffer, 
             _,
             new_origin_agent_idx, 
             correspondence
             ) = self.dset.select_closest_max_num_agents(
                 ag_states, 
                 ag_types, 
                 agent_mask, 
                 actions, 
                 rtgs, 
                 rtg_mask, 
                 moving_agent_mask,
                 origin_agent_idx=0, 
                 timestep=normalize_timestep, 
                 active_agents=unaccounted_veh_ids + 1) # +1 because ego is index 0
            
            # correspondence[i] is the index of the 
            # i'th element in state_buffer in ag_states
            # This is because the ego is always closest 
            # to the ego and we define ego as first position
            assert correspondence[0] == 0
            # This now tells us the mapping to data_dict['agents']
            # as data_dict['agents'] does not include ego
            correspondence -= 1

            assert np.all(
                np.isin(correspondence[1:], 
                np.where(data_dict['agent_active'] == 1
            )[0]))
            
            lanes, lanes_mask = self.dset.get_normalized_lanes_in_fov(
                data_dict['lanes'], 
                normalize_dict
            )
            state_buffer = normalize_agents(
                state_buffer, 
                normalize_dict
            )
            
            # add ego indicator
            is_ego = np.zeros(len(state_buffer))
            is_ego[new_origin_agent_idx] = 1
            is_ego = is_ego.astype(int)
            is_ego = np.tile(is_ego[:, None, None], 
                             (1, self.cfg_dataset.train_context_length, 1))

            # EXIST_IDX still last index
            state_buffer = np.concatenate(
                [state_buffer[:, :, :-1], 
                 is_ego, 
                 state_buffer[:, :, -1:]], axis=-1)

            # filter out agents / lane positions that are not in the FOV
            state_buffer[~agent_mask_buffer.astype(bool)] = 0
            rtg_mask_buffer[~agent_mask_buffer.astype(bool)] = 0
            lanes = np.concatenate(
                [lanes, lanes_mask[:, :, None]]
                , axis=-1)

            motion_data = dict()
            motion_data['idx'] = self.t
            motion_data['agent'] = from_numpy({
                'agent_states': add_batch_dim(state_buffer),
                'agent_types': add_batch_dim(agent_type_buffer), 
                'actions': add_batch_dim(action_buffer),
                'rtgs': add_batch_dim(rtg_buffer[:, :, None]),
                'rtg_mask': add_batch_dim(rtg_mask_buffer[:, :, None]),
                'timesteps': add_batch_dim(timestep_buffer),
                'moving_agent_mask': add_batch_dim(moving_agent_mask)
            })
            motion_data['map'] = from_numpy({
                'road_points': add_batch_dim(lanes),
            })
            motion_data = CtRLSimData(motion_data)
            
            unaccounted_veh_ids = np.setdiff1d(
                unaccounted_veh_ids, 
                correspondence[1:])

            motion_datas[motion_data_id] = motion_data 
            correspondences[motion_data_id] = correspondence 
            motion_data_id += 1

        return motion_datas, correspondences


    def get_tilt_logits(self, tilt):
        """ Get tilted logits for reward-to-go prediction."""
        rtg_bin_values = np.zeros((self.cfg_dataset.rtg_discretization, 1), dtype=np.float64)
        rtg_bin_values[:, 0] = tilt * np.linspace(0, 1, self.cfg_dataset.rtg_discretization, dtype=np.float64)

        return rtg_bin_values
    
    def process_predicted_rtg(
            self, 
            rtg_logits, 
            token_index, 
            data_dict, 
            motion_data, 
            tensor_id, 
            veh_id, 
            is_tilted=False
        ):
        """ Process predicted reward-to-go for a single agent."""
        next_rtg_logits = rtg_logits[0, tensor_id, token_index].reshape(
            self.cfg_dataset.rtg_discretization, 
            self.cfg_model.num_reward_components
        )
        
        if is_tilted:
            tilt_logits = torch.from_numpy(
                self.get_tilt_logits(self.tilt)
            ).cuda()
        else:
            tilt_logits = torch.from_numpy(
                self.get_tilt_logits(0)
            ).cuda()
        
        next_rtg_dis = F.softmax(
            next_rtg_logits[:, 0] 
            + tilt_logits[:, 0], dim=0)
        next_rtg = torch.multinomial(
            next_rtg_dis, 
            1)
        motion_data['agent'].rtgs[0, tensor_id, token_index, 0] = next_rtg.item()
        data_dict['agent_next_rtg'][veh_id] = next_rtg.item()

        return data_dict, motion_data

    
    def predict(self, motion_datas, data_dict, correspondences):
        """ Predict next actions and rtgs for all agents given motion data."""
        if self.t < self.cfg_dataset.train_context_length:
            token_index = self.t 
        else:
            token_index = -1
        
        data_dict['agent_next_action'] = np.zeros(
            len(data_dict['agent'][0]), dtype=np.float64)
        data_dict['agent_next_rtg'] = np.zeros(
            len(data_dict['agent'][0]), dtype=np.float64)
        for motion_data_id in motion_datas:
            motion_data = motion_datas[motion_data_id]
            correspondence = correspondences[motion_data_id]

            motion_data = motion_data.cuda()
            # s --> R
            if self.predict_rtgs:
                preds = self.model(motion_data, eval=True)
                rtg_logits = preds['rtg_preds']
                
                # start from 1, as we don't predict the ego
                for tensor_id, veh_id in enumerate(correspondence):
                    if tensor_id == 0:
                        continue
                    data_dict, motion_data = self.process_predicted_rtg(
                        rtg_logits, 
                        token_index, 
                        data_dict, 
                        motion_data, 
                        tensor_id, 
                        veh_id, 
                        is_tilted=True
                    )
            
            # R --> a
            preds = self.model(motion_data, eval=True)
            # [batch_size=1, num_agents, timesteps, action_dim]
            logits = preds['action_preds']

            # temperature sampling for action prediction
            for tensor_id, veh_id in enumerate(correspondence):
                if tensor_id == 0:
                    continue
                next_action_logits = logits[0, tensor_id, token_index]
                next_action_dis = F.softmax(
                    next_action_logits 
                    / self.action_temperature, dim=0)
                next_action = torch.multinomial(
                    next_action_dis, 1)
                data_dict['agent_next_action'][veh_id] = next_action.item()
        
        return data_dict
    
    
    def step(self, data_dict):
        """ Step function for behaviour model to predict next actions and rtgs."""
        self.update_state(data_dict)
        motion_datas, correspondences = self.get_motion_data(data_dict)
        data_dict = self.predict(motion_datas, data_dict, correspondences)

        self.t += 1
        return data_dict