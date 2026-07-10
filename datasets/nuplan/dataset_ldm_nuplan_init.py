import os
import glob
import pickle
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Dataset

from utils.data_container import ScenarioDreamerData
from utils.torch_helpers import from_numpy
from utils.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite


class NuplanDatasetLDMInit(Dataset):
    """Seeds image-conditioned `initial_scene` generation (test.py) from real reference scenes.

    Unlike ``NuplanDatasetLDM`` (used for training), this dataset keeps ``lane.x``/``agent.x`` as empty
    placeholders for the diffusion sampler to fill in, but attaches real structural info (``num_lanes``,
    ``num_agents``, ``map_id``, ``lg_type``) and real image conditioning (``dino_feats``, ``depth_map``)
    from the reference scene, rather than sampling synthetic counts from a probability matrix.
    """
    def __init__(self, cfg: Any, split_name: str = "test") -> None:
        super(NuplanDatasetLDMInit, self).__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.dataset_dir = os.path.join(self.cfg.dataset_path, f"{self.split_name}")

        load_scene_type = self.cfg.get('load_scene_type', None)
        scene_type_glob = f"/*_[{load_scene_type}].pkl" if load_scene_type is not None else "/*.pkl"
        self.files = sorted(glob.glob(self.dataset_dir + scene_type_glob))
        self.img_latents_dir = os.path.join(self.cfg.get('img_latents_dir', ''), f"{self.split_name}")
        self.dset_len = len(self.files)


    def get_data(self, data, idx, raw_file_name):
        """Return a real-scene-seeded sample for initial_scene generation."""
        lane_mu = data['lane_mu']
        agent_mu = data['agent_mu']
        num_lanes = lane_mu.shape[0]
        num_agents = agent_mu.shape[0]
        map_id = np.array([data['map_id']], dtype=int)
        scene_type = data['scene_type']

        d = ScenarioDreamerData()
        d['idx'] = idx
        d['raw_file_name'] = raw_file_name
        d['num_lanes'] = num_lanes
        d['num_agents'] = num_agents
        d['lg_type'] = scene_type
        d['map_id'] = from_numpy(map_id)
        d['ego_state_og'] = data['ego_state_og']

        # placeholders for the diffusion sampler to fill in
        d['lane'].x = torch.empty((num_lanes, lane_mu.shape[1]))
        d['agent'].x = torch.empty((num_agents, agent_mu.shape[1]))
        d['lane', 'to', 'lane'].edge_index = get_edge_index_complete_graph(num_lanes)
        d['agent', 'to', 'agent'].edge_index = get_edge_index_complete_graph(num_agents)
        d['lane', 'to', 'agent'].edge_index = get_edge_index_bipartite(num_lanes, num_agents)

        # real reference-scene image conditioning (every sample here is conditioned; no CFG dropout at inference)
        img_latents_name = f'{raw_file_name}_dino_depths.npz'
        img_latents_name = img_latents_name.replace("_1_dino_depths.npz", "_0_dino_depths.npz")
        img_latents_name = img_latents_name.replace("_2_dino_depths.npz", "_0_dino_depths.npz")
        img_latents_path = os.path.join(self.img_latents_dir, img_latents_name)

        img_data = np.load(img_latents_path, allow_pickle=False)
        dino_feats = torch.from_numpy(img_data["dino_feats"])
        # raw "depths" is (1, H, W) -- the leading dim becomes the batch dim once PyG concatenates
        # samples along dim 0, so the channel dim (size 1) must be inserted at position 1, not 0
        depth_map = torch.nan_to_num(torch.from_numpy(img_data["depths"]).unsqueeze(1))

        depth_min = torch.amin(depth_map, dim=(-2, -1), keepdim=True)
        depth_max = torch.amax(depth_map, dim=(-2, -1), keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

        d['dino_feats'] = dino_feats
        d['depth_map'] = depth_map

        return d


    def get(self, idx: int):
        raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
        raw_path = os.path.join(self.dataset_dir, f'{raw_file_name}.pkl')
        with open(raw_path, 'rb') as f:
            data = pickle.load(f)

        return self.get_data(data, idx, raw_file_name)


    def len(self):
        return self.dset_len
