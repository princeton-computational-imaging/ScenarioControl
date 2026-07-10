import os
import sys
import json
import glob
import hydra
import torch
import pickle
import random
import sys
from tqdm import tqdm
from cfgs.config import CONFIG_PATH
from typing import Any

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
torch.set_printoptions(threshold=100000)
import numpy as np
np.set_printoptions(suppress=True, threshold=sys.maxsize)

from utils.data_container import ScenarioDreamerData
from utils.torch_helpers import from_numpy
from utils.data_helpers import sample_latents, reorder_indices

class NuplanDatasetLDM(Dataset):
    def __init__(self, cfg: Any, split_name: str = "train") -> None:
        """Instantiate a :class:`NuplanDatasetLDM`.

        Parameters
        ----------
        cfg
            Hydra configuration object containing dataset configs (cfg.dataset in global config)
        split_name
            One of ``{"train", "val", "test"}`` selecting which split
            to load from ``cfg.dataset.dataset_path``.
        """
        super(NuplanDatasetLDM, self).__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.dataset_dir = os.path.join(self.cfg.dataset_path, f"{self.split_name}")
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir, exist_ok=True)

        # if set, only load cached scenes whose filename ends in one of these scene-type digits (e.g. '2' or '12')
        self.load_scene_type = self.cfg.get('load_scene_type', None)
        scene_type_glob = f"/*_[{self.load_scene_type}].pkl" if self.load_scene_type is not None else "/*.pkl"
        self.files = sorted(glob.glob(self.dataset_dir + scene_type_glob))
        self.files_base = self.files.copy()

        # every sample is conditioned by default; is_conditioned_flags tracks which directory each file
        # was drawn from (main dataset_dir vs. uncond_dataset_path), drop_condition_flags additionally
        # drops image conditioning (but not the underlying scene) on some already-conditioned samples
        self.is_conditioned_flags = [True] * len(self.files)
        self.is_conditioned_base = self.is_conditioned_flags.copy()
        self.drop_condition_flags = None

        self.load_single_img_cond = self.cfg.get('load_single_img_cond', False)
        self.img_latents_dir = os.path.join(self.cfg.get('img_latents_dir', ''), f"{self.split_name}")
        self.uncond_dataset_path = self.cfg.get('uncond_dataset_path', None)
        if self.load_single_img_cond and self.split_name == 'train' and self.uncond_dataset_path:
            self.uncond_dataset_dir = os.path.join(self.uncond_dataset_path, f"{self.split_name}")
            self.all_uncond_files = sorted(glob.glob(self.uncond_dataset_dir + scene_type_glob))
            self.num_to_add = int(self.cfg.get('uncond_ratio', 0.0) * len(self.files))
            self.refresh_uncond_files()

        self.dset_len = len(self.files)


    def refresh_uncond_files(self):
        """Re-sample the unconditional-dataset mixin and condition-dropout mask for a new training epoch."""
        if not hasattr(self, 'all_uncond_files'):
            return
        num_to_add = min(self.num_to_add, len(self.all_uncond_files))
        sampled_uncond_files = random.sample(self.all_uncond_files, num_to_add)
        self.files = self.files_base + sampled_uncond_files
        self.is_conditioned_flags = self.is_conditioned_base + [False] * len(sampled_uncond_files)
        drop_cond_ratio = self.cfg.get('drop_cond_ratio', 0.0)
        self.drop_condition_flags = np.random.rand(len(self.files)) > drop_cond_ratio
        self.dset_len = len(self.files)

    
    def get_data(self, data, idx, is_conditioned=True, raw_file_name=None):
        """Return a sample for ldm training"""
        idx = data['idx']
        agent_states = data['agent_states']
        road_points = data['road_points']
        lane_mu = data['lane_mu']
        agent_mu = data['agent_mu']
        lane_log_var = data['lane_log_var']
        agent_log_var = data['agent_log_var']
        edge_index_lane_to_lane = data['edge_index_lane_to_lane']
        edge_index_lane_to_agent = data['edge_index_lane_to_agent']
        edge_index_agent_to_agent = data['edge_index_agent_to_agent']
        scene_type = data['scene_type']
        map_id = np.array([data['map_id']], dtype=int)
        num_lanes = lane_mu.shape[0]
        num_agents = agent_mu.shape[0]

        # apply recursive ordering
        agent_mu, agent_log_var, lane_mu, lane_log_var, edge_index_lane_to_lane, agent_partition_mask, lane_partition_mask, agent_fov_mask, lane_fov_mask, _, _ = reorder_indices(
            agent_mu,
            agent_log_var,
            lane_mu,
            lane_log_var,
            edge_index_lane_to_lane,
            agent_states,
            road_points,
            scene_type,
            dataset='nuplan')
        edge_index_lane_to_lane = torch.from_numpy(edge_index_lane_to_lane)

        # sample for ldm training
        d = ScenarioDreamerData()
        d['idx'] = idx
        d['num_lanes'] = num_lanes
        d['num_agents'] = num_agents
        d['lg_type'] = scene_type
        d['map_id'] = from_numpy(map_id)
        d['agent'].x = from_numpy(agent_mu)
        d['lane'].x = from_numpy(lane_mu)
        d['agent'].partition_mask = from_numpy(agent_partition_mask)
        d['lane'].partition_mask = from_numpy(lane_partition_mask)
        d['agent'].fov_mask = from_numpy(agent_fov_mask)
        d['agent'].log_var = from_numpy(agent_log_var)
        d['lane'].log_var = from_numpy(lane_log_var)
        d['agent'].latents, d['lane'].latents = sample_latents(
            d,
            self.cfg.agent_latents_mean,
            self.cfg.agent_latents_std,
            self.cfg.lane_latents_mean,
            self.cfg.lane_latents_std,
            normalize=True) # sample normalized latents for training

        d['lane', 'to', 'lane'].edge_index = from_numpy(edge_index_lane_to_lane)
        d['agent', 'to', 'agent'].edge_index = from_numpy(edge_index_agent_to_agent)
        d['lane', 'to', 'agent'].edge_index = from_numpy(edge_index_lane_to_agent)

        if self.load_single_img_cond:
            if is_conditioned:
                # the DINO+depth features are extracted once per front-camera frame (scene-type suffix "_0"),
                # so scene-type 1/2 variants of the same frame share the same cached features
                img_latents_name = f'{raw_file_name}_dino_depths.npz'
                img_latents_name = img_latents_name.replace("_1_dino_depths.npz", "_0_dino_depths.npz")
                img_latents_name = img_latents_name.replace("_2_dino_depths.npz", "_0_dino_depths.npz")
                img_latents_path = os.path.join(self.img_latents_dir, img_latents_name)

                img_data = np.load(img_latents_path, allow_pickle=False)
                dino_feats = torch.from_numpy(img_data["dino_feats"])
                # raw "depths" is (1, H, W) -- the leading dim becomes the batch dim once PyG concatenates
                # samples along dim 0, so the channel dim (size 1) must be inserted at position 1, not 0
                depth_map = torch.nan_to_num(torch.from_numpy(img_data["depths"]).unsqueeze(1))

                # per-sample local min-max normalization
                depth_min = torch.amin(depth_map, dim=(-2, -1), keepdim=True)
                depth_max = torch.amax(depth_map, dim=(-2, -1), keepdim=True)
                depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

                d['dino_feats'] = dino_feats
                d['depth_map'] = depth_map
            else:
                d['dino_feats'] = torch.zeros(torch.Size(self.cfg.dino_feats_shape))
                d['depth_map'] = torch.zeros(torch.Size(self.cfg.depth_map_shape))

        return d


    def get(self, idx: int):
        is_conditioned = self.is_conditioned_flags[idx]
        raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
        raw_path = os.path.join(self.dataset_dir if is_conditioned else self.uncond_dataset_dir, f'{raw_file_name}.pkl')
        with open(raw_path, 'rb') as f:
            data = pickle.load(f)

        # drop_condition_flags only ever downgrades an already-conditioned sample to unconditioned
        # (for image-loading purposes); it never affects which underlying scene/geometry is loaded
        if is_conditioned and self.drop_condition_flags is not None:
            is_conditioned = bool(self.drop_condition_flags[idx])

        d = self.get_data(data, idx, is_conditioned=is_conditioned, raw_file_name=raw_file_name)

        return d

    
    def len(self):
        return self.dset_len

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    cfg = cfg.ldm
    dset = NuplanDatasetLDM(cfg.dataset, split_name='train')

    print(cfg.dataset.dataset_path)
    
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

    print(len(dset))

    if not os.path.exists(cfg.dataset.latent_stats_path):
        cfg.dataset.agent_latents_mean = 0.0
        cfg.dataset.agent_latents_std = 1.0
        cfg.dataset.lane_latents_mean = 0.0
        cfg.dataset.lane_latents_std = 1.0
    
    dloader = DataLoader(dset, 
               batch_size=1024, 
               shuffle=True, 
               num_workers=0,
               pin_memory=True,
               drop_last=True)

    agent_latents_all = []
    lane_latents_all = []
    for i, d in enumerate(tqdm(dloader)):
        agent_latents, lane_latents = sample_latents(
            d, 
            cfg.dataset.agent_latents_mean,
            cfg.dataset.agent_latents_std,
            cfg.dataset.lane_latents_mean,
            cfg.dataset.lane_latents_std,
            normalize=False)
        
        agent_latents_all.append(agent_latents)
        lane_latents_all.append(lane_latents)

        if i == 5:
            break
    
    agent_latents_all = torch.cat(agent_latents_all, dim=0)
    lane_latents_all = torch.cat(lane_latents_all, dim=0)

    print(agent_latents_all.mean(), agent_latents_all.std())
    print(lane_latents_all.mean(), lane_latents_all.std())



if __name__ == '__main__':
    main()