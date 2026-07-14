import os
import glob
import json
import pickle
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Dataset

from utils.data_container import ScenarioDreamerData
from utils.torch_helpers import from_numpy
from utils.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite
from utils.data_helpers import reparameterize, normalize_latents


class NuplanDatasetLDMInit(Dataset):
    """Seeds conditioned `initial_scene` generation (test.py) from real reference scenes.

    Unlike ``NuplanDatasetLDM`` (used for training), this dataset keeps ``lane.x``/``agent.x`` as empty
    placeholders for the diffusion sampler to fill in, but attaches real structural info (``num_lanes``,
    ``num_agents``, ``map_id``, ``lg_type``) and real conditioning (image and/or text, per
    ``cfg.load_single_img_cond``/``cfg.load_captions``) from the reference scene, rather than sampling
    synthetic counts from a probability matrix.
    """
    def __init__(self, cfg: Any, split_name: str = "test") -> None:
        super(NuplanDatasetLDMInit, self).__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.dataset_dir = os.path.join(self.cfg.dataset_path, f"{self.split_name}")

        load_scene_type = self.cfg.get('load_scene_type', None)
        scene_type_glob = f"/*_[{load_scene_type}].pkl" if load_scene_type is not None else "/*.pkl"
        self.files = sorted(glob.glob(self.dataset_dir + scene_type_glob))

        self.load_single_img_cond = self.cfg.get('load_single_img_cond', False)
        self.img_latents_dir = os.path.join(self.cfg.get('img_latents_dir', ''), f"{self.split_name}")

        self.load_captions = self.cfg.get('load_captions', False)
        self.captions_dir = os.path.join(self.cfg.get('captions_dir', ''), f"{self.split_name}")
        self.use_cached_text_embeds = self.cfg.get('use_cached_text_embeds', False)
        self.text_embeds_dir = os.path.join(self.cfg.get('text_embeds_dir', ''), f"{self.split_name}")

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

        # real ground-truth latents of the reference scene (distinct from the .x placeholders above) --
        # only used when ldm.eval.visualize_gt=true, to decode/visualize the real scene alongside the
        # generated one for qualitative comparison
        agent_latents = reparameterize(from_numpy(agent_mu), from_numpy(data['agent_log_var']))
        lane_latents = reparameterize(from_numpy(lane_mu), from_numpy(data['lane_log_var']))
        agent_latents, lane_latents = normalize_latents(
            agent_latents, lane_latents,
            self.cfg.agent_latents_mean, self.cfg.agent_latents_std,
            self.cfg.lane_latents_mean, self.cfg.lane_latents_std)
        d['agent'].latents = agent_latents
        d['lane'].latents = lane_latents

        # only used to locate the raw conditioning camera image for visualize_gt; 'cam_infos' is only
        # present in the latent cache when it was cached with ae.dataset.load_images=True. Left unset
        # (rather than {}) when load_single_img_cond=False, so convert_batch_to_scenarios's
        # `data.get('cam_infos', None) is not None` check correctly skips it for non-image-conditioned runs.
        if self.load_single_img_cond:
            cam_infos = data.get('cam_infos', {})
            d['cam_infos'] = {k: v for k, v in cam_infos.items() if k == 'CAM_F0'}

        d['lane', 'to', 'lane'].edge_index = get_edge_index_complete_graph(num_lanes)
        d['agent', 'to', 'agent'].edge_index = get_edge_index_complete_graph(num_agents)
        d['lane', 'to', 'agent'].edge_index = get_edge_index_bipartite(num_lanes, num_agents)

        # real reference-scene conditioning (every sample here is conditioned; no CFG dropout at inference)
        if self.load_single_img_cond:
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

        d['text'] = None
        d['text_embeds'] = None
        if self.load_captions:
            # captions are extracted once per raw camera frame (scene-type suffix "_0"), so scene-type
            # 1/2 variants of the same frame share the same caption -- same sharing convention as dino_feats
            caption_name = f'{raw_file_name}_captions.json'
            caption_name = caption_name.replace("_1_captions.json", "_0_captions.json")
            caption_name = caption_name.replace("_2_captions.json", "_0_captions.json")
            caption_path = os.path.join(self.captions_dir, caption_name)

            if os.path.exists(caption_path):
                with open(caption_path, 'r') as f:
                    d['text'] = json.load(f)['BEV']
            else:
                d['text'] = "A realistic driving scene."

            # this dataset is inference-only, so always provide the negative prompt for classifier-free guidance
            d['neg_text'] = self.cfg.get('neg_text', "An unrealistic driving scene.")

        if self.use_cached_text_embeds:
            text_embeds_name = f'{raw_file_name}_captions.pt'
            text_embeds_name = text_embeds_name.replace("_1_captions.pt", "_0_captions.pt")
            text_embeds_name = text_embeds_name.replace("_2_captions.pt", "_0_captions.pt")
            text_embeds_path = os.path.join(self.text_embeds_dir, text_embeds_name)
            if not os.path.exists(text_embeds_path):
                raise FileNotFoundError(f"Text embeds file not found: {text_embeds_path}")
            d['text_embeds'] = torch.load(text_embeds_path)

            neg_text_embeds_path = os.path.join(self.text_embeds_dir, 'negative_prompt.pt')
            if not os.path.exists(neg_text_embeds_path):
                raise FileNotFoundError(f"Negative text embeds file not found: {neg_text_embeds_path}")
            d['neg_text_embeds'] = torch.load(neg_text_embeds_path)

        return d


    def get(self, idx: int):
        raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
        raw_path = os.path.join(self.dataset_dir, f'{raw_file_name}.pkl')
        with open(raw_path, 'rb') as f:
            data = pickle.load(f)

        return self.get_data(data, idx, raw_file_name)


    def len(self):
        return self.dset_len
