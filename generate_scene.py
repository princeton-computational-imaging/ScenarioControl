"""Generate driving scenes from one front-camera image or one BEV text prompt. No dataset is needed.

Image-conditioned: the road ahead of the camera (scene type 2), conditioned on DINOv3 patch features and
a ZoeDepth map computed from the image exactly as data_processing/nuplan/extract_dino_depth_features.py
and resize_dino_depth_features.py compute them for nuPlan frames:
    python generate_scene.py generate_scene.image=front.jpg ldm.eval.run_name=my_scenes

Prompt-conditioned: the scene around the ego (scene type 0), conditioned on a BEV layout description
written like the training captions ("ROAD & CONTROL: ... AGENTS: ..."):
    python generate_scene.py "generate_scene.prompt='ROAD & CONTROL: ... AGENTS: ...'" ldm.eval.run_name=my_scenes
(quote the prompt for Hydra as shown when it contains commas)

Writes generate_scene.num_samples scenes to <ldm.eval.save_dir>/<ldm.eval.run_name>/initial_scene_samples/
scene_<i>.pkl, in the same format as test.py (next step: data_processing/postprocess_simulation_environments.py),
and a BEV plot of each to <ldm.eval.save_dir>/<ldm.eval.run_name>/viz/scene_<i>.png. The image mode also saves
the camera frame it conditioned on (centre-cropped to 16:9, 1920x1080) there as first_frame.png. Each sample's
number of lanes and agents is drawn from the nuPlan prior of its city (generate_scene.map_id).
"""
import glob
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")  # environment.yml has tensorflow (Waymo tools); keep transformers from importing it

import cv2
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from PIL import Image

from cfgs.config import CONFIG_PATH
from datasets.nuplan.dataset_ldm_nuplan_init import image_conditioning
from test import test_ldm
from utils.data_container import ScenarioDreamerData
from utils.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite

REGULAR, FRONT_ONLY = 0, 2  # scene types (lg_type): 64x64 m around the ego / 64x64 m ahead of the ego
CAMERA_SIZE = (1920, 1080)  # nuPlan CAM_F0 (width, height), the resolution the image features were extracted at


def to_camera_frame(path):
    """Load an RGB image, centre-crop it to 16:9 and resize it to CAM_F0 size (a no-op for nuPlan frames)."""
    img = np.array(Image.open(path).convert("RGB"))  # loaded as in extract_dino_depth_features.py
    h, w = img.shape[:2]
    cw, ch = min(w, h * 16 // 9), min(h, w * 9 // 16)
    img = img[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
    if img.shape[1::-1] != CAMERA_SIZE:
        img = cv2.resize(img, CAMERA_SIZE, interpolation=cv2.INTER_AREA)
    return img


def image_features(img, depth_hw):
    """RGB uint8 image -> (dino_feats (1, 384, 14, 14), depths (1, H, W)) as cached on disk for nuPlan."""
    from data_processing.nuplan.extract_dino_depth_features import DinoDepthExtractor
    extractor = DinoDepthExtractor(resize_depth=True)  # the extraction script's defaults
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)[None, None]  # [B=1, N=1, 3, H, W] in [0, 1]
    depths, feats = extractor(x.to(extractor.backbone.device))
    dino_feats, depths = feats[0].float().cpu().numpy(), depths[0].float().cpu().numpy()
    del extractor
    torch.cuda.empty_cache()
    if depths.shape[-2:] != tuple(depth_hw):  # resize_dino_depth_features.py
        depths = cv2.resize(depths[0], (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_LINEAR)[None]
    return dino_feats, depths


def make_samples(cfg, scene_type, prob_matrix, dino_feats=None, depths=None, prompt=None):
    """Conditioning-only LDM inputs, laid out like datasets.nuplan.dataset_ldm_nuplan_init.NuplanDatasetLDMInit."""
    g = cfg.generate_scene
    max_agents = cfg.ldm.dataset.max_num_agents
    samples = []
    for i in range(g.num_samples):
        map_id = g.map_id if g.map_id is not None else i % cfg.ldm.dataset.num_map_ids
        # +1 because there is an index for "no agents" and "no lanes" (as in ScenarioControlLDM._initialize_pyg_dset)
        probs = prob_matrix[map_id]  # [num_lanes + 1, num_agents + 1]
        if g.num_agents is not None:  # P(num_lanes | num_agents)
            probs = probs * (torch.arange(probs.shape[1]) == g.num_agents)
            assert probs.sum() > 0, f"num_agents={g.num_agents} never occurs in nuPlan city {map_id}; try another value or map_id"
        folded = torch.multinomial(probs.reshape(1, -1), 1).item()
        num_lanes, num_agents = folded // (max_agents + 1), folded % (max_agents + 1)
        assert num_lanes > 0 and num_agents > 0, "prior sampled a scene with no lanes or no agents"

        d = ScenarioDreamerData()
        d['idx'] = i
        d['raw_file_name'] = f'scene_{i}'
        d['num_lanes'] = num_lanes
        d['num_agents'] = num_agents
        d['lg_type'] = scene_type
        d['map_id'] = torch.tensor([map_id])
        # placeholders for the diffusion sampler to fill in
        d['lane'].x = torch.empty((num_lanes, cfg.ldm.model.lane_latent_dim))
        d['agent'].x = torch.empty((num_agents, cfg.ldm.model.agent_latent_dim))
        d['lane', 'to', 'lane'].edge_index = get_edge_index_complete_graph(num_lanes)
        d['agent', 'to', 'agent'].edge_index = get_edge_index_complete_graph(num_agents)
        d['lane', 'to', 'agent'].edge_index = get_edge_index_bipartite(num_lanes, num_agents)
        if dino_feats is not None:
            d['dino_feats'], d['depth_map'] = image_conditioning(dino_feats, depths)
        d['text'] = prompt
        d['text_embeds'] = None
        if prompt is not None:
            d['neg_text'] = cfg.ldm.dataset.neg_text
        samples.append(d)
    return samples


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config3dtemp")
def main(cfg):
    g = cfg.generate_scene
    if (g.image is None) == (g.prompt is None):
        raise ValueError("set exactly one of generate_scene.image=<path> or generate_scene.prompt=<text>")
    use_image = g.image is not None
    scene_type = g.scene_type if g.scene_type is not None else (FRONT_ONLY if use_image else REGULAR)

    OmegaConf.set_struct(cfg, False)
    # the image- and prompt-conditioned LDMs (see the README, Inference)
    cfg.ldm.model.img_conditioning = use_image
    cfg.ldm.model.text_conditioning = not use_image
    cfg.ldm.model.decode_in_training = True
    cfg.ldm.dataset.use_cached_text_embeds = False
    cfg_ae, ldm = cfg.ae, cfg.ldm
    ldm.dataset_name = cfg_ae.dataset_name = cfg.dataset_name.name
    ldm.model_name = 'ldm_cond'
    ldm.dimension = cfg.dimension
    ldm.ckpt_path = g.img_cond_ckpt if use_image else g.prompt_cond_ckpt
    save_dir = os.path.join(ldm.eval.save_dir, ldm.eval.run_name)
    ldm.eval.mode = 'initial_scene'
    ldm.eval.cache_samples = True
    ldm.eval.visualize = True
    ldm.eval.viz_dir = os.path.join(save_dir, 'viz')
    pl.seed_everything(ldm.eval.seed, workers=True)

    os.makedirs(save_dir, exist_ok=True)
    dino_feats = depths = None
    if use_image:
        img = to_camera_frame(g.image)
        Image.fromarray(img).save(os.path.join(save_dir, 'first_frame.png'))
        dino_feats, depths = image_features(img, ldm.dataset.depth_map_shape[-2:])

    prob_matrix = torch.load(g.prob_matrix_front if scene_type == FRONT_ONLY else g.prob_matrix_regular)
    samples = make_samples(cfg, scene_type, prob_matrix, dino_feats, depths, None if use_image else g.prompt)
    ldm.datamodule.test_batch_size = len(samples)  # one batch, so BEV plot k is scene_k
    test_ldm(ldm, cfg_ae, save_dir, test_set=samples)

    for i in range(len(samples)):
        for f in glob.glob(os.path.join(ldm.eval.viz_dir, f'*_batch_0_sample_{i}_pred_*.png')):
            os.replace(f, os.path.join(ldm.eval.viz_dir, f'scene_{i}.png'))
    print(f"Wrote {len(samples)} scenes to {os.path.join(save_dir, 'initial_scene_samples')}")


if __name__ == '__main__':
    main()
