import os
import hydra
from omegaconf import OmegaConf
from models.scenario_control_autoencoder import ScenarioControlAutoEncoder
from models.scenario_control_ldm import ScenarioControlLDM

import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.strategies import DDPStrategy
from cfgs.config import CONFIG_PATH
from utils.train_helpers import set_latent_stats
import utils.sim_env_helpers as _sim_env_helpers


def _find_last_ckpt(save_dir):
    """ Find the last checkpoint in the save directory."""
    ckpt_path = None
    for file in os.listdir(save_dir):
        if file.endswith('.ckpt') and 'last' in file:
            ckpt_path = os.path.join(save_dir, file)
            print("Loading checkpoint: ", ckpt_path)
            break
    return ckpt_path


def generate_simulation_environments(cfg, cfg_ae, save_dir=None):
    """ Generate simulation environments using the ScenarioControl Latent Diffusion Model.

    This involves 1 step of initial scene generation followed by multiple steps of
    inpainting to extend the scenario until the desired route length is reached.
    Additional rule-based heuristics are applied to ensure scenario validity.
    """
    cfg = set_latent_stats(cfg)

    ckpt_path = cfg.ckpt_path
    if ckpt_path is not None:
        print(f"Overriding checkpoint path to {ckpt_path}")
    else:
        ckpt_path = _find_last_ckpt(save_dir)
    assert ckpt_path is not None, "No checkpoint found in the save directory."

    model = ScenarioControlLDM.load_from_checkpoint(ckpt_path, cfg=cfg, cfg_ae=cfg_ae, strict=False).to('cuda')
    _sim_env_helpers.generate_simulation_environments(model, cfg, save_dir)


def eval_ldm(cfg, cfg_ae, save_dir=None):
    """ Evaluate the ScenarioControl Latent Diffusion Model."""
    cfg = set_latent_stats(cfg)

    if cfg.eval.mode == 'metrics':
        raise NotImplementedError(
            "Metrics computation has not been ported to this repo yet. "
            "Use mode=initial_scene, lane_conditioned, inpainting, or simulation_environments."
        )

    ckpt_path = cfg.ckpt_path
    if ckpt_path is not None:
        print(f"Overriding checkpoint path to {ckpt_path}")
    else:
        ckpt_path = _find_last_ckpt(save_dir)
    assert ckpt_path is not None, "No checkpoint found in the save directory."

    model = ScenarioControlLDM.load_from_checkpoint(ckpt_path, cfg=cfg, cfg_ae=cfg_ae).to('cuda')
    model.generate(
        mode=cfg.eval.mode, # ScenarioControl supports multiple generation modes: initial_scene, lane_conditioned, and inpainting
        num_samples=cfg.eval.num_samples,
        batch_size=cfg.eval.batch_size,
        cache_samples=cfg.eval.cache_samples,
        visualize=cfg.eval.visualize,
        conditioning_path=cfg.eval.conditioning_path,
        cache_dir=os.path.join(save_dir, f'{cfg.eval.mode}_samples'),
        viz_dir=cfg.eval.viz_dir,
        save_wandb=False,
        return_samples=False,
        vis_gt=cfg.eval.get('visualize_gt', False),
    )


def eval_autoencoder(cfg, save_dir=None):
    """ Evaluate the ScenarioControl AutoEncoder model."""
    model = ScenarioControlAutoEncoder(cfg)
    model_summary = ModelSummary(max_depth=-1)

    ckpt_path = _find_last_ckpt(save_dir)
    assert ckpt_path is not None, "No checkpoint found in the save directory."

    tester = pl.Trainer(accelerator='auto',
                         devices=1,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_summary],
                         precision='32-true'
                        )

    tester.test(model, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config3dtemp")
def main(cfg):
    # need to track whether we are evaluating a nuplan or waymo model as
    # nuplan predicts lane types (lane/green light/red light) and waymo does not
    dataset_name = cfg.dataset_name.name
    ckpt_path = cfg.get('ckpt_path', None)  # root-level field; must capture before cfg is reassigned below
    dimension = cfg.get('dimension', '2d')  # root-level field; must capture before cfg is reassigned below
    if 'autoencoder' in cfg.model_name:
        model_name = cfg.model_name
        cfg = cfg.ae
        # not the cleanest solution, but need to track dataset name
        OmegaConf.set_struct(cfg, False)   # unlock to allow setting dataset name
        cfg.dataset_name = dataset_name
        cfg.model_name = model_name
        OmegaConf.set_struct(cfg, True)    # relock
    else:
        model_name = cfg.model_name
        cfg_ae = cfg.ae
        cfg = cfg.ldm
        OmegaConf.set_struct(cfg, False)   # unlock to allow setting dataset name
        OmegaConf.set_struct(cfg_ae, False)
        cfg.dataset_name = dataset_name
        cfg_ae.dataset_name = dataset_name
        cfg.model_name = model_name
        cfg.ckpt_path = ckpt_path
        cfg.dimension = dimension
        OmegaConf.set_struct(cfg, True)    # relock
        OmegaConf.set_struct(cfg_ae, True)

    pl.seed_everything(cfg.eval.seed, workers=True)

    # checkpoints loaded from here
    save_dir = os.path.join(cfg.eval.save_dir, cfg.eval.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"Evaluating ScenarioControl {model_name} trained on {cfg.dataset_name} dataset.")

    if 'autoencoder' in model_name:
        eval_autoencoder(cfg, save_dir)
    elif 'ldm' in model_name:
        if cfg.eval.mode == 'simulation_environments':
            generate_simulation_environments(cfg, cfg_ae, save_dir)
        else:
            eval_ldm(cfg, cfg_ae, save_dir)


if __name__ == '__main__':
    main()
