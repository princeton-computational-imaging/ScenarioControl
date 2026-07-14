import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid the fork-after-tokenizer-init warning from DataLoader workers
import hydra
from models.scenario_control_ldm import ScenarioControlLDM
from datasets.nuplan.dataset_ldm_nuplan_init import NuplanDatasetLDMInit

import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from cfgs.config import CONFIG_PATH
from omegaconf import OmegaConf
from utils.train_helpers import set_latent_stats
from torch_geometric.loader import DataLoader


def test_ldm(cfg, cfg_ae, save_dir=None):
    """ Generate image-conditioned samples from the ScenarioControl Latent Diffusion Model,
    seeded from real reference scenes (see datasets.nuplan.dataset_ldm_nuplan_init)."""
    cfg = set_latent_stats(cfg)

    if cfg.dataset_name != "nuplan":
        raise NotImplementedError("test.py currently only supports dataset_name=nuplan.")

    test_set = NuplanDatasetLDMInit(cfg.dataset, split_name="test")
    test_loader = DataLoader(test_set, batch_size=cfg.datamodule.test_batch_size, shuffle=False)

    ckpt_path = cfg.ckpt_path
    if ckpt_path is not None:
        print(f"Overriding checkpoint path to {ckpt_path}")
    else:
        ckpt_path = None
        for file in os.listdir(save_dir):
            if file.endswith('.ckpt') and 'last' in file:
                ckpt_path = os.path.join(save_dir, file)
                print("Loading checkpoint: ", ckpt_path)
                break
    assert ckpt_path is not None, "No checkpoint found in the save directory."

    # strict=False: this may be an older checkpoint predating an architecture change (e.g. before
    # img/text-conditioning layers existed) -- PyTorch Lightning will warn about any missing/unexpected
    # keys so you can judge how much of the loaded model is actually from the checkpoint vs. fresh init
    model = ScenarioControlLDM.load_from_checkpoint(ckpt_path, cfg=cfg, cfg_ae=cfg_ae, map_location='cpu', strict=False)

    trainer = pl.Trainer(accelerator='auto',
                         devices=1,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         precision=cfg.train.precision,
                         logger=None,
                        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    trainer.test(model, dataloaders=test_loader)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config3dtemp")
def main(cfg):
    # need to track whether we are testing a nuplan or waymo model as
    # nuplan predicts lane types (lane/green light/red light) and waymo does not
    dataset_name = cfg.dataset_name.name
    if 'ldm' not in cfg.model_name:
        raise NotImplementedError("test.py only supports LDM generation (model_name containing 'ldm').")

    model_name = cfg.model_name
    ckpt_path = cfg.get('ckpt_path', None)  # root-level field; must capture before cfg is reassigned below
    dimension = cfg.get('dimension', '2d')  # root-level field; must capture before cfg is reassigned below
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

    # checkpoints loaded from here, outputs cached under the same directory
    save_dir = os.path.join(cfg.eval.save_dir, cfg.eval.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"Testing ScenarioControl {model_name} on {cfg.dataset_name} dataset.")
    test_ldm(cfg, cfg_ae, save_dir)


if __name__ == '__main__':
    main()
