import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid the fork-after-tokenizer-init warning from DataLoader workers
import hydra
from models.scenario_control_autoencoder import ScenarioControlAutoEncoder
from models.scenario_control_ldm import ScenarioControlLDM

import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from cfgs.config import CONFIG_PATH
from hydra.utils import instantiate
from omegaconf import OmegaConf
from utils.train_helpers import cache_latent_stats, set_latent_stats


def _make_logger(cfg, save_dir):
    if not cfg.train.track:
        return None
    return WandbLogger(
        project=cfg.train.wandb_project,
        name=cfg.train.run_name,
        entity=cfg.train.wandb_entity,
        log_model=False,
        save_dir=save_dir
    )


def train_autoencoder(cfg, save_dir=None):
    """ Train the ScenarioControl AutoEncoder model."""
    datamodule = instantiate(cfg.datamodule, dataset_cfg=cfg.dataset)
    model = ScenarioControlAutoEncoder(cfg)

    # we always track the last epoch checkpoint for evaluation or resume training
    model_checkpoint = ModelCheckpoint(filename='model', save_last=True, save_top_k=0, dirpath=save_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=-1)

    trainer = pl.Trainer(accelerator=cfg.train.accelerator,
                         devices=cfg.train.devices,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_summary, model_checkpoint, lr_monitor],
                         max_steps=cfg.train.max_steps,
                         check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
                         precision=cfg.train.precision,
                         limit_train_batches=cfg.train.limit_train_batches,
                         limit_val_batches=cfg.train.limit_val_batches,
                         gradient_clip_val=cfg.train.gradient_clip_val,
                         logger=_make_logger(cfg, save_dir)
                        )

    trainer.fit(model, datamodule)


def train_ldm(cfg, cfg_ae, save_dir=None):
    """ Train the ScenarioControl Latent Diffusion Model."""
    # check if latent stats are cached, if not, compute them
    if not os.path.exists(cfg.dataset.latent_stats_path):
        cache_latent_stats(cfg)
    cfg = set_latent_stats(cfg)

    datamodule = instantiate(cfg.datamodule, dataset_cfg=cfg.dataset)

    if cfg.train.save_top_k > 0:
        model_checkpoint = ModelCheckpoint(monitor='val_loss', save_last=True, save_top_k=cfg.train.save_top_k, dirpath=save_dir, every_n_epochs=1, filename="model-{epoch:02d}")
    else:
        # we always track the last epoch checkpoint for evaluation or resume training
        model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=cfg.train.save_top_k, dirpath=save_dir, every_n_epochs=1, filename="model-{epoch:02d}")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=-1)

    trainer = pl.Trainer(accelerator=cfg.train.accelerator,
                         devices=cfg.train.devices,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_summary, model_checkpoint, lr_monitor],
                         max_steps=cfg.train.max_steps,
                         check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
                         precision=cfg.train.precision,
                         limit_train_batches=cfg.train.limit_train_batches,
                         limit_val_batches=cfg.train.limit_val_batches,
                         gradient_clip_val=cfg.train.gradient_clip_val,
                         logger=_make_logger(cfg, save_dir)
                        )

    if cfg.train.finetune:
        model = ScenarioControlLDM(cfg=cfg, cfg_ae=cfg_ae)
        ckpt_path = os.path.join(cfg.train.pretrained_dir, 'last.ckpt')
        print("Finetuning from checkpoint: ", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        ret = model.load_state_dict(state_dict, strict=False)
        print("[load] missing:", len(ret.missing_keys), "unexpected:", len(ret.unexpected_keys))

        if cfg.train.freeze_pretrained:
            model.apply_freeze_policy_from_missing(missing_keys=ret.missing_keys)
    else:
        model = ScenarioControlLDM(cfg=cfg, cfg_ae=cfg_ae)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params} (trainable: {trainable_params})")

    trainer.fit(model, datamodule)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config3dtemp")
def main(cfg):
    # need to track whether we are training a nuplan or waymo model as
    # nuplan predicts lane types (lane/green light/red light) and waymo does not
    dataset_name = cfg.dataset_name.name
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
        cfg.dimension = dimension
        OmegaConf.set_struct(cfg, True)    # relock
        OmegaConf.set_struct(cfg_ae, True)

    pl.seed_everything(cfg.train.seed, workers=True)

    # checkpoints saved here
    save_dir = os.path.join(cfg.train.save_dir, cfg.train.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"Training ScenarioControl {model_name} on {cfg.dataset_name} dataset.")

    if 'autoencoder' in model_name:
        train_autoencoder(cfg, save_dir)
    elif 'ldm' in model_name:
        train_ldm(cfg, cfg_ae, save_dir)


if __name__ == '__main__':
    main()
