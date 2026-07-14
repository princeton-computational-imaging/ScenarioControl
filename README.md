

# Official Repository for ScenarioControl

<p align="left">
<a href="https://arxiv.org/abs/2604.17147" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2604.17147-b31b1b.svg?style=flat" /></a>
<a href="https://princeton-computational-imaging.github.io/ScenarioControl/" alt="webpage">
    <img src="https://img.shields.io/badge/Project Page-ScenarioControl-blue" /></a>

> [**ScenarioControl: Vision-Language Controllable Vectorized Latent Scenario Generation**](https://arxiv.org/abs/2604.17147)  <br>
> Lili Gao<sup>1*</sup>, Yanbo Xu<sup>2*</sup>, William Koch<sup>2*</sup>, Samuele Ruffino<sup>1</sup>, Luke Rowe<sup>3</sup>, Behdad Chalaki<sup>1</sup>, Dmitriy Rivkin<sup>1</sup>, Julian Ost<sup>1,2</sup>, Roger Girgis<sup>1,3</sup>, Mario Bijelic<sup>1,2</sup>, Felix Heide<sup>1,2</sup>  <br>
> <sup>1</sup> Torc Robotics, <sup>2</sup> Princeton University, <sup>3</sup> Mila <br>
> <br>
> European Conference on Computer Vision (ECCV), 2026 <br>
> <br>
> <sup>*</sup> Equal contribution <br>
>

We propose **ScenarioControl**, a vision-language controllable framework for learned driving scenario generation.

<video src="https://github.com/user-attachments/assets/3e657c21-56b1-4c1c-9713-7bbe75ce9cc1" width="640" height="250"></video>

## Table of Contents
1. [Setup](#setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Pre-Trained Checkpoints](#pretrained-checkpoints)
4. [Inference](#inference)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Simulation](#simulation)
8. [Citation](#citation)
9. [Acknowledgements](#acknowledgements)

## Setup <a name="setup"></a>

Start by cloning the repository
```
git clone https://github.com/princeton-computational-imaging/ScenarioControl.git
cd ScenarioControl
```

This repository assumes you have a "scratch" directory for larger files (datasets, checkpoints, etc.). If disk space is not an issue, you can keep everything in the repository directory:
```
export SCRATCH_ROOT=$(pwd) # prefer a separate drive? Point SCRATCH_ROOT there instead.
```

Define environment variables to let the code know where things live:
```
source $(pwd)/scripts/define_env_variables.sh
```

### Conda Setup 

```
# create conda environment
conda env create -f environment.yml
conda activate ScenarioControl

# login to wandb for experiment logging
export WANDB_API_KEY=<your_api_key>
wandb login
```

## Dataset Preparation <a name="dataset-preparation"></a>

Dataset for downloading is coming soon! You could also follow the steps below to extract and preprocess the data yourself.

### NuPlan

We use the same extracted NuPlan data as [SLEDGE](https://github.com/autonomousvision/sledge), with minor modifications tailored for **ScenarioControl**. Our modified fork for extracting the Nuplan data is available [here](https://github.com/lilligao/sledge-scenario-control).

#### Step-by-Step Instructions

1. **Install dependencies & download raw NuPlan data**  
   Follow the guide in the [`installation.md`](https://github.com/lilligao/sledge-scenario-control/blob/main/docs/installation.md) file of our forked repo.  
   This will walk you through:
   - Downloading the NuPlan dataset
   - Setting up the correct environment variables
   - Installing the `sledge-devkit`

2. **Extract NuPlan data**  
   Run the following in the forked Repo to preprocess the NuPlan data:
   ```
   cd $SLEDGE_DEVKIT_ROOT/scripts/autoencoder/rvae/
   bash feature_caching_rvae_temporal.sh
   bash feature_caching_rvae_temporal_test.sh
   python merge_meta.py
   ```

4. **Extract train/val/test splits and preprocess data for training**  
   Run the following to extract train/val/test splits and create the preprocessed data for training.
   ```
   bash scripts/extract_nuplan_data_3dtemp_wimages.sh # create train/val/test splits and create eval set for computing metrics
   bash scripts/preprocess_nuplan_dataset_3dtemp_wimages.sh # preprocess data to facilitate efficient model training
   ```

5. **Extract DINO patch features and depth maps for image conditioning**  
   Run the following to extract DINOv3 patch features and a monocular depth map for each camera frame.
   ```
   export NUPLAN_DATA_FOLDER=/path/to/nuplan-v1.1 # root of the raw nuPlan dataset (contains sensor_blobs/)
   bash scripts/extract_dino_depth_features.sh
   ```

6. **Extract captions for prompt conditioning**  
   Coming soon!


### Waymo

Coming soon

## Pre-Trained Checkpoints <a name="pretrained-checkpoints"></a>

Pre-trained checkpoints can be downloaded from <a href="https://drive.google.com/drive/folders/1V2GC7eU7CnfaOeebHP-G5S0rKJBbJ0Kp?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a>. Place the `checkpoints` directory into your scratch (`$SCRATCH_ROOT`) directory. 

#### Checkpoints

   - Autoencoder: put under `$SCRATCH_ROOT/checkpoints/scenario_control_autoencoder3d_nuplan`
   - Unconditional Pretrained LDM: put under `$SCRATCH_ROOT/checkpoints/scenario_control_ldm_base_nuplan`
   - LDM for image-conditioning: coming soon!
   - LDM for prompt-conditioning: coming soon! 



### Autoencoder Latent Caching:
````bash
# Cached latents are saved to: ae.eval.cache_latents.latent_dir
python eval.py \
  dataset_name=nuplan \
  model_name=autoencoder3dtemp \
  ae.eval.run_name=scenario_control_autoencoder3d_nuplan \
  ae.dataset.load_images=True \
  ae.eval.cache_latents.enable_caching=True \
  ae.eval.cache_latents.split_name=[train|val|test] \
  --config-name=config3dtemp
````


## Training <a name="training"></a>

### Pretrain Unconditional LDM
Trains the base unconditional LDM from scratch on all scene types (`load_scene_type='012'`). Image/prompt conditioning are later added on top of this checkpoint via `ldm.train.finetune=True` + `ldm.train.pretrained_dir` (see below).
````bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  dataset_name=nuplan \
  model_name=ldm_cond \
  ldm.model.autoencoder_run_name=scenario_control_autoencoder3d_nuplan \
  ldm.dataset.load_scene_type='012' \
  ldm.model.img_conditioning=False \
  ldm.train.run_name=scenario_control_ldm_base_nuplan \
  ldm.train.devices=4 \
  ldm.train.lr=5e-5 \
  ldm.train.track=True \
  ldm.train.save_top_k=-1 \
  ldm.train.check_val_every_n_epoch=5 \
  ldm.train.max_steps=500000 \
  ldm.train.num_samples_to_visualize=3 \
  ldm.datamodule.train_batch_size=64 \
  ldm.datamodule.val_batch_size=64 \
  --config-name=config3dtemp
````

### Finetune LDM with Image Conditioning
Finetunes a pretrained unconditional LDM checkpoint to add single-image conditioning (DINO patch features + depth map, see [Dataset Preparation](#dataset-preparation)). `ldm.train.freeze_pretrained=True` freezes every weight that loaded from `ldm.train.pretrained_dir`, training only the newly-added image-conditioning layers.
````bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  dataset_name=nuplan \
  model_name=ldm_cond \
  ldm.model.autoencoder_run_name=scenario_control_autoencoder3d_nuplan \
  ldm.dataset.load_single_img_cond=True \
  ldm.dataset.load_scene_type='02' \
  ldm.model.img_conditioning=True \
  ldm.model.decode_in_training=True \
  ldm.train.run_name=scenario_control_ldm_img_cond_nuplan \
  ldm.train.devices=4 \
  ldm.train.lr=5e-5 \
  ldm.train.track=True \
  ldm.train.finetune=True \
  ldm.train.pretrained_dir=$SCRATCH_ROOT/checkpoints/scenario_control_ldm_base_nuplan \
  ldm.train.freeze_pretrained=True \
  ldm.train.collision_weight=0.001 \
  ldm.datamodule.train_batch_size=64 \
  ldm.datamodule.val_batch_size=64 \
  --config-name=config3dtemp
````

### Finetune LDM with Prompt Conditioning
Finetunes a pretrained unconditional LDM checkpoint to add text-prompt conditioning via a frozen UMT5 text encoder. Captions can be cached (coming soon!) and are encoded on the fly since `ldm.dataset.use_cached_text_embeds=False`; set it to `True` instead if you've precomputed a `.pt` text-embedding cache. 
````bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  dataset_name=nuplan \
  model_name=ldm_cond \
  ldm.model.autoencoder_run_name=scenario_control_autoencoder3d_nuplan \
  ldm.dataset.load_scene_type='01' \
  ldm.dataset.load_captions=True \
  ldm.model.text_conditioning=True \
  ldm.dataset.use_cached_text_embeds=False \
  ldm.model.decode_in_training=True \
  ldm.train.run_name=scenario_control_ldm_prompt_cond_nuplan \
  ldm.train.devices=4 \
  ldm.train.lr=5e-5 \
  ldm.train.track=True \
  ldm.train.finetune=True \
  ldm.train.pretrained_dir=$SCRATCH_ROOT/checkpoints/scenario_control_ldm_base_nuplan \
  ldm.train.freeze_pretrained=True \
  ldm.train.collision_weight=0.001 \
  ldm.datamodule.train_batch_size=64 \
  ldm.datamodule.val_batch_size=64 \
  --config-name=config3dtemp
````

## Inference <a name="inference"></a>
### Generate Initial Scenes with LDM with Conditioning
Initial Scene Generation (Image Conditioning):
````bash
python test.py \
  dataset_name=nuplan \
  model_name=ldm_cond \
  ckpt_path=$SCRATCH_ROOT/checkpoints/scenario_control_ldm_img_cond_nuplan/last.ckpt \
  ldm.model.autoencoder_run_name=scenario_control_autoencoder3d_nuplan \
  ldm.model.img_conditioning=True \
  ldm.model.decode_in_training=True \
  ldm.dataset.load_single_img_cond=True \
  ldm.dataset.load_scene_type='02' \
  ldm.eval.mode=initial_scene \
  ldm.eval.run_name=scenario_control_ldm_img_cond_nuplan_test \
  ldm.eval.num_samples=100 \
  ldm.eval.visualize=True \
  ldm.eval.visualize_gt=True \
  ldm.eval.cache_samples=True \
  ldm.datamodule.test_batch_size=64 \
  --config-name=config3dtemp
````

Initial Scene Generation (Prompt Conditioning):
````bash
python test.py \
  dataset_name=nuplan \
  model_name=ldm_cond \
  ckpt_path=$SCRATCH_ROOT/checkpoints/scenario_control_ldm_prompt_cond_nuplan/last.ckpt \
  ldm.model.autoencoder_run_name=scenario_control_autoencoder3d_nuplan \
  ldm.model.text_conditioning=True \
  ldm.model.decode_in_training=True \
  ldm.dataset.load_captions=True \
  ldm.dataset.use_cached_text_embeds=False \
  ldm.dataset.load_scene_type='01' \
  ldm.eval.mode=initial_scene \
  ldm.eval.run_name=scenario_control_ldm_prompt_cond_nuplan_test \
  ldm.eval.num_samples=100 \
  ldm.eval.visualize=True \
  ldm.eval.cache_samples=True \
  ldm.datamodule.test_batch_size=64 \
  --config-name=config3dtemp
````

### Outpainting 
Coming soon! 

## Evaluation <a name="evaluation"></a>
Coming soon!

## Citation <a name="citation"></a>

If you find this work useful, please cite ScenarioControl:

```bibtex
@inproceedings{gao2026scenariocontrol,
  title     = {ScenarioControl: Vision-Language Controllable Vectorized Latent Scenario Generation},
  author    = {Gao, Lili and Xu, Yanbo and Koch, William and Ruffino, Samuele and Rowe, Luke and Chalaki, Behdad and Rivkin, Dmitriy and Ost, Julian and Girgis, Roger and Bijelic, Mario and Heide, Felix},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```


## Acknowledgements <a name="acknowledgements"></a>

Special thanks to the authors of the following open-source repositories:
- [Scenario Dreamer](https://github.com/princeton-computational-imaging/scenario-dreamer)
- [SLEDGE](https://github.com/autonomousvision/sledge)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [ctrl-sim](https://github.com/montrealrobotics/ctrl-sim)
