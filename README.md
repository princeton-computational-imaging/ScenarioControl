

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
1. [Quick start: scene → video](#quick-start)
2. [Setup](#setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Pre-Trained Checkpoints](#pretrained-checkpoints)
5. [Training](#training)
6. [Inference](#inference)
7. [Evaluation](#evaluation)
8. [Simulation](#simulation)
9. [Video Generation](#video-generation)
10. [Citation](#citation)
11. [Acknowledgements](#acknowledgements)

## Quick start: scene → video <a name="quick-start"></a>

One command generates a scene, rolls it out in closed loop and turns the rollout into a front-camera driving video. It needs no dataset. There are two modes:

| `--mode` | You give | Scene | Video |
|---|---|---|---|
| `first-frame` | one real front-camera image | image-conditioned LDM: the road ahead of the camera | `first-frame` model, continuing your image |
| `prompt` | a BEV layout prompt and an appearance prompt | prompt-conditioned LDM: the road ahead of the ego | `prompt-only` model, appearance from the text |

You need Linux, [conda](https://docs.conda.io/), [uv](https://docs.astral.sh/uv/), an NVIDIA GPU with ≥40 GB (e.g. A6000, A100, H100; 32 GB with `--lowvram`), about 48 GB of RAM (loading the prompt mode's 11 GB text encoder peaks near 40 GB) and about 70 GB of disk for checkpoints and the Hugging Face cache.

**1. Install.** Two environments: conda for scenes and rollouts, a Python 3.13 venv for the video model. The script calls both.
```bash
git clone https://github.com/princeton-computational-imaging/ScenarioControl.git && cd ScenarioControl
export SCRATCH_ROOT=$(pwd)  # checkpoints go to $SCRATCH_ROOT/checkpoints
conda env create -f environment.yml && conda activate ScenarioControl
uv venv -p 3.13 videogen-env
uv pip install -p videogen-env torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -p videogen-env "flash-attn==2.7.4.post1+cu128.torch210" --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0
uv pip install -p videogen-env "scenariocontrol-video[lowvram] @ git+https://github.com/princeton-computational-imaging/ScenarioControl#subdirectory=videogen"
```

**2. Get access and download the checkpoints.** Log in to Hugging Face (`hf auth login`, or `export HF_TOKEN=<token>`) and accept the terms of two gated models: [DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) (image features, `first-frame` mode; Meta reviews each request) and the [video model](https://huggingface.co/pci-lab/scenariocontrol-videogen). Then download the scene, text-encoder and CtRL-Sim checkpoints:
```bash
C=$SCRATCH_ROOT/checkpoints
mkdir -p $C/scenario_control_autoencoder3d_nuplan $C/scenario_control_ldm_img_cond_nuplan $C/scenario_control_ldm_prompt_cond_nuplan
gdown 1UM0mnyaX2GwfxxhM9nT7LhcftrrtXIzq -O $C/scenario_control_autoencoder3d_nuplan/last.ckpt
gdown 1WgDahzGzlZTyaStzxp-bUJgRLSrIgO-I -O $C/scenario_control_ldm_img_cond_nuplan/last.ckpt     # first-frame mode
gdown 1aS7rKr7MnCaZFDPV_y3tTskMbPi6XSyE -O $C/scenario_control_ldm_prompt_cond_nuplan/last.ckpt  # prompt mode
hf download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth --local-dir $C/text_encoders/umt5-xxl  # prompt mode (UMT5-XXL, 11 GB)
gdown --folder https://drive.google.com/drive/folders/1vw84BCfqqolY4DTl3YXOZwfY6nYnCmNo -O $C/ctrl_sim_waymo
```
ZoeDepth, DINOv3, the UMT5 tokenizer and the video base model (`nvidia/Cosmos3-Nano`, ~32 GB) download into `$HF_HOME` on first use.

**3. Run** from the repository root, with the conda environment active:
```bash
bash scripts/scene_to_video.sh --mode first-frame --image my_front_camera.jpg --out out/first_frame
bash scripts/scene_to_video.sh --mode prompt --out out/prompt \
  --scene-prompt "ROAD & CONTROL: Two through lanes ahead of the ego on a straight road with dashed lane markings; no intersections, crosswalks or traffic lights. AGENTS: A vehicle is directly ahead in the ego lane and another is slightly ahead in the left lane; a few cars are parked along the right curb." \
  --video-prompt "A quiet residential street with smooth asphalt and clear lane markings, lined with two-story houses, green lawns and tall trees. The weather is clear, with warm low sun in the early evening."
```
Each run takes 6–14 minutes on an A6000, mostly loading checkpoints and running the video model. It writes `scene.png` (BEV of the generated scene), `rollout/<scene>_seed0/frame_*.pkl` (41 BEV frames), `rollout_bev.mp4`, `control.mp4` (wireframe) and `video.mp4` (41 frames, 10 fps) to `--out`. The steps, which you can also run one by one: [`generate_scene.py`](generate_scene.py) (scene from an image or a prompt), [postprocessing and `run_simulation.py`](#simulation), and [`scenariocontrol generate`](#video-generation). Notes:
- `--scene-prompt` describes the BEV layout in the style of the training captions (`ROAD & CONTROL: ... AGENTS: ...`). `--video-prompt` describes the appearance (place, architecture, weather, lighting) and should not describe vehicles; it has a default.
- The image should come from a forward-facing camera at about roof height. It is centre-cropped to 16:9 and resized to 1920×1080 (the nuPlan front camera), then conditions both the scene and the video.
- Generated scenes have no camera calibration, so the wireframe is drawn with a standard nuPlan front-camera rig at the ego pose. It ignores your camera's real height and pitch, so in `first-frame` mode the wireframe may not line up exactly with your image.
- `--num-scenes` scenes (default 16) are generated, and the first that passes postprocessing is rolled out (scenes with traffic-light lanes or an ego route shorter than 30 m are skipped, and the reason is printed). If none passes, the next two seeds are tried.
- Scenes cover 64×64 m ahead of the ego. A fast ego (about 15 m/s) reaches the end of the scene near the end of the 4 s rollout; the last wireframes then show little road and the video model continues it on its own. The prompt LDM was fine-tuned on scenes centred on the ego; `--scene-type 0` uses those, but leaves only 32 m of road ahead.
- The number of agents is drawn from a nuPlan prior and does not depend on the image or prompt. For a busy street, set it, e.g. `--num-agents 25` (counts include the ego). Other options: `--seed`, `--lowvram`, `--steps`, `--model-dir` (local video weights); see the script header.
- The scene and video models are trained on nuPlan and CtRL-Sim on Waymo, so they are for non-commercial use only (see [Simulation](#simulation) and [Video Generation](#video-generation)). DINOv3 has its own license.

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
The environment runs on NVIDIA GPUs from Ampere (e.g. A6000) to Blackwell (e.g. RTX 5090). Attention uses [flash-attn](https://github.com/Dao-AILab/flash-attention) when it is installed and PyTorch SDPA otherwise; for faster training, install it with `pip install flash-attn --no-build-isolation`.

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

3. **Extract train/val/test splits and preprocess data for training**  
   Run the following to extract train/val/test splits and create the preprocessed data for training.
   ```
   bash scripts/extract_nuplan_data_3dtemp_wimages.sh # create train/val/test splits and create eval set for computing metrics
   bash scripts/preprocess_nuplan_dataset_3dtemp_wimages.sh # preprocess data to facilitate efficient model training
   ```

4. **Extract DINO patch features and depth maps for image conditioning**  
   Run the following to extract DINOv3 patch features and a monocular depth map for each camera frame.
   ```
   export NUPLAN_DATA_FOLDER=/path/to/nuplan-v1.1 # root of the raw nuPlan dataset (contains sensor_blobs/)
   bash scripts/extract_dino_depth_features.sh
   ```

5. **Prepare captions for prompt conditioning**  
   Download our [captions extracted with GPT-4.1-mini](https://drive.google.com/file/d/1V8Koo8LkKDFXFE_SHr-ar0XkTvfi_zK-/view?usp=sharing) and place them under `$SCRATCH_ROOT/captions_bev/gpt_bev`


### Waymo

Coming soon

## Pre-Trained Checkpoints <a name="pretrained-checkpoints"></a>

Pre-trained checkpoints can be downloaded from <a href="https://drive.google.com/drive/folders/1V2GC7eU7CnfaOeebHP-G5S0rKJBbJ0Kp?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a>. Place the `checkpoints` directory into your scratch (`$SCRATCH_ROOT`) directory. 

#### Checkpoints

   - Autoencoder: put under `$SCRATCH_ROOT/checkpoints/scenario_control_autoencoder3d_nuplan`
   - Unconditional Pretrained LDM: put under `$SCRATCH_ROOT/checkpoints/scenario_control_ldm_base_nuplan`
   - LDM for image-conditioning: put under `$SCRATCH_ROOT/checkpoints/scenario_control_ldm_img_cond_nuplan`
   - LDM for prompt-conditioning: put under `$SCRATCH_ROOT/checkpoints/scenario_control_ldm_prompt_cond_nuplan`
   - Text encoder for prompt-conditioning (UMT5-XXL from [Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)): `hf download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth --local-dir $SCRATCH_ROOT/checkpoints/text_encoders/umt5-xxl`



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
  ldm.dataset.load_scene_type='12' \
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
These commands condition on nuPlan test frames, so they need the prepared dataset. To generate from your own image or prompt instead, use [`generate_scene.py`](generate_scene.py) (no dataset needed):
````bash
python generate_scene.py generate_scene.image=front.jpg ldm.eval.run_name=my_scenes
python generate_scene.py "generate_scene.prompt='ROAD & CONTROL: ... AGENTS: ...'" ldm.eval.run_name=my_scenes
````
It writes `generate_scene.num_samples` scenes to `$SCRATCH_ROOT/checkpoints/my_scenes/initial_scene_samples` and BEV plots to `$SCRATCH_ROOT/checkpoints/my_scenes/viz`. Image conditioning computes the DINOv3 features and depth map of the image on the fly (the image is centre-cropped to 16:9), and each scene's number of lanes and agents is drawn from the nuPlan prior of a city (`generate_scene.map_id`).

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
  ldm.dataset.load_scene_type='2' \
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
  ldm.dataset.load_scene_type='0' \
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

## Simulation <a name="simulation"></a>
Generated scenes can be rolled out in closed loop. An IDM planner drives the ego, and [CtRL-Sim](https://github.com/montrealrobotics/ctrl-sim) controls the other agents. Each rollout is saved as a sequence of BEV frames (`frame_XXXX.pkl`), which the video generator renders into driving videos (see the [project page](https://princeton-computational-imaging.github.io/ScenarioControl/)).

### 1. Download the CtRL-Sim checkpoint
We use the Waymo-trained CtRL-Sim checkpoint released by [Scenario Dreamer](https://github.com/princeton-computational-imaging/scenario-dreamer) ([Google Drive](https://drive.google.com/drive/folders/1vw84BCfqqolY4DTl3YXOZwfY6nYnCmNo), 83.3 MB). Place it at `$SCRATCH_ROOT/checkpoints/ctrl_sim_waymo/last.ckpt` (the default `sim.behaviour_model.model_path`):
```bash
gdown --folder https://drive.google.com/drive/folders/1vw84BCfqqolY4DTl3YXOZwfY6nYnCmNo -O $SCRATCH_ROOT/checkpoints/ctrl_sim_waymo
sha256sum $SCRATCH_ROOT/checkpoints/ctrl_sim_waymo/last.ckpt
# 8ed2d3a0546a06907f797224492c44b38013ae804af1de0fe9991814d12d0062
```
The checkpoint was trained on the Waymo Open Dataset, so its use is subject to the [Waymo Open Dataset license terms](https://waymo.com/open/terms/) (non-commercial).

### 2. Generate simulation environments
Generate image- or prompt-conditioned scenes with `test.py` and `ldm.eval.cache_samples=True` (see [Inference](#inference)). The samples are written to `$SCRATCH_ROOT/checkpoints/<ldm.eval.run_name>/initial_scene_samples`. Each sample keeps the conditioning nuPlan frame's ego pose (`ego_state_og`), and image-conditioned samples also keep its camera (`cam_infos`). Scenes from [`generate_scene.py`](generate_scene.py) are written to the same place and have neither. Convert the samples into simulator-ready scenes. This samples an ego route, removes invalid agents and compresses the lane graph:
```bash
python data_processing/postprocess_simulation_environments.py \
  dataset_name=nuplan \
  ldm.eval.run_name=scenario_control_ldm_img_cond_nuplan_test
```
The scenes are written to `$SCRATCH_ROOT/simulation_environment_datasets/scenario_control_nuplan_pickles`. Scenes with traffic-light lanes, without a valid ego route, or with an ego route shorter than `postprocess_sim_envs.route_length` (default 30 m) are skipped, and the reason is printed.

You can also bring your own scenes. A scene is a `.pkl` dict with these keys:
- `agents`: `[N, 1, 10]`, with the ego as the last row (layout as in the table below, in the scene frame).
- `agent_types`: `[N, 5]`.
- `lanes`: `[L, 50, 2]`.
- `lane_graph`, `num_lanes`.
- `route`: `[R, 2]`, points 1 m apart.
- `route_lane_indices`.
- Optional: `cam_infos` and `ego_state_og`.

### 3. Run rollouts
```bash
python run_simulation.py \
  sim.dataset_path=$SCRATCH_ROOT/simulation_environment_datasets/scenario_control_nuplan_pickles \
  sim.output_dir=$SCRATCH_ROOT/rollouts \
  sim.steps=40 \
  sim.num_seeds=1 \
  sim.stop_on_termination=False
```
- A rollout writes `sim.steps + 1` frames at 10 Hz (41 frames for 40 steps, the video generator's default length). `sim.stop_on_termination=False` keeps every rollout at full length. With the default `True`, a rollout stops early if the ego collides, leaves the route or completes it.
- `sim.visualize=True` also writes a BEV video of each rollout to `sim.movie_path`.
- These flags control when agents are removed from the scene: `sim.keep_agents_leaving_fov`, `sim.remove_colliding_agents`, `sim.remove_offroad_agents` and `sim.agent_offroad_threshold`. The defaults reproduce stock CtRL-Sim behaviour. See [cfgs/sim/base.yaml](cfgs/sim/base.yaml).
- To split the scenes across jobs, pass `sim.task_id=<i> sim.num_tasks=<n>`.

### Output format
Each rollout is written to `<sim.output_dir>/<scenario>_seed<k>/frame_0000.pkl ... frame_<steps>.pkl`. Frame 0 is the initial state, and frame `t` is the state after `t` steps of 0.1 s. Agents and lanes are in the **current ego frame**: the ego is at the origin, x points forward and y points left (metres, radians). The ego is the **last** agent row.

| key | type / shape | content |
|---|---|---|
| `agents` | float64 `[N+1, 1, 10]` | `x, y, vx, vy, theta, length, width, exists, z, height` (m, m/s, rad). `z` is relative to the ego. The ego row is zero except `length, width, exists=1, height`, so ego speed is not stored. |
| `agent_types` | float64 `[N+1, 5]` | one-hot `[unset, vehicle, pedestrian, cyclist, other]`. The ego row is `vehicle`. |
| `agent_active` | bool `[N+1]` | which agents are in the scene at this frame (ego is always `True`). Inactive rows keep their last state and `exists` can still be 1, so filter by this mask. |
| `lanes` | float64 `[L, 50, 2]` | lane centerlines, current ego frame. |
| `lane_graph` | `networkx.DiGraph` | lane successor graph (node = lane index). |
| `route`, `route_lane_indices` | float64 `[R, 2]`, list | ego route in the **initial** scene frame (not ego-centred), and the lane ids it follows. |
| `num_agents`, `num_lanes` | int | `N+1`, `L` |
| `ego_state_og` | list | only if the scene has it. Global ego pose: `[translation (3,), quaternion wxyz (4,), (length, width, height) (3,), heading]`, yaw-only and updated every frame. |
| `cam_infos` | dict | only if the scene has it. The nuPlan `CAM_F0` calibration (`intrinsic`, `translation`, `rotation`, `distortion`, `filename_jpg`, ...). Its global `ego_pose` `[x, y, z, qw, qx, qy, qz]` moves rigidly with the ego every frame. |

`python tests/test_sequence_saver.py` checks this format without a checkpoint.

## Video Generation <a name="video-generation"></a>

Turn a rolled-out scene into a front-camera driving video. The scene is rendered as a wireframe control video (lanes and 3D agent boxes seen from the front camera), and a fine-tuned [NVIDIA Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) generates the video. The weights are on the Hugging Face Hub at [`pci-lab/scenariocontrol-videogen`](https://huggingface.co/pci-lab/scenariocontrol-videogen), in two variants:

| Variant | Inputs | Use it for |
|---|---|---|
| `first-frame` | wireframes + the real first frame | continuing a real front-camera image |
| `prompt-only` | wireframes + a text prompt | new appearance from text (weather, time of day, place) |

> **Non-commercial.** The video models were fine-tuned on nuPlan camera data (© Motional, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) plus the [nuPlan Terms of Use](https://www.nuscenes.org/terms-of-use)). The weights, and videos you generate with them, are for non-commercial use only. The base model is NVIDIA's, under [OpenMDW-1.1](videogen/LICENSE-OpenMDW-1.1); see [`videogen/NOTICE`](videogen/NOTICE). The code in this repository is under this repository's terms.

### 1. Install

The video code is a separate pip package in [`videogen/`](videogen) with its own Python 3.13 venv (required by the prebuilt flash-attention wheel); it does not need the conda environment above. You need Linux (x86-64), an NVIDIA GPU with ≥40 GB (e.g. A6000, A100, H100) or 32 GB with `--lowvram`, and [`uv`](https://docs.astral.sh/uv/). Create the venv with the four `uv` lines of [Quick start](#quick-start), step 1, then `source videogen-env/bin/activate` to use the commands below.

### 2. Get access

The weights are gated. Log in and accept the terms on the [model page](https://huggingface.co/pci-lab/scenariocontrol-videogen):
```bash
hf auth login   # or: export HF_TOKEN=<token>
```
The first run also downloads the base model (`nvidia/Cosmos3-Nano`, ~32 GB, at a pinned revision), the Wan2.2 VAE and the Qwen3-VL processor into the Hugging Face cache (`$HF_HOME`).

### 3. (Optional) Make BEV rollouts

Follow [Simulation](#simulation). Each rollout is a directory of `frame_XXXX.pkl` files at 10 fps, which is the `--bev` input below; the first 41 frames are used by default.

### 4. Generate a video

From a BEV rollout:
```bash
scenariocontrol generate --variant prompt-only --bev path/to/frames \
  --prompt "A rainy evening drive through a downtown street with neon signs." --out out/
scenariocontrol generate --variant first-frame --bev path/to/frames --image first_frame.png --out out/
```
or from a wireframe control video you already have (`--wireframe control.mp4`, 1280×720 at 10 fps; frame 0 is the real first frame or a wireframe, frames 1+ are wireframes). The output is `out/video.mp4`, plus the rendered `out/control.mp4` for `--bev`. In Python:
```python
from scenariocontrol_video import ScenarioControlVideoPipeline

pipe = ScenarioControlVideoPipeline.from_hub("pci-lab/scenariocontrol-videogen", variant="prompt-only")
video = pipe(bev="path/to/frames", prompt="A snowy morning on a suburban road.", seed=43)
video.save("out.mp4")
```
Notes:
- Videos are 41 frames (4.1 s at 10 fps), the training length. `--frames` accepts more; longer videos are generated as independent 41-frame chunks, which is untested and may show seams.
- On a 32 GB GPU add `--lowvram` (int8 base weights). Loading still peaks near 31 GB, so the GPU must otherwise be empty.
- Frames without camera calibration (`cam_infos`) use a standard nuPlan front-camera rig at the ego pose, which ignores the real camera height and pitch.
- The released models were trained on wireframes that also had small agent-centre dots, which this renderer leaves out; outputs are visually equivalent in our checks.
- `first-frame` was trained with one fixed caption, which is used by default; give `prompt-only` a scene description.

### 5. (Optional) Train or fine-tune

Training uses cosmos-framework's trainer and `ffmpeg`/`ffprobe`, on 8 GPUs with ≥48 GB each. Set up its environment at the pinned commit and add this package:
```bash
git clone https://github.com/NVIDIA/cosmos-framework && cd cosmos-framework
git checkout fa63e9934a38eb8127a0c3d6b23969fd4d0e32a7
uv sync --all-extras --group=cu128-train && source .venv/bin/activate
uv pip install -e /path/to/ScenarioControl/videogen
```
Data: put paired clips in `$DATASET_PATH/train/<id>.mp4` (real 41-frame front-camera clip) and `$DATASET_PATH/src_videos/<id>.mp4` (its wireframe control, frame 0 = the real first frame; render it with `scenariocontrol_video.render`), then write the training lists (`--captions` is a JSON of per-clip captions, needed for `prompt_only`):
```bash
python -m scenariocontrol_video.data $DATASET_PATH --captions captions.json
```
Make a starting checkpoint and train. `--from-variant` starts from a released variant (fine-tuning); without it, training starts from the base model:
```bash
scenariocontrol export dcp $SCRATCH_ROOT/videogen/init --from-variant prompt-only
export DATASET_PATH WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth BASE_CHECKPOINT_PATH=$SCRATCH_ROOT/videogen/init
torchrun --nproc_per_node=8 -m scenariocontrol_video.train --sft-toml=prompt_only
```
To fine-tune `first-frame`, use `--from-variant first-frame`, `--sft-toml=first_frame` and append `-- checkpoint.keys_to_skip_loading=[net_ema.]` (the recipe otherwise starts `inpaint_embedder` from zero). Checkpoints are written under `$IMAGINAIRE_OUTPUT_ROOT`.

Export your checkpoint to the Hub layout and load it like the released one:
```bash
scenariocontrol export adapter <run>/checkpoints/<checkpoint>/model my-model/prompt-only --variant prompt-only
scenariocontrol generate --model-dir my-model/prompt-only --bev path/to/frames --prompt "..." --out out/
```

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
- [ctrl-sim](https://github.com/montrealrobotics/ctrl-sim) (MIT, see [licenses/ctrl-sim.txt](licenses/ctrl-sim.txt))
