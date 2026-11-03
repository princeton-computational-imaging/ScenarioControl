#!/usr/bin/env bash
# One command from a scene to a front-camera driving video (see README, Quick start):
#   first-frame: real front-camera image -> image-conditioned scene -> rollout -> first-frame video
#   prompt:      BEV layout prompt       -> prompt-conditioned scene -> rollout -> prompt-only video
#
#   bash scripts/scene_to_video.sh --mode first-frame --image front.jpg --out out/first_frame
#   bash scripts/scene_to_video.sh --mode prompt --scene-prompt "ROAD & CONTROL: ... AGENTS: ..." \
#       [--video-prompt "<scene appearance: place, architecture, weather, lighting>"] --out out/prompt
#
# Run from the repository root with SCRATCH_ROOT set and the conda environment active (or point
# SC_PYTHON at its python). The video step runs in the video venv, VIDEOGEN_ENV (default ./videogen-env).
# Options:
#   --seed N          scene seed (default 0). If no scene survives postprocessing, seeds N+1 and N+2 are tried.
#   --num-scenes K    scenes generated per seed (default 16); the first one that survives postprocessing is used
#   --num-agents N    agents per scene including the ego (default: drawn from the nuPlan prior, which ignores the image/prompt)
#   --scene-type T    2 (default): 64x64 m ahead of the ego; 0: 64x64 m around it, the type the prompt LDM was fine-tuned
#                     on, which leaves only 32 m of road ahead (about 3 s of driving) for the 4 s video
#   --model-dir DIR   local video model variant dir, instead of the Hub repo --repo (default pci-lab/scenariocontrol-videogen)
#   --lowvram         int8 video model, for 32 GB GPUs
#   --steps N         video sampling steps (default 35)
# Writes to --out: scene.png (BEV of the scene used), scene/ (generated scenes, plus first_frame.png in
# first-frame mode), sim_env/ (the simulator-ready scene), rollout/<scene>_seed0/frame_*.pkl,
# rollout_bev.mp4, control.mp4 (wireframe), video.mp4 and timings.txt.
set -euo pipefail

MODE= IMAGE= SCENE_PROMPT= OUT= SEED=0 NUM_SCENES=16 MODEL_DIR= REPO=pci-lab/scenariocontrol-videogen VG_EXTRA=() COND=() SCENE_TYPE=2
VIDEO_PROMPT="A suburban street with a straight, multi-lane asphalt road and clear lane markings, lined with low buildings, trees and utility poles. The weather is clear and sunny, with soft late-afternoon light."
die() { echo "scene_to_video: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --scene-prompt) SCENE_PROMPT=$2; shift 2 ;;
    --video-prompt) VIDEO_PROMPT=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --num-scenes) NUM_SCENES=$2; shift 2 ;;
    --num-agents) COND+=(generate_scene.num_agents="$2"); shift 2 ;;
    --scene-type) SCENE_TYPE=$2; shift 2 ;;
    --model-dir) MODEL_DIR=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --lowvram) VG_EXTRA+=(--lowvram); shift ;;
    --steps) VG_EXTRA+=(--steps "$2"); shift 2 ;;
    *) die "unknown argument $1 (see the header of $0)" ;;
  esac
done
[[ -n $OUT ]] || die "--out is required"
case $MODE in
  first-frame) [[ -f $IMAGE ]] || die "--mode first-frame needs --image <front-camera image>"
               export SC_IMAGE=$(realpath "$IMAGE"); COND+=('generate_scene.image=${oc.env:SC_IMAGE}') ;;
  prompt) [[ -n $SCENE_PROMPT ]] || die "--mode prompt needs --scene-prompt <BEV layout description>"
          export SC_SCENE_PROMPT=$SCENE_PROMPT; COND+=('generate_scene.prompt=${oc.env:SC_SCENE_PROMPT}') ;;
  *) die "--mode must be first-frame or prompt" ;;
esac

COND+=(generate_scene.scene_type="$SCENE_TYPE")
cd "$(dirname "$0")/.."
: "${SCRATCH_ROOT:?set SCRATCH_ROOT (see README, Setup)}"
export PYTHONPATH=${PYTHONPATH:-}; source scripts/define_env_variables.sh
PY=${SC_PYTHON:-python}
VG=${VIDEOGEN_ENV:-$PWD/videogen-env}/bin/scenariocontrol
[[ -x $VG ]] || die "no video generator at $VG: install it (README, Video Generation) or set VIDEOGEN_ENV"
OUT=$(realpath -m "$OUT"); mkdir -p "$OUT"; : > "$OUT/timings.txt"
timed() { local name=$1 t0=$SECONDS; shift; "$@"; echo "$name: $((SECONDS - t0)) s" | tee -a "$OUT/timings.txt"; }

# 1. Scene generation + postprocessing (ego route, lane graph); retry with the next seed if no scene survives
for s in "$SEED" $((SEED + 1)) $((SEED + 2)); do
  rm -rf "$OUT/scene" "$OUT/sim_env"
  timed "scene generation (seed $s)" "$PY" generate_scene.py "${COND[@]}" generate_scene.num_samples="$NUM_SCENES" \
    ldm.eval.seed="$s" ldm.eval.save_dir="$OUT" ldm.eval.run_name=scene hydra.run.dir="$OUT/logs/generate"
  timed "postprocess" "$PY" data_processing/postprocess_simulation_environments.py \
    postprocess_sim_envs.pre_path="$OUT/scene/initial_scene_samples" postprocess_sim_envs.post_path="$OUT/sim_env" \
    postprocess_sim_envs.max_num_envs=1 hydra.run.dir="$OUT/logs/postprocess"
  compgen -G "$OUT/sim_env/*.pkl" > /dev/null && break
  echo "scene_to_video: none of the $NUM_SCENES scenes from seed $s survived postprocessing (reasons above)"
done
SCENE=$(basename "$(compgen -G "$OUT/sim_env/*.pkl" | head -1)" .pkl) || true
[[ -n $SCENE ]] || die "no usable scene after 3 seeds; try --num-scenes 32 or another --seed"
cp "$OUT/scene/viz/$SCENE.png" "$OUT/scene.png"

# 2. Closed-loop rollout: IDM ego + CtRL-Sim agents, 40 steps -> 41 frames at 10 Hz
timed "rollout" "$PY" run_simulation.py sim.dataset_path="$OUT/sim_env" sim.output_dir="$OUT/rollout" sim.steps=40 \
  sim.num_seeds=1 sim.stop_on_termination=False sim.visualize=True sim.movie_path="$OUT/rollout_bev" \
  hydra.run.dir="$OUT/logs/rollout"
cp "$OUT/rollout_bev/${SCENE}_seed0.mp4" "$OUT/rollout_bev.mp4"

# 3. Wireframe control video + video model (separate venv; the repo root must not shadow its packages)
if [[ $MODE == first-frame ]]; then VG_ARGS=(--variant first-frame --image "$OUT/scene/first_frame.png")
else VG_ARGS=(--variant prompt-only --prompt "$VIDEO_PROMPT"); fi
if [[ -n $MODEL_DIR ]]; then VG_ARGS+=(--model-dir "$MODEL_DIR"); else VG_ARGS+=(--repo "$REPO"); fi
timed "video generation" env -u PYTHONPATH "$VG" generate "${VG_ARGS[@]}" --bev "$OUT/rollout/${SCENE}_seed0" --out "$OUT" "${VG_EXTRA[@]}"
echo "scene_to_video: done -> $OUT/video.mp4 (wireframe: $OUT/control.mp4, BEV: $OUT/scene.png, $OUT/rollout_bev.mp4)"
