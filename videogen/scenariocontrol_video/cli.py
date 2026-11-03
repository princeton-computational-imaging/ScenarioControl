"""usage: scenariocontrol <command> [options]   (scenariocontrol <command> --help for details)

commands:
  generate   control video or BEV rollout -> front-camera video
  export     convert between training checkpoints and the Hub layout

Training runs through torchrun: torchrun --nproc_per_node=8 -m scenariocontrol_video.train
"""

import argparse
import sys
from pathlib import Path

from scenariocontrol_video.pipeline import DEFAULT_REPO, VARIANTS


def generate(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="scenariocontrol generate",
                                 description="Control video or BEV rollout -> front-camera video.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wireframe", help="control video (.mp4)")
    src.add_argument("--bev", help="directory of frame_XXXX.pkl rollout frames")
    ap.add_argument("--variant", choices=VARIANTS, default="prompt-only")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo (default: %(default)s)")
    ap.add_argument("--model-dir", help="local variant dir (config.json + adapter.safetensors); skips --repo")
    ap.add_argument("--prompt", help="scene description (required for prompt-only)")
    ap.add_argument("--image", help="real first frame (required for first-frame)")
    ap.add_argument("--frames", type=int, default=41, help="frames to generate; only 41 is validated")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--steps", type=int, help="sampling steps (default: 35)")
    ap.add_argument("--lowvram", action="store_true", help="int8 base weights, for 32 GB GPUs")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)

    from scenariocontrol_video.pipeline import ScenarioControlVideoPipeline

    if args.model_dir:
        pipe = ScenarioControlVideoPipeline.from_pretrained(args.model_dir, lowvram=args.lowvram)
    else:
        pipe = ScenarioControlVideoPipeline.from_hub(args.repo, variant=args.variant, lowvram=args.lowvram)
    extra = {"num_steps": args.steps} if args.steps else {}
    video = pipe(wireframe=args.wireframe, bev=args.bev, prompt=args.prompt, image=args.image,
                 num_frames=args.frames, seed=args.seed, output_dir=args.out, **extra)
    print(f"saved {video.save(Path(args.out) / 'video.mp4')}")



def main(argv=None) -> None:
    from scenariocontrol_video import export

    argv = sys.argv[1:] if argv is None else list(argv)
    commands = {"generate": generate, "export": lambda a: export.main(a, prog="scenariocontrol export")}
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return
    if not argv or argv[0] not in commands:
        sys.exit(__doc__ + (f"\nunknown command: {argv[0]}" if argv else ""))
    commands[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
