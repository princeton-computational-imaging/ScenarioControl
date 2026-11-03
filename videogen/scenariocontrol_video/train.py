"""Training entry point: registers the ScenarioControl experiments, then runs cosmos-framework's trainer.

    torchrun --nproc_per_node=8 -m scenariocontrol_video.train --sft-toml=prompt_only [-- hydra.overrides=...]

``--sft-toml`` takes a path or one of the packaged recipes: first_frame, prompt_only.
"""

import runpy
import sys
from pathlib import Path

import torch.multiprocessing

import scenariocontrol_video.experiments  # noqa: F401  (registers the experiments)

CONFIGS = Path(__file__).parent / "configs"


def main() -> None:
    # Share dataloader tensors via files: the default file_descriptor strategy exhausts /dev/shm.
    torch.multiprocessing.set_sharing_strategy("file_system")
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--sft-toml=") and (CONFIGS / f"{arg.split('=', 1)[1]}.toml").exists():
            sys.argv[i] = f"--sft-toml={CONFIGS / arg.split('=', 1)[1]}.toml"
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
