"""Run the GPU-native PD-MORL MO-TD3 workflow."""

import argparse
from pathlib import Path

import jax
from omegaconf import OmegaConf

from evorl.recorders import LogRecorder
from pmo_evorl.workflow import MOTD3Workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/mo_td3_walker2d.yaml"
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow = MOTD3Workflow.build_from_config(config, enable_jit=True)
    workflow.add_recorders(
        [LogRecorder(str(output_dir / "training.log"), console=True)]
    )
    try:
        state = workflow.init(jax.random.PRNGKey(config.seed))
        workflow.learn(state)
    finally:
        workflow.close()


if __name__ == "__main__":
    main()
