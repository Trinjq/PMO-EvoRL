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
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.num_envs is not None:
        if args.num_envs % config.num_preference_workers:
            parser.error("--num-envs must be divisible by num_preference_workers")
        config.num_envs = args.num_envs
        config.num_updates_per_iter = args.num_envs // 10
        config.fold_iters = min(config.fold_iters, 100)
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.output_dir is not None:
        config.output_dir = args.output_dir
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
