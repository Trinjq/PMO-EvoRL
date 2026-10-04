"""Measure steady-state training throughput without changing update ratios."""

import argparse
import time

import jax
from omegaconf import OmegaConf

from pmo_evorl.workflow import MOTD3Workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/mo_td3_walker2d.yaml"
    )
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.num_envs % 10:
        parser.error("--num-envs must be divisible by 10")
    config.num_envs = args.num_envs
    config.num_updates_per_iter = args.num_envs // 10
    config.fold_iters = args.steps
    config.replay_buffer_capacity = 100_000
    config.checkpoint.enable = False
    config.interpolator_eval_episodes = 1
    config.eval_episodes = 1
    config.pareto_step_size = 0.5

    workflow = MOTD3Workflow.build_from_config(config, enable_jit=True)
    try:
        state = workflow.init(jax.random.PRNGKey(config.seed))
        _, state = workflow._multi_steps(state)  # compile and warm up
        jax.block_until_ready(state.metrics.iterations)

        started = time.perf_counter()
        _, state = workflow._multi_steps(state)
        jax.block_until_ready(state.metrics.iterations)
        elapsed = time.perf_counter() - started

        raw_transitions = args.steps * args.num_envs * config.rollout_length
        critic_updates = (
            args.steps
            * config.num_updates_per_iter
            * config.actor_update_interval
        )
        actor_updates = args.steps * config.num_updates_per_iter
        print(
            {
                "num_envs": args.num_envs,
                "seconds": elapsed,
                "raw_transitions_per_second": raw_transitions / elapsed,
                "critic_updates_per_second": critic_updates / elapsed,
                "actor_updates_per_second": actor_updates / elapsed,
                "critic_updates_per_transition": critic_updates
                / raw_transitions,
                "actor_updates_per_transition": actor_updates
                / raw_transitions,
            }
        )
    finally:
        workflow.close()


if __name__ == "__main__":
    main()
