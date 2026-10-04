"""Measure steady-state training throughput without changing update ratios."""

import argparse
import time

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from pmo_evorl.workflow import MOTD3Workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/mo_td3_walker2d.yaml"
    )
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=100_000)
    parser.add_argument("--key-eval", action="store_true")
    parser.add_argument("--pareto-eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--pareto-step-size", type=float, default=0.005)
    parser.add_argument("--pareto-eval-batch-size", type=int)
    args = parser.parse_args()
    if args.key_eval and args.pareto_eval:
        parser.error("choose only one evaluation mode")

    config = OmegaConf.load(args.config)
    if args.num_envs % 10:
        parser.error("--num-envs must be divisible by 10")
    config.num_envs = args.num_envs
    config.num_updates_per_iter = args.num_envs // 10
    config.fold_iters = args.steps
    config.replay_buffer_capacity = args.capacity
    config.checkpoint.enable = False
    if not args.key_eval:
        config.interpolator_eval_episodes = 1
    config.eval_episodes = args.eval_episodes
    config.pareto_step_size = args.pareto_step_size
    if args.pareto_eval_batch_size is not None:
        config.pareto_eval_batch_size = args.pareto_eval_batch_size

    build_started = time.perf_counter()
    workflow = MOTD3Workflow.build_from_config(config, enable_jit=True)
    build_seconds = time.perf_counter() - build_started
    try:
        init_started = time.perf_counter()
        state = workflow.init(jax.random.PRNGKey(config.seed))
        jax.block_until_ready(state.metrics.sampled_timesteps)
        init_seconds = time.perf_counter() - init_started
        if args.key_eval:
            state = workflow._update_interpolator(state, jnp.uint32(2))
            jax.block_until_ready(
                state.agent_state.extra_state.projected_key_values
            )
            started = time.perf_counter()
            state = workflow._update_interpolator(state, jnp.uint32(3))
            jax.block_until_ready(
                state.agent_state.extra_state.projected_key_values
            )
            print({"key_evaluation_seconds": time.perf_counter() - started})
            return
        if args.pareto_eval:
            warmup_started = time.perf_counter()
            metrics, state = workflow.evaluate(state)
            jax.block_until_ready(metrics.hypervolume)
            warmup_seconds = time.perf_counter() - warmup_started
            started = time.perf_counter()
            metrics, state = workflow.evaluate(state)
            jax.block_until_ready(metrics.hypervolume)
            print(
                {
                    "pareto_evaluation_seconds": time.perf_counter() - started,
                    "preferences": round(1 / args.pareto_step_size) + 1,
                    "episodes_per_preference": args.eval_episodes,
                    "pareto_eval_batch_size": config.pareto_eval_batch_size,
                    "build_seconds": build_seconds,
                    "init_seconds": init_seconds,
                    "first_evaluation_seconds": warmup_seconds,
                    **metrics.to_local_dict(),
                },
                flush=True,
            )
            return
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
                "replay_buffer_capacity": args.capacity,
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
