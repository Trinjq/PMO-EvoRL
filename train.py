"""Run the GPU-native PD-MORL MO-TD3 workflow."""

import argparse
import time
from pathlib import Path

def configure_update_schedule(config) -> None:
    """Derive integer TD3 loop counts from per-transition update ratios."""
    if (
        config.actor_updates_per_transition <= 0
        or config.critic_updates_per_transition <= 0
    ):
        raise ValueError("update ratios must be positive")
    actor_updates = config.num_envs * config.actor_updates_per_transition
    actor_interval = (
        config.critic_updates_per_transition
        / config.actor_updates_per_transition
    )
    if (
        not float(actor_updates).is_integer()
        or not float(actor_interval).is_integer()
    ):
        raise ValueError(
            "update ratios must produce integer actor updates and critic/actor interval"
        )
    config.num_updates_per_iter = int(actor_updates)
    config.actor_update_interval = int(actor_interval)


def main() -> None:
    import jax
    from evorl.recorders import LogRecorder
    from omegaconf import OmegaConf

    from pmo_evorl.workflow import MOTD3Workflow

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mo_td3_walker2d.yaml")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--interpolator-eval-episodes", type=int)
    parser.add_argument("--key-update-interval", type=int)
    parser.add_argument("--critic-updates-per-transition", type=float)
    parser.add_argument("--actor-updates-per-transition", type=float)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold-iters", type=int)
    parser.add_argument("--pareto-step-size", type=float)
    parser.add_argument("--pareto-eval-batch-size", type=int)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.num_envs is not None:
        if args.num_envs % config.num_preference_workers:
            parser.error("--num-envs must be divisible by num_preference_workers")
        config.num_envs = args.num_envs
        config.fold_iters = min(config.fold_iters, 100)
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.interpolator_eval_episodes is not None:
        config.interpolator_eval_episodes = args.interpolator_eval_episodes
    if args.key_update_interval is not None:
        config.key_update_interval = args.key_update_interval
    if args.critic_updates_per_transition is not None:
        config.critic_updates_per_transition = args.critic_updates_per_transition
    if args.actor_updates_per_transition is not None:
        config.actor_updates_per_transition = args.actor_updates_per_transition
    if args.eval_episodes is not None:
        config.eval_episodes = args.eval_episodes
    if args.seed is not None:
        config.seed = args.seed
    if args.fold_iters is not None:
        config.fold_iters = args.fold_iters
    if args.pareto_step_size is not None:
        config.pareto_step_size = args.pareto_step_size
    if args.pareto_eval_batch_size is not None:
        config.pareto_eval_batch_size = args.pareto_eval_batch_size
    if config.key_update_interval < 1:
        parser.error("--key-update-interval must be at least 1")
    try:
        configure_update_schedule(config)
    except ValueError as error:
        parser.error(str(error))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU training requires the JAX GPU backend")

    build_started = time.perf_counter()
    workflow = MOTD3Workflow.build_from_config(config, enable_jit=True)
    print({"build_seconds": time.perf_counter() - build_started}, flush=True)
    workflow.add_recorders(
        [LogRecorder(str(output_dir / "training.log"), console=True)]
    )
    try:
        init_started = time.perf_counter()
        state = workflow.init(jax.random.PRNGKey(config.seed))
        jax.block_until_ready(state.metrics.sampled_timesteps)
        print({"init_seconds": time.perf_counter() - init_started}, flush=True)
        workflow.learn(state)
    finally:
        workflow.close()


if __name__ == "__main__":
    main()
