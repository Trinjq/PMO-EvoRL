"""Measure synchronized PMO-EvoRL component and workflow performance."""

import argparse
import json
import statistics
import subprocess
import time
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from pmo_evorl.workflow import MOTD3Workflow
from train import configure_update_schedule


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        check=False,
        text=True,
    )
    return bool(result.stdout.strip())


def _timed(call, ready, trace_dir: Path | None = None):
    context = jax.profiler.trace(str(trace_dir)) if trace_dir else nullcontext()
    with context:
        started = time.perf_counter()
        value = call()
        jax.block_until_ready(ready(value))
        elapsed = time.perf_counter() - started
    return value, elapsed


def _summary(samples: list[float]) -> dict[str, object]:
    return {"seconds": samples, "median_seconds": statistics.median(samples)}


def _tree_nbytes(value) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(value)
        if hasattr(leaf, "size") and hasattr(leaf, "dtype")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mo_td3_walker2d.yaml")
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=2_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=(
            "full",
            "train",
            "env-only",
            "learner-only",
            "key-eval",
            "pareto-eval",
        ),
        default="full",
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--key-eval", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pareto-eval", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--pareto-step-size", type=float, default=0.005)
    parser.add_argument("--pareto-eval-batch-size", type=int)
    parser.add_argument("--critic-updates-per-transition", type=float)
    parser.add_argument("--actor-updates-per-transition", type=float)
    parser.add_argument("--lazy-preference-her", action="store_true", default=None)
    parser.add_argument("--sample-many", action="store_true", default=None)
    args = parser.parse_args()
    if args.key_eval and args.pareto_eval:
        parser.error("choose only one legacy evaluation flag")
    if args.key_eval:
        args.mode = "key-eval"
    elif args.pareto_eval:
        args.mode = "pareto-eval"
    if args.repeats < 1 or args.steps < 1:
        parser.error("--repeats and --steps must be positive")

    config = OmegaConf.load(args.config)
    if args.num_envs % config.num_preference_workers:
        parser.error("--num-envs must be divisible by num_preference_workers")
    config.num_envs = args.num_envs
    if args.critic_updates_per_transition is not None:
        config.critic_updates_per_transition = args.critic_updates_per_transition
    if args.actor_updates_per_transition is not None:
        config.actor_updates_per_transition = args.actor_updates_per_transition
    if args.lazy_preference_her is not None:
        config.lazy_preference_her = args.lazy_preference_her
    if args.sample_many is not None:
        config.sample_many = args.sample_many
    configure_update_schedule(config)
    config.fold_iters = args.steps
    config.replay_buffer_capacity = args.capacity
    config.checkpoint.enable = False
    config.eval_episodes = args.eval_episodes
    config.pareto_step_size = args.pareto_step_size
    if args.pareto_eval_batch_size is not None:
        config.pareto_eval_batch_size = args.pareto_eval_batch_size
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU benchmark requires the JAX GPU backend")

    build_started = time.perf_counter()
    workflow = MOTD3Workflow.build_from_config(config, enable_jit=True)
    build_seconds = time.perf_counter() - build_started
    try:
        init_started = time.perf_counter()
        state = workflow.init(jax.random.PRNGKey(config.seed))
        jax.block_until_ready(state.metrics.sampled_timesteps)
        init_seconds = time.perf_counter() - init_started
        initial_buffer_size = int(state.replay_buffer_state.buffer_size)
        common = {
            "mode": args.mode,
            "num_envs": args.num_envs,
            "steps_per_repeat": args.steps,
            "repeats": args.repeats,
            "replay_buffer_capacity": args.capacity,
            "initial_buffer_size": initial_buffer_size,
            "build_seconds": build_seconds,
            "init_seconds": init_seconds,
            "device": str(jax.devices()[0]),
            "replay_buffer_bytes": _tree_nbytes(state.replay_buffer_state.data),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "interpolator_eval_episodes": config.interpolator_eval_episodes,
            "key_update_interval": config.key_update_interval,
            "lazy_preference_her": bool(config.get("lazy_preference_her", False)),
            "sample_many": bool(config.get("sample_many", False)),
            "versions": {
                name: _package_version(name)
                for name in ("jax", "mujoco", "mujoco-mjx", "evorl", "optax")
            },
        }

        if args.mode == "key-eval":
            state = workflow._update_interpolator(state, jnp.uint32(2))
            jax.block_until_ready(state.agent_state.extra_state.rbf_coefficients)
            samples = []
            for repeat in range(args.repeats):
                state, elapsed = _timed(
                    lambda repeat=repeat: workflow._update_interpolator(
                        state, jnp.uint32(repeat + 3)
                    ),
                    lambda value: value.agent_state.extra_state.rbf_coefficients,
                    args.profile_dir if repeat == 0 else None,
                )
                samples.append(elapsed)
            print(json.dumps({**common, **_summary(samples)}, indent=2))
            return

        if args.mode == "pareto-eval":
            metrics, state = workflow.evaluate(state)
            jax.block_until_ready(metrics.hypervolume)
            samples = []
            for repeat in range(args.repeats):
                (metrics, state), elapsed = _timed(
                    lambda: workflow.evaluate(state),
                    lambda value: value[0].hypervolume,
                    args.profile_dir if repeat == 0 else None,
                )
                samples.append(elapsed)
            print(
                json.dumps(
                    {
                        **common,
                        **_summary(samples),
                        "preferences": round(1 / args.pareto_step_size) + 1,
                        "episodes_per_preference": args.eval_episodes,
                        "pareto_eval_batch_size": config.pareto_eval_batch_size,
                        **metrics.to_local_dict(),
                    },
                    indent=2,
                    default=float,
                )
            )
            return

        actions = jnp.zeros((args.num_envs, workflow.env.action_space.shape[0]))

        def env_steps(env_state):
            return jax.lax.scan(
                lambda carry, _: (workflow.env.step(carry, actions), None),
                env_state,
                (),
                length=args.steps,
            )[0]

        env_steps = jax.jit(env_steps)

        if args.mode == "env-only":
            env_state = env_steps(state.env_state)
            jax.block_until_ready(env_state.done)
            samples = []
            for repeat in range(args.repeats):
                env_state, elapsed = _timed(
                    lambda: env_steps(env_state),
                    lambda value: value.done,
                    args.profile_dir if repeat == 0 else None,
                )
                samples.append(elapsed)
            median = statistics.median(samples)
            print(
                json.dumps(
                    {
                        **common,
                        **_summary(samples),
                        "raw_transitions_per_second": (
                            args.steps * args.num_envs / median
                        ),
                    },
                    indent=2,
                )
            )
            return

        if args.mode == "full":
            def run_full(completed):
                _, next_state = workflow._multi_steps(state)
                return workflow._update_interpolator(next_state, completed)

            state = run_full(jnp.uint32(2))
            jax.block_until_ready(state.agent_state.extra_state.rbf_coefficients)
            samples = []
            for repeat in range(args.repeats):
                state, elapsed = _timed(
                    lambda repeat=repeat: run_full(jnp.uint32(repeat + 3)),
                    lambda value: value.agent_state.extra_state.rbf_coefficients,
                    args.profile_dir if repeat == 0 else None,
                )
                samples.append(elapsed)
        else:
            _, state = workflow._multi_steps(state)
            jax.block_until_ready(state.metrics.iterations)
            train_samples = []
            for repeat in range(args.repeats):
                (_, state), elapsed = _timed(
                    lambda: workflow._multi_steps(state),
                    lambda value: value[1].metrics.iterations,
                    args.profile_dir if repeat == 0 else None,
                )
                train_samples.append(elapsed)
            samples = train_samples

        if args.mode == "learner-only":
            env_state = env_steps(workflow.env.reset(jax.random.PRNGKey(991)))
            jax.block_until_ready(env_state.done)
            env_samples = []
            for _ in range(args.repeats):
                env_state, elapsed = _timed(
                    lambda: env_steps(env_state), lambda value: value.done
                )
                env_samples.append(elapsed)
            estimates = [
                max(train - env, 0.0)
                for train, env in zip(train_samples, env_samples)
            ]
            print(
                json.dumps(
                    {
                        **common,
                        "method": (
                            "train path minus env-only; includes policy "
                            "and replay overhead"
                        ),
                        "train_seconds": train_samples,
                        "env_seconds": env_samples,
                        **_summary(estimates),
                    },
                    indent=2,
                )
            )
            return

        median = statistics.median(samples)
        raw_transitions = args.steps * args.num_envs * config.rollout_length
        critic_updates = (
            args.steps
            * config.num_updates_per_iter
            * config.actor_update_interval
        )
        actor_updates = args.steps * config.num_updates_per_iter
        print(
            json.dumps(
                {
                    **common,
                    **_summary(samples),
                    "final_buffer_size": int(state.replay_buffer_state.buffer_size),
                    "key_evaluations_per_repeat": int(args.mode == "full"),
                    "raw_transitions_per_second": raw_transitions / median,
                    "critic_updates_per_second": critic_updates / median,
                    "actor_updates_per_second": actor_updates / median,
                    "critic_updates_per_transition": critic_updates
                    / raw_transitions,
                    "actor_updates_per_transition": actor_updates
                    / raw_transitions,
                },
                indent=2,
            )
        )
    finally:
        workflow.close()


if __name__ == "__main__":
    main()
