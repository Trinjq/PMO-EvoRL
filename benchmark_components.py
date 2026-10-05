"""Measure the JAX/MJX stages used by the PMO-EvoRL learner."""

import argparse
import json
import statistics
import subprocess
import time
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jax
import jax.tree_util as jtu
import optax
from evorl.algorithms.offpolicy_utils import clean_trajectory
from evorl.replay_buffers import ReplayBuffer
from evorl.rollout import rollout
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory
from omegaconf import OmegaConf

from pmo_evorl.interpolator import evaluate_linear_rbf
from pmo_evorl.mo_td3 import LazyPreferenceHERReplayBuffer, PreferenceHERReplayBuffer
from pmo_evorl.workflow import MOTD3Workflow
from train import configure_update_schedule


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def _ready(value):
    return jax.tree_util.tree_map(jax.block_until_ready, value)


def _timed(call, ready, repeats: int, trace_dir: Path | None = None) -> list[float]:
    value = call()
    _ready(ready(value))
    samples = []
    for repeat in range(repeats):
        context = (
            jax.profiler.trace(str(trace_dir))
            if trace_dir is not None and repeat == 0
            else nullcontext()
        )
        with context:
            started = time.perf_counter()
            value = call()
            _ready(ready(value))
        samples.append(time.perf_counter() - started)
    return samples


def _summary(samples: list[float]) -> dict[str, object]:
    return {"seconds": samples, "median_seconds": statistics.median(samples)}


def _tree_nbytes(value) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(value)
        if hasattr(leaf, "size") and hasattr(leaf, "dtype")
    )


def _replay_buffer(config, buffer_type, spec):
    kwargs = {
        "capacity": config.replay_buffer_capacity,
        "min_sample_timesteps": max(
            config.batch_size, config.learning_start_timesteps
        ),
        "sample_batch_size": config.batch_size,
    }
    if buffer_type is not ReplayBuffer:
        kwargs.update(
            her_start_timesteps=config.her_start_timesteps,
            weight_num=config.her_weight_num,
            seed=config.seed,
        )
    replay = buffer_type(**kwargs)
    return replay, replay.init(spec)


def _gradient_step(workflow, state, batch, kind):
    params_name = f"{kind}_params"
    opt_state = getattr(state.opt_state, kind)

    def loss_fn(params):
        params_state = state.agent_state.params.replace(**{params_name: params})
        agent_state = state.agent_state.replace(params=params_state)
        loss_dict = getattr(workflow.agent, f"{kind}_loss")(
            agent_state, batch, jax.random.PRNGKey(17)
        )
        return getattr(loss_dict, f"{kind}_loss"), loss_dict

    (loss, loss_dict), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
        getattr(state.agent_state.params, params_name)
    )
    updates, opt_state = workflow.optimizer.update(
        gradients, opt_state, getattr(state.agent_state.params, params_name)
    )
    params = optax.apply_updates(
        getattr(state.agent_state.params, params_name), updates
    )
    return loss, loss_dict, params, opt_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mo_td3_walker2d.yaml")
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--her-start-timesteps", type=int)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or args.capacity < 1 or args.repeats < 1:
        parser.error("num-envs, steps, capacity, and repeats must be positive")
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    if args.num_envs % config.num_preference_workers:
        parser.error("--num-envs must be divisible by num_preference_workers")
    config.num_envs = args.num_envs
    configure_update_schedule(config)
    config.fold_iters = args.steps
    config.replay_buffer_capacity = args.capacity
    config.checkpoint.enable = False
    if args.her_start_timesteps is not None:
        config.her_start_timesteps = args.her_start_timesteps
    if jax.default_backend() != "gpu":
        raise RuntimeError("component benchmark requires the JAX GPU backend")

    workflow = MOTD3Workflow.build_from_config(config, enable_jit=True)
    try:
        state = workflow.init(jax.random.PRNGKey(config.seed))
        _ready(state.metrics.sampled_timesteps)
        trajectory, _ = rollout(
            env_fn=workflow.env.step,
            action_fn=workflow.agent.compute_actions,
            env_state=state.env_state,
            agent_state=state.agent_state,
            key=jax.random.PRNGKey(1),
            rollout_length=args.steps,
            env_extra_fields=("ori_obs", "termination"),
        )
        trajectory = tree_stop_gradient(
            flatten_rollout_trajectory(clean_trajectory(trajectory))
        )
        spec = jtu.tree_map(lambda value: value[0], trajectory)
        raw_buffer, raw_state = _replay_buffer(config, ReplayBuffer, spec)
        eager_buffer, eager_state = _replay_buffer(
            config, PreferenceHERReplayBuffer, spec
        )
        lazy_buffer, lazy_state = _replay_buffer(
            config, LazyPreferenceHERReplayBuffer, spec
        )
        raw_add_fn = jax.jit(raw_buffer.add)
        eager_add_fn = jax.jit(eager_buffer.add)
        lazy_add_fn = jax.jit(lazy_buffer.add)

        raw_state = raw_add_fn(raw_state, trajectory)
        eager_state = eager_add_fn(eager_state, trajectory)
        lazy_state = lazy_add_fn(lazy_state, trajectory)
        _ready((raw_state, eager_state, lazy_state))

        active_buffer = workflow.replay_buffer
        active_state = workflow.replay_buffer.add(
            state.replay_buffer_state, trajectory
        )
        _ready(active_state)
        active_sample_fn = jax.jit(active_buffer.sample)
        sample_batch = active_sample_fn(active_state, jax.random.PRNGKey(3))
        _ready(sample_batch)

        rollout_fn = jax.jit(
            lambda env_state: rollout(
                env_fn=workflow.env.step,
                action_fn=workflow.agent.compute_actions,
                env_state=env_state,
                agent_state=state.agent_state,
                key=jax.random.PRNGKey(4),
                rollout_length=args.steps,
                env_extra_fields=("ori_obs", "termination"),
            )
        )
        sample_fn = jax.jit(
            lambda: workflow.agent.critic_loss(
                state.agent_state, sample_batch, jax.random.PRNGKey(6)
            )
        )
        critic_update_fn = jax.jit(
            lambda: _gradient_step(workflow, state, sample_batch, "critic")
        )
        actor_update_fn = jax.jit(
            lambda: _gradient_step(workflow, state, sample_batch, "actor")
        )
        rbf_fn = jax.jit(
            lambda: evaluate_linear_rbf(
                sample_batch.obs.preference,
                state.agent_state.extra_state.rbf_coefficients,
            )
        )
        key_eval_fn = jax.jit(
            lambda: workflow._evaluate_key_objectives(state.agent_state)
        )
        full_iteration_fn = jax.jit(lambda value: workflow.step(value))

        stage_calls = {
            "rollout": (
                lambda: rollout_fn(state.env_state),
                lambda value: value[1].done,
            ),
            "replay_add": (
                lambda: raw_add_fn(raw_state, trajectory),
                lambda value: value.buffer_size,
            ),
            "replay_add_her": (
                lambda: eager_add_fn(eager_state, trajectory),
                lambda value: value.buffer_size,
            ),
            "replay_add_lazy_her": (
                lambda: lazy_add_fn(lazy_state, trajectory),
                lambda value: value.buffer_size,
            ),
            "replay_sample": (
                lambda: active_sample_fn(
                    active_state, jax.random.PRNGKey(5)
                ),
                lambda value: value.obs.state,
            ),
            "critic_forward": (
                sample_fn,
                lambda value: value.critic_loss,
            ),
            "critic_update": (
                critic_update_fn,
                lambda value: value[0],
            ),
            "actor_update": (
                actor_update_fn,
                lambda value: value[0],
            ),
            "rbf_projection": (
                rbf_fn,
                lambda value: value,
            ),
            "key_eval": (
                key_eval_fn,
                lambda value: value,
            ),
            "full_iteration": (
                lambda: full_iteration_fn(state),
                lambda value: value[1].metrics.iterations,
            ),
        }
        stages = {}
        for name, (call, ready) in stage_calls.items():
            trace_dir = args.profile_dir / name if args.profile_dir else None
            stages[name] = _summary(_timed(call, ready, args.repeats, trace_dir))
        print(
            json.dumps(
                {
                    "num_envs": args.num_envs,
                    "steps": args.steps,
                    "capacity": args.capacity,
                    "repeats": args.repeats,
                    "device": str(jax.devices()[0]),
                    "git_commit": _commit(),
                    "git_dirty": bool(
                        subprocess.run(
                            ["git", "status", "--porcelain"],
                            capture_output=True,
                            check=False,
                            text=True,
                        ).stdout.strip()
                    ),
                    "versions": {
                        name: _version(name)
                        for name in ("jax", "mujoco", "mujoco-mjx", "evorl", "optax")
                    },
                    "buffer_sizes": {
                        "raw": int(raw_state.buffer_size),
                        "eager_her": int(eager_state.buffer_size),
                        "lazy_her": int(lazy_state.buffer_size),
                    },
                    "buffer_bytes": {
                        "raw": _tree_nbytes(raw_state.data),
                        "eager_her": _tree_nbytes(eager_state.data),
                        "lazy_her": _tree_nbytes(lazy_state.data),
                    },
                    "stages": stages,
                },
                indent=2,
            )
        )
    finally:
        workflow.close()


if __name__ == "__main__":
    main()
