"""Reproduce the three fixed-preference PD-MORL Walker2d Key objectives."""

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from pmo_evorl.interpolator import KEY_PREFERENCES
from pmo_evorl.key_training import KeyTD3Workflow


ORIGINAL_OBJECTIVES = jnp.asarray(
    [
        [497.2413299560547, 2494.5884033203124],
        [1639.5962036132812, 2156.5128173828125],
        [2602.103564453125, 691.5010070800781],
    ],
    dtype=jnp.float32,
)


def make_config(preference, output_dir, total_timesteps, fold_iters):
    config = OmegaConf.load("configs/mo_td3_walker2d.yaml")
    config.num_envs = 1
    config.num_eval_envs = 10
    config.num_preference_workers = 1
    config.rollout_length = 1
    config.batch_size = 100
    config.replay_size = 500_000
    config.replay_buffer_capacity = 500_000
    config.random_timesteps = 200
    config.learning_start_timesteps = 200
    config.start_timesteps = 25_000
    config.learning_rate = 3e-4
    config.grad_clip_norm = 100.0
    config.policy_freq = 2
    config.discount = 0.99
    config.exploration_epsilon = 0.1
    config.policy_noise = 0.2
    config.clip_policy_noise = 0.5
    config.tau = 0.005
    config.max_episode_len = 500
    config.eval_episodes = 10
    config.total_timesteps = total_timesteps
    config.fold_iters = fold_iters
    config.preference = list(map(float, preference))
    config.output_dir = str(output_dir)
    config.save_replay_buffer = False
    config.eval_interval = 10_000_000
    config.checkpoint.enable = False
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--fold-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    preference = KEY_PREFERENCES[args.key_index]
    seed = args.key_index if args.seed is None else args.seed
    config = make_config(
        preference, args.output_dir, args.total_timesteps, args.fold_iters
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow = KeyTD3Workflow.build_from_config(config, enable_jit=True)
    started = time.perf_counter()
    try:
        state = workflow.init(jax.random.PRNGKey(seed))
        initial_timesteps = int(jax.device_get(state.metrics.sampled_timesteps))
        remaining = max(args.total_timesteps - initial_timesteps, 0)
        full_folds, tail_steps = divmod(remaining, args.fold_iters)
        history = []
        best_score = -math.inf
        best_objective = None
        next_eval_episode = 100

        def record_evaluation():
            nonlocal best_score, best_objective, next_eval_episode
            objective = workflow.evaluate_fixed(state.agent_state)
            objective = jax.device_get(objective)
            score = float(jnp.dot(preference, objective))
            objective_list = [float(value) for value in objective]
            improved = score >= best_score
            if improved:
                best_score = score
                best_objective = objective_list
            history.append(
                {
                    "completed_episodes": int(
                        jax.device_get(state.metrics.sampled_episodes)
                    ),
                    "scalarized_return": score,
                    "objective": objective_list,
                    "best": improved,
                }
            )
            next_eval_episode += 100

        def maybe_evaluate():
            nonlocal next_eval_episode
            episodes = int(jax.device_get(state.metrics.sampled_episodes))
            while episodes >= next_eval_episode:
                record_evaluation()

        for _ in range(full_folds):
            _, state = workflow._multi_steps(state)
            maybe_evaluate()

        if tail_steps:
            def run_tail(current_state):
                def one_step(carry, _):
                    _, next_state = workflow.step(carry)
                    return next_state, None

                return jax.lax.scan(one_step, current_state, None, length=tail_steps)[0]

            state = jax.jit(run_tail)(state)
            maybe_evaluate()

        final_objective = jax.device_get(workflow.evaluate_fixed(state.agent_state))
        final_score = float(jnp.dot(preference, final_objective))
        if final_score >= best_score:
            best_score = final_score
            best_objective = [float(value) for value in final_objective]
            history.append(
                {
                    "completed_episodes": int(
                        jax.device_get(state.metrics.sampled_episodes)
                    ),
                    "scalarized_return": final_score,
                    "objective": best_objective,
                    "best": True,
                    "final_evaluation": True,
                }
            )

        result = {
            "key_index": args.key_index,
            "preference": [float(value) for value in preference],
            "seed": seed,
            "initial_timesteps": initial_timesteps,
            "final_timesteps": int(jax.device_get(state.metrics.sampled_timesteps)),
            "final_iterations": int(jax.device_get(state.metrics.iterations)),
            "completed_episodes": int(jax.device_get(state.metrics.sampled_episodes)),
            "best_scalarized_return": best_score,
            "best_objective": best_objective,
            "final_objective": [float(value) for value in final_objective],
            "history": history,
            "training_seconds": time.perf_counter() - started,
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
    finally:
        workflow.close()


if __name__ == "__main__":
    main()
