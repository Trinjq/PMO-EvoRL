"""One-iteration GPU smoke check for the complete MO-TD3 workflow."""

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from pmo_evorl.workflow import MOTD3Workflow


def main() -> None:
    config = OmegaConf.create(
        {
            "num_envs": 10,
            "num_eval_envs": 10,
            "num_preference_workers": 10,
            "discount": 0.995,
            "exploration_epsilon": 0.1,
            "policy_noise": 0.2,
            "clip_policy_noise": 0.5,
            "angle_coefficient": 10.0,
            "optimizer": {"lr": 3e-4},
            "replay_buffer_capacity": 256,
            "batch_size": 8,
            "learning_start_timesteps": 20,
            "random_timesteps": 0,
            "her_start_timesteps": 100,
            "her_weight_num": 3,
            "seed": 1,
            "rollout_length": 1,
            "actor_update_interval": 10,
            "num_updates_per_iter": 1,
            "tau": 0.005,
            "output_dir": "",
            "checkpoint": {"enable": False},
            "save_replay_buffer": False,
            "fold_iters": 1,
            "total_timesteps": 30,
            "eval_interval": 100,
            "eval_episodes": 10,
        }
    )
    workflow = MOTD3Workflow.build_from_config(config, enable_jit=True)
    state = workflow.init(jax.random.PRNGKey(0))
    assert int(state.replay_buffer_state.buffer_size) == 20

    metrics, state = workflow.step(state)
    assert int(state.metrics.sampled_timesteps) == 30
    assert int(state.replay_buffer_state.buffer_size) == 30
    assert bool(jnp.isfinite(metrics.critic_loss))
    assert bool(jnp.isfinite(metrics.actor_loss))
    print(
        "MO-TD3 workflow check passed:",
        float(metrics.critic_loss),
        float(metrics.actor_loss),
    )


if __name__ == "__main__":
    main()
