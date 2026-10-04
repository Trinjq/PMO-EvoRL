"""EvoRL workflow wiring for continuous PD-MORL."""

import jax.numpy as jnp
import optax

from evorl.algorithms.td3 import TD3Workflow
from evorl.evaluators import Evaluator
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict

from pmo_evorl.envs import create_mo_walker2d_env
from pmo_evorl.mo_td3 import MOTD3Agent, PreferenceHERReplayBuffer
from pmo_evorl.networks import PreferenceActor, TwinVectorCritic


class MOTD3Workflow(TD3Workflow):
    """Stock EvoRL TD3 loop with PD-MORL environment, agent, and HER."""

    @classmethod
    def name(cls):
        return "PD-MORL-MO-TD3-HER"

    @classmethod
    def _build_from_config(cls, config):
        env = create_mo_walker2d_env(
            config.num_envs, config.num_preference_workers
        )
        agent = MOTD3Agent(
            actor_network=PreferenceActor(action_size=env.action_space.shape[0]),
            critic_network=TwinVectorCritic(reward_size=2),
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            angle_coefficient=config.angle_coefficient,
        )
        optimizer = optax.adam(config.optimizer.lr)
        replay_buffer = PreferenceHERReplayBuffer(
            capacity=config.replay_buffer_capacity,
            min_sample_timesteps=max(
                config.batch_size, config.learning_start_timesteps
            ),
            sample_batch_size=config.batch_size,
            learning_start_timesteps=config.learning_start_timesteps,
            weight_num=config.her_weight_num,
            seed=config.seed,
        )
        eval_env = create_mo_walker2d_env(
            config.num_eval_envs,
            config.num_preference_workers,
            autoreset=False,
        )
        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=500,
        )
        return cls(env, agent, optimizer, evaluator, replay_buffer, config)

    def _setup_replaybuffer(self, key):
        dummy_obs = self.env.obs_space.sample(key)
        dummy = SampleBatch(
            obs=dummy_obs,
            actions=jnp.zeros(self.env.action_space.shape),
            rewards=jnp.zeros(2),
            extras=PyTreeDict(
                policy_extras=PyTreeDict(),
                env_extras=PyTreeDict(
                    ori_obs=dummy_obs.state,
                    termination=jnp.zeros(()),
                ),
            ),
        )
        return self.replay_buffer.init(dummy)
