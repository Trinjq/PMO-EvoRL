"""EvoRL workflow wiring for continuous PD-MORL."""

import jax
import jax.numpy as jnp
import optax

from evorl.algorithms.td3 import TD3Workflow
from evorl.evaluators import Evaluator
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict

from pmo_evorl.envs import create_mo_walker2d_env
from pmo_evorl.interpolator import KEY_PREFERENCES, update_key_objectives
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
            her_start_timesteps=config.her_start_timesteps,
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
        workflow = cls(env, agent, optimizer, evaluator, replay_buffer, config)
        key_preferences = jnp.repeat(
            KEY_PREFERENCES, config.interpolator_eval_episodes, axis=0
        )

        key_env = create_mo_walker2d_env(
            len(key_preferences),
            num_preference_workers=3,
            autoreset=False,
        )
        workflow.key_eval_env = key_env
        workflow.key_preferences = key_preferences
        return workflow

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

    def step(self, state):
        train_metrics, state = super().step(state)
        completed = state.env_state.info.episode_count.min()
        previous = state.agent_state.extra_state.interpolator_updates

        def update_interpolator(value):
            candidate = self._evaluate_key_objectives(value.agent_state)
            raw, projected = update_key_objectives(
                value.agent_state.extra_state.key_objectives, candidate
            )
            extra_state = value.agent_state.extra_state.replace(
                key_objectives=raw,
                projected_key_values=projected,
                interpolator_updates=completed,
            )
            return value.replace(
                agent_state=value.agent_state.replace(extra_state=extra_state)
            )

        state = jax.lax.cond(
            completed > previous, update_interpolator, lambda value: value, state
        )
        return train_metrics, state

    def _evaluate_key_objectives(self, agent_state):
        env_state = self.key_eval_env.reset(jax.random.PRNGKey(0))
        returns = jnp.zeros((len(self.key_preferences), 2))
        finished = jnp.zeros(len(self.key_preferences), dtype=bool)

        def evaluate_step(carry, key):
            env_state, returns, finished = carry
            obs = env_state.obs.replace(preference=self.key_preferences)
            actions, _ = self.agent.evaluate_actions(
                agent_state, SampleBatch(obs=obs), key
            )
            next_state = self.key_eval_env.step(env_state, actions)
            returns += (~finished)[:, None] * next_state.reward
            finished |= next_state.done.astype(bool)
            return (next_state, returns, finished), None

        keys = jax.random.split(jax.random.PRNGKey(1), 500)
        (_, returns, _), _ = jax.lax.scan(
            evaluate_step, (env_state, returns, finished), keys
        )
        return returns.reshape(
            3, self.config.interpolator_eval_episodes, 2
        ).mean(axis=1)
