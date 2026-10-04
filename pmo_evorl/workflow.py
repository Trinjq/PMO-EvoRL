"""EvoRL workflow wiring for continuous PD-MORL."""

import jax
import jax.numpy as jnp
import optax
import chex

from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state
from evorl.algorithms.td3 import TD3Workflow
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase
from evorl.recorders import add_prefix
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict

from pmo_evorl.envs import create_mo_walker2d_env
from pmo_evorl.interpolator import KEY_PREFERENCES, update_key_objectives
from pmo_evorl.metrics import hypervolume_2d, sparsity
from pmo_evorl.mo_td3 import MOTD3Agent, PreferenceHERReplayBuffer
from pmo_evorl.networks import PreferenceActor, TwinVectorCritic


class MORLEvaluateMetric(MetricBase):
    hypervolume: chex.Array
    sparsity: chex.Array


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

        preference_count = round(1.0 / config.pareto_step_size) + 1
        pareto_preferences = jnp.stack(
            (
                jnp.linspace(0.0, 1.0, preference_count),
                jnp.linspace(1.0, 0.0, preference_count),
            ),
            axis=-1,
        )
        workflow.pareto_preferences = jnp.repeat(
            pareto_preferences, config.eval_episodes, axis=0
        )
        workflow.pareto_sample_count = len(workflow.pareto_preferences)
        batch_size = config.pareto_eval_batch_size
        padded_count = (
            (workflow.pareto_sample_count + batch_size - 1) // batch_size
        ) * batch_size
        workflow.pareto_preferences = jnp.pad(
            workflow.pareto_preferences,
            ((0, padded_count - workflow.pareto_sample_count), (0, 0)),
        ).reshape(-1, batch_size, 2)
        workflow.pareto_eval_env = create_mo_walker2d_env(
            batch_size,
            num_preference_workers=1,
            autoreset=False,
        )
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

    def _update_interpolator(self, state, completed):
        candidate = self._evaluate_key_objectives(state.agent_state)
        raw, projected = update_key_objectives(
            state.agent_state.extra_state.key_objectives, candidate
        )
        extra_state = state.agent_state.extra_state.replace(
            key_objectives=raw,
            projected_key_values=projected,
            interpolator_updates=completed,
        )
        return state.replace(
            agent_state=state.agent_state.replace(extra_state=extra_state)
        )

    def _maybe_update_interpolator(self, state):
        completed = state.env_state.info.episode_count.min().tolist()
        previous = state.agent_state.extra_state.interpolator_updates.tolist()
        if completed > previous:
            state = self._update_interpolator(
                state, jnp.asarray(completed, dtype=jnp.uint32)
            )
        return state

    def _evaluate_key_objectives(self, agent_state):
        returns = self._evaluate_preferences(
            agent_state,
            self.key_eval_env,
            self.key_preferences,
            jax.random.PRNGKey(0),
        )
        return returns.reshape(
            3, self.config.interpolator_eval_episodes, 2
        ).mean(axis=1)

    def _evaluate_preferences(self, agent_state, env, preferences, reset_key):
        env_state = env.reset(reset_key)
        returns = jnp.zeros((len(preferences), 2))
        finished = jnp.zeros(len(preferences), dtype=bool)

        def evaluate_step(carry, key):
            env_state, returns, finished = carry
            obs = env_state.obs.replace(preference=preferences)
            actions, _ = self.agent.evaluate_actions(
                agent_state, SampleBatch(obs=obs), key
            )
            next_state = env.step(env_state, actions)
            returns += (~finished)[:, None] * next_state.reward
            finished |= next_state.done.astype(bool)
            return (next_state, returns, finished), None

        keys = jax.random.split(jax.random.PRNGKey(1), 500)
        (_, returns, _), _ = jax.lax.scan(
            evaluate_step, (env_state, returns, finished), keys
        )
        return returns

    def evaluate(self, state):
        key, _ = jax.random.split(state.key)

        def evaluate_batch(_, preferences):
            returns = self._evaluate_preferences(
                state.agent_state,
                self.pareto_eval_env,
                preferences,
                jax.random.PRNGKey(11),
            )
            return None, returns

        _, returns = jax.lax.scan(
            evaluate_batch, None, self.pareto_preferences
        )
        returns = returns.reshape(-1, 2)[: self.pareto_sample_count]
        objectives = returns.reshape(-1, self.config.eval_episodes, 2).mean(1)
        metrics = MORLEvaluateMetric(
            hypervolume=hypervolume_2d(objectives),
            sparsity=sparsity(objectives),
        )
        return metrics, state.replace(key=key)

    def learn(self, state):
        """Train until the raw-transition budget and always finalize once."""
        next_eval = (
            state.metrics.iterations.tolist() // self.config.eval_interval + 1
        ) * self.config.eval_interval

        while state.metrics.sampled_timesteps.tolist() < self.config.total_timesteps:
            train_metrics, state = self._multi_steps(state)
            state = self._maybe_update_interpolator(state)
            iterations = state.metrics.iterations.tolist()
            is_final = (
                state.metrics.sampled_timesteps.tolist()
                >= self.config.total_timesteps
            )
            self.recorder.write(train_metrics.to_local_dict(), iterations)
            self.recorder.write(state.metrics.to_local_dict(), iterations)
            print(
                {
                    "iterations": iterations,
                    "raw_transitions": state.metrics.sampled_timesteps.tolist(),
                    "critic_loss": train_metrics.critic_loss.tolist(),
                    "actor_loss": train_metrics.actor_loss.tolist(),
                    "interpolator_updates": state.agent_state.extra_state.interpolator_updates.tolist(),
                },
                flush=True,
            )

            if iterations >= next_eval or is_final:
                eval_metrics, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
                )
                print(
                    {"iterations": iterations, **eval_metrics.to_local_dict()},
                    flush=True,
                )
                while next_eval <= iterations:
                    next_eval += self.config.eval_interval

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            self.checkpoint_manager.save(
                iterations, saved_state, force=is_final
            )

        return state

    @classmethod
    def enable_jit(cls):
        super().enable_jit()
        cls._multi_steps = jax.jit(
            cls._multi_steps, static_argnums=(0,), donate_argnums=(1,)
        )
        cls._update_interpolator = jax.jit(
            cls._update_interpolator, static_argnums=(0,)
        )
