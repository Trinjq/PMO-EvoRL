"""EvoRL workflow wiring for continuous PD-MORL."""

import time

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from evorl.algorithms.offpolicy_utils import clean_trajectory, skip_replay_buffer_state
from evorl.algorithms.td3 import TD3TrainMetric, TD3Workflow
from evorl.distributed import psum
from evorl.distributed.gradients import agent_gradient_update
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils import running_statistics
from evorl.utils.jax_utils import scan_and_mean, tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from pmo_evorl.envs import create_mo_walker2d_env
from pmo_evorl.interpolator import (
    KEY_PREFERENCES,
    fit_linear_rbf,
    update_key_objectives,
)
from pmo_evorl.metrics import hypervolume_2d, sparsity
from pmo_evorl.mo_td3 import (
    LazyPreferenceHERReplayBuffer,
    MOTD3Agent,
    PreferenceHERReplayBuffer,
)
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
        env = create_mo_walker2d_env(config.num_envs, config.num_preference_workers)
        agent = MOTD3Agent(
            actor_network=PreferenceActor(action_size=env.action_space.shape[0]),
            critic_network=TwinVectorCritic(reward_size=2),
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            angle_coefficient=config.angle_coefficient,
            logical_group_count=config.num_preference_workers,
            random_warmup_transitions=config.get(
                "random_warmup_transitions", 10_000
            ),
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
            optax.adam(config.optimizer.lr),
        )
        replay_buffer_type = (
            LazyPreferenceHERReplayBuffer
            if config.get("lazy_preference_her", False)
            else PreferenceHERReplayBuffer
        )
        replay_capacity = config.replay_buffer_capacity
        if replay_buffer_type is LazyPreferenceHERReplayBuffer:
            replay_capacity //= config.her_weight_num + 1
        replay_buffer = replay_buffer_type(
            capacity=replay_capacity,
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

    def step(self, state):
        if not self.config.get("sample_many", False):
            metrics, next_state = super().step(state)
        else:
            metrics, next_state = self._step_sample_many(state)
        return metrics, self._advance_logical_group_counters(state, next_state)

    def _advance_logical_group_counters(self, state, next_state):
        extra_state = state.agent_state.extra_state
        lanes_per_group = self.config.num_envs // self.config.num_preference_workers
        transitions = jnp.asarray(
            self.config.rollout_length * lanes_per_group, dtype=jnp.uint32
        )
        in_warmup = (
            extra_state.logical_group_transition_count
            < self.config.get("random_warmup_transitions", 10_000)
        )
        next_extra_state = next_state.agent_state.extra_state.replace(
            logical_group_transition_count=(
                extra_state.logical_group_transition_count + transitions
            ),
            logical_group_random_transition_count=(
                extra_state.logical_group_random_transition_count
                + jnp.where(in_warmup, transitions, 0)
            ),
            logical_group_policy_transition_count=(
                extra_state.logical_group_policy_transition_count
                + jnp.where(~in_warmup, transitions, 0)
            ),
        )
        return next_state.replace(
            agent_state=next_state.agent_state.replace(extra_state=next_extra_state)
        )

    def _step_sample_many(self, state):
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)
        trajectory, env_state = rollout(
            env_fn=self.env.step,
            action_fn=self.agent.compute_actions,
            env_state=state.env_state,
            agent_state=state.agent_state,
            key=rollout_key,
            rollout_length=self.config.rollout_length,
            env_extra_fields=("ori_obs", "termination"),
        )
        trajectory_dones = trajectory.dones
        trajectory = tree_stop_gradient(
            flatten_rollout_trajectory(clean_trajectory(trajectory))
        )

        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    dp_axis_name=self.dp_axis_name,
                )
            )

        replay_buffer_state = self.replay_buffer.add(
            state.replay_buffer_state, trajectory
        )

        def critic_loss_fn(agent_state, sample_batch, loss_key):
            loss_dict = self.agent.critic_loss(agent_state, sample_batch, loss_key)
            return loss_dict.critic_loss, loss_dict

        def actor_loss_fn(agent_state, sample_batch, loss_key):
            loss_dict = self.agent.actor_loss(agent_state, sample_batch, loss_key)
            return loss_dict.actor_loss, loss_dict

        critic_update_fn = agent_gradient_update(
            critic_loss_fn,
            self.optimizer,
            dp_axis_name=self.dp_axis_name,
            has_aux=True,
            attach_fn=lambda current, params: current.replace(
                params=current.params.replace(critic_params=params)
            ),
            detach_fn=lambda current: current.params.critic_params,
        )
        actor_update_fn = agent_gradient_update(
            actor_loss_fn,
            self.optimizer,
            dp_axis_name=self.dp_axis_name,
            has_aux=True,
            attach_fn=lambda current, params: current.replace(
                params=current.params.replace(actor_params=params)
            ),
            detach_fn=lambda current: current.params.actor_params,
        )

        num_actor_updates = self.config.num_updates_per_iter
        critic_updates_per_actor = self.config.actor_update_interval
        sample_key, update_key = jax.random.split(learn_key)
        sample_batches = self.replay_buffer.sample_many(
            replay_buffer_state,
            sample_key,
            num_actor_updates * critic_updates_per_actor,
        )
        sample_batches = jtu.tree_map(
            lambda value: value.reshape(
                (num_actor_updates, critic_updates_per_actor) + value.shape[1:]
            ),
            sample_batches,
        )

        def update_actor_step(carry, batches):
            key, current_agent, opt_state = carry
            critic_opt_state = opt_state.critic
            actor_opt_state = opt_state.actor

            def update_critic(carry, sample_batch):
                key, current_agent, critic_opt_state = carry
                key, loss_key = jax.random.split(key)
                (_, _), current_agent, critic_opt_state = (
                    critic_update_fn(
                        critic_opt_state,
                        current_agent,
                        sample_batch,
                        loss_key,
                    )
                )
                return (key, current_agent, critic_opt_state), None

            key, critic_key, actor_key = jax.random.split(key, num=3)
            if critic_updates_per_actor > 1:
                (
                    (_, current_agent, critic_opt_state),
                    _,
                ) = jax.lax.scan(
                    update_critic,
                    (critic_key, current_agent, critic_opt_state),
                    jtu.tree_map(lambda value: value[:-1], batches),
                )

            actor_batch = jtu.tree_map(lambda value: value[-1], batches)
            (critic_loss, critic_loss_dict), current_agent, critic_opt_state = (
                critic_update_fn(
                    critic_opt_state,
                    current_agent,
                    actor_batch,
                    critic_key,
                )
            )
            (actor_loss, actor_loss_dict), current_agent, actor_opt_state = (
                actor_update_fn(
                    actor_opt_state,
                    current_agent,
                    actor_batch,
                    actor_key,
                )
            )

            target_actor_params = soft_target_update(
                current_agent.params.target_actor_params,
                current_agent.params.actor_params,
                self.config.tau,
            )
            target_critic_params = soft_target_update(
                current_agent.params.target_critic_params,
                current_agent.params.critic_params,
                self.config.tau,
            )
            current_agent = current_agent.replace(
                params=current_agent.params.replace(
                    target_actor_params=target_actor_params,
                    target_critic_params=target_critic_params,
                )
            )
            return (
                (key, current_agent, opt_state.replace(
                    actor=actor_opt_state, critic=critic_opt_state
                )),
                (critic_loss, actor_loss, critic_loss_dict, actor_loss_dict),
            )

        (key, agent_state, opt_state), (
            critic_loss,
            actor_loss,
            critic_loss_dict,
            actor_loss_dict,
        ) = scan_and_mean(
            update_actor_step,
            (update_key, agent_state, state.opt_state),
            sample_batches,
        )
        train_metrics = TD3TrainMetric(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
        ).all_reduce(dp_axis_name=self.dp_axis_name)
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.dp_axis_name,
        )
        sampled_episodes = psum(
            trajectory_dones.sum().astype(jnp.uint32), axis_name=self.dp_axis_name
        )
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(dp_axis_name=self.dp_axis_name)
        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )

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
            rbf_coefficients=fit_linear_rbf(projected),
            interpolator_updates=completed,
        )
        return state.replace(
            agent_state=state.agent_state.replace(extra_state=extra_state)
        )

    def _maybe_update_interpolator(self, state):
        completed = state.env_state.info.episode_count.min().tolist()
        previous = state.agent_state.extra_state.interpolator_updates.tolist()
        if completed >= previous + self.config.key_update_interval:
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
        return returns.reshape(3, self.config.interpolator_eval_episodes, 2).mean(
            axis=1
        )

    def _evaluate_preferences(self, agent_state, env, preferences, reset_key):
        env_state = env.reset(reset_key)
        returns = jnp.zeros((len(preferences), 2))
        finished = jnp.zeros(len(preferences), dtype=bool)

        def evaluate_step(carry):
            step, env_state, returns, finished = carry
            obs = env_state.obs.replace(preference=preferences)
            actions, _ = self.agent.evaluate_actions(
                agent_state,
                SampleBatch(obs=obs),
                jax.random.fold_in(jax.random.PRNGKey(1), step),
            )
            next_state = env.step(env_state, actions)
            returns += (~finished)[:, None] * next_state.reward
            finished |= next_state.done.astype(bool)
            return step + 1, next_state, returns, finished

        _, _, returns, _ = jax.lax.while_loop(
            lambda carry: (carry[0] < 500) & ~jnp.all(carry[3]),
            evaluate_step,
            (jnp.zeros((), dtype=jnp.uint32), env_state, returns, finished),
        )
        return returns

    def evaluate(self, state):
        key, _ = jax.random.split(state.key)
        returns = jnp.concatenate(
            [
                self._evaluate_pareto_batch(state.agent_state, preferences)
                for preferences in self.pareto_preferences
            ]
        )[: self.pareto_sample_count]
        objectives = returns.reshape(-1, self.config.eval_episodes, 2).mean(1)
        metrics = MORLEvaluateMetric(
            hypervolume=hypervolume_2d(objectives),
            sparsity=sparsity(objectives),
        )
        return metrics, state.replace(key=key)

    def _evaluate_pareto_batch(self, agent_state, preferences):
        return self._evaluate_preferences(
            agent_state,
            self.pareto_eval_env,
            preferences,
            jax.random.PRNGKey(11),
        )

    def learn(self, state):
        """Train to the raw-transition budget and save before offline evaluation."""
        learn_started = time.perf_counter()

        while state.metrics.sampled_timesteps.tolist() < self.config.total_timesteps:
            train_metrics, state = self._multi_steps(state)
            state = self._maybe_update_interpolator(state)
            iterations = state.metrics.iterations.tolist()
            is_final = (
                state.metrics.sampled_timesteps.tolist() >= self.config.total_timesteps
            )
            self.recorder.write(train_metrics.to_local_dict(), iterations)
            self.recorder.write(state.metrics.to_local_dict(), iterations)
            print(
                {
                    "iterations": iterations,
                    "raw_transitions": state.metrics.sampled_timesteps.tolist(),
                    "critic_loss": train_metrics.critic_loss.tolist(),
                    "actor_loss": train_metrics.actor_loss.tolist(),
                    "interpolator_updates": (
                        state.agent_state.extra_state.interpolator_updates.tolist()
                    ),
                    "random_transitions": int(
                        state.agent_state.extra_state.logical_group_random_transition_count.sum()
                    ),
                    "policy_transitions": int(
                        state.agent_state.extra_state.logical_group_policy_transition_count.sum()
                    ),
                },
                flush=True,
            )

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            checkpoint_started = time.perf_counter()
            self.checkpoint_manager.save(iterations, saved_state, force=is_final)
            if is_final:
                self.checkpoint_manager.wait_until_finished()
            print(
                {
                    "iterations": iterations,
                    "checkpoint_seconds": time.perf_counter() - checkpoint_started,
                },
                flush=True,
            )

        print(
            {"training_total_seconds": time.perf_counter() - learn_started},
            flush=True,
        )
        return state

    @classmethod
    def enable_jit(cls):
        cls.step = jax.jit(cls.step, static_argnums=(0,))
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )
        cls._multi_steps = jax.jit(
            cls._multi_steps, static_argnums=(0,), donate_argnums=(1,)
        )
        cls._update_interpolator = jax.jit(
            cls._update_interpolator, static_argnums=(0,)
        )
        cls._evaluate_pareto_batch = jax.jit(
            cls._evaluate_pareto_batch, static_argnums=(0,)
        )
