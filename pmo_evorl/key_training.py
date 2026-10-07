"""Fixed-preference MO-TD3 used to reproduce the PD-MORL Key artifact."""

import jax
import jax.numpy as jnp
import optax
from evorl.agent import Agent, AgentState
from evorl.algorithms.offpolicy_utils import clean_trajectory
from evorl.algorithms.td3 import TD3TrainMetric, TD3Workflow
from evorl.distributed.gradients import agent_gradient_update
from evorl.evaluators import Evaluator
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update
from evorl.replay_buffers import ReplayBuffer

from pmo_evorl.envs import create_mo_walker2d_env
from evorl.algorithms.td3 import TD3NetworkParams
from pmo_evorl.networks import PreferenceActor, TwinVectorCritic


class FixedPreferenceTD3Agent(Agent):
    """The source Key agent: fixed preference, no HER or interpolator."""

    actor_network: PreferenceActor
    critic_network: TwinVectorCritic
    fixed_preference: jax.Array
    discount: float = 0.99
    exploration_epsilon: float = 0.1
    policy_noise: float = 0.2
    clip_policy_noise: float = 0.5
    random_warmup_transitions: int = 25_000

    def init(self, obs_space, action_space, key):
        actor_key, critic_key, sample_key = jax.random.split(key, 3)
        dummy_obs = obs_space.sample(sample_key)
        dummy_state = dummy_obs.state[None]
        dummy_preference = jnp.broadcast_to(
            self.fixed_preference, (1, self.fixed_preference.shape[-1])
        )
        dummy_action = action_space.sample(sample_key)[None]
        critic_params = self.critic_network.init(
            critic_key, dummy_state, dummy_preference, dummy_action
        )
        actor_params = self.actor_network.init(
            actor_key, dummy_state, dummy_preference
        )
        return AgentState(
            params=TD3NetworkParams(
                critic_params=critic_params,
                actor_params=actor_params,
                target_critic_params=critic_params,
                target_actor_params=actor_params,
            ),
            extra_state=PyTreeDict(random_warmup_steps=jnp.zeros((), dtype=jnp.uint32)),
        )

    def _preference(self, state):
        return jnp.broadcast_to(
            self.fixed_preference, state.shape[:-1] + (self.fixed_preference.shape[-1],)
        )

    def compute_actions(self, agent_state, sample_batch, key):
        state = sample_batch.obs.state
        preference = self._preference(state)
        policy_actions = self.actor_network.apply(
            agent_state.params.actor_params, state, preference
        )
        policy_key, random_key = jax.random.split(key)
        noisy_actions = policy_actions + self.exploration_epsilon * jax.random.normal(
            policy_key, policy_actions.shape
        )
        random_actions = jax.random.uniform(
            random_key, policy_actions.shape, minval=-1.0, maxval=1.0
        )
        use_random = (
            agent_state.extra_state.random_warmup_steps
            < self.random_warmup_transitions
        )
        return jnp.clip(
            jnp.where(use_random, random_actions, noisy_actions), -1.0, 1.0
        ), PyTreeDict()

    def evaluate_actions(self, agent_state, sample_batch, key):
        del key
        state = sample_batch.obs.state
        return self.actor_network.apply(
            agent_state.params.actor_params, state, self._preference(state)
        ), PyTreeDict()

    def critic_loss(self, agent_state, sample_batch, key):
        state = sample_batch.obs.state
        preference = self._preference(state)
        next_state = sample_batch.extras.env_extras.ori_obs
        next_preference = self._preference(next_state)
        next_actions = self.actor_network.apply(
            agent_state.params.target_actor_params, next_state, next_preference
        )
        noise = jnp.clip(
            jax.random.normal(key, next_actions.shape) * self.policy_noise,
            -self.clip_policy_noise,
            self.clip_policy_noise,
        )
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
        target_q = self.critic_network.apply(
            agent_state.params.target_critic_params,
            next_state,
            next_preference,
            next_actions,
        )
        scalar_q = jnp.einsum("bo,bco->bc", preference, target_q)
        critic_index = jnp.argmin(scalar_q, axis=-1)
        selected_q = jnp.take_along_axis(
            target_q, critic_index[:, None, None], axis=1
        ).squeeze(1)
        target = sample_batch.rewards + self.discount * (
            1.0 - sample_batch.extras.env_extras.termination[..., None]
        ) * selected_q
        target = jax.lax.stop_gradient(target)
        current_q = self.critic_network.apply(
            agent_state.params.critic_params,
            state,
            preference,
            sample_batch.actions,
        )
        loss = sum(
            optax.huber_loss(current_q[:, index], target).mean()
            for index in range(2)
        )
        return PyTreeDict(critic_loss=loss, q_value=current_q.mean())

    def actor_loss(self, agent_state, sample_batch, key):
        del key
        state = sample_batch.obs.state
        preference = self._preference(state)
        actions = self.actor_network.apply(
            agent_state.params.actor_params, state, preference
        )
        q1 = self.critic_network.apply(
            agent_state.params.critic_params, state, preference, actions
        )[:, 0]
        return PyTreeDict(actor_loss=-jnp.einsum("bo,bo->b", preference, q1).mean())


class KeyTD3Workflow(TD3Workflow):
    """EvoRL workflow with the source Key update schedule."""

    @classmethod
    def _build_from_config(cls, config):
        env = create_mo_walker2d_env(
            config.num_envs, num_preference_workers=1, autoreset=True
        )
        fixed_preference = jnp.asarray(config.preference, dtype=jnp.float32)
        agent = FixedPreferenceTD3Agent(
            actor_network=PreferenceActor(
                action_size=env.action_space.shape[0], hidden_layer_sizes=(400,)
            ),
            critic_network=TwinVectorCritic(
                reward_size=2, hidden_layer_sizes=(400,)
            ),
            fixed_preference=fixed_preference,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            random_warmup_transitions=config.start_timesteps,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm),
            optax.adam(config.learning_rate),
        )
        replay_buffer = ReplayBuffer(
            capacity=config.replay_size,
            min_sample_timesteps=config.learning_start_timesteps,
            sample_batch_size=config.batch_size,
        )
        eval_env = create_mo_walker2d_env(
            config.eval_episodes, num_preference_workers=1, autoreset=False
        )
        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.max_episode_len,
        )
        workflow = cls(env, agent, optimizer, evaluator, replay_buffer, config)
        workflow.eval_env = eval_env
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
        key, rollout_key, learn_key = jax.random.split(state.key, 3)
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
        replay_buffer_state = self.replay_buffer.add(
            state.replay_buffer_state, trajectory
        )
        sample_batch = self.replay_buffer.sample(
            replay_buffer_state, jax.random.fold_in(learn_key, 0)
        )

        critic_update_fn = agent_gradient_update(
            lambda agent_state, batch, loss_key: self._critic_loss_with_aux(
                agent_state, batch, loss_key
            ),
            self.optimizer,
            dp_axis_name=self.dp_axis_name,
            has_aux=True,
            attach_fn=lambda current, params: current.replace(
                params=current.params.replace(critic_params=params)
            ),
            detach_fn=lambda current: current.params.critic_params,
        )
        actor_update_fn = agent_gradient_update(
            lambda agent_state, batch, loss_key: self._actor_loss_with_aux(
                agent_state, batch, loss_key
            ),
            self.optimizer,
            dp_axis_name=self.dp_axis_name,
            has_aux=True,
            attach_fn=lambda current, params: current.replace(
                params=current.params.replace(actor_params=params)
            ),
            detach_fn=lambda current: current.params.actor_params,
        )
        critic_opt_state = state.opt_state.critic
        actor_opt_state = state.opt_state.actor
        (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
            critic_update_fn(
                critic_opt_state,
                state.agent_state,
                sample_batch,
                jax.random.fold_in(learn_key, 1),
            )
        )
        next_iteration = state.metrics.iterations + 1

        def update_actor(carry):
            agent_state, actor_opt_state = carry
            (actor_loss, actor_loss_dict), agent_state, actor_opt_state = (
                actor_update_fn(
                    actor_opt_state,
                    agent_state,
                    sample_batch,
                    jax.random.fold_in(learn_key, 2),
                )
            )
            target_actor = soft_target_update(
                agent_state.params.target_actor_params,
                agent_state.params.actor_params,
                self.config.tau,
            )
            target_critic = soft_target_update(
                agent_state.params.target_critic_params,
                agent_state.params.critic_params,
                self.config.tau,
            )
            agent_state = agent_state.replace(
                params=agent_state.params.replace(
                    target_actor_params=target_actor,
                    target_critic_params=target_critic,
                )
            )
            return agent_state, actor_opt_state, actor_loss, actor_loss_dict

        def skip_actor(carry):
            agent_state, actor_opt_state = carry
            return (
                agent_state,
                actor_opt_state,
                jnp.zeros_like(critic_loss),
                PyTreeDict(actor_loss=jnp.zeros_like(critic_loss)),
            )

        agent_state, actor_opt_state, actor_loss, actor_loss_dict = jax.lax.cond(
            next_iteration % self.config.policy_freq == 0,
            update_actor,
            skip_actor,
            (agent_state, actor_opt_state),
        )
        extra_state = agent_state.extra_state.replace(
            random_warmup_steps=(
                agent_state.extra_state.random_warmup_steps
                + self.config.rollout_length * self.config.num_envs
            )
        )
        agent_state = agent_state.replace(extra_state=extra_state)
        metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps
            + self.config.rollout_length * self.config.num_envs,
            sampled_episodes=state.metrics.sampled_episodes
            + trajectory_dones.sum().astype(jnp.uint32),
            iterations=next_iteration,
        ).all_reduce(dp_axis_name=self.dp_axis_name)
        train_metrics = TD3TrainMetric(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
        ).all_reduce(dp_axis_name=self.dp_axis_name)
        return train_metrics, state.replace(
            key=key,
            metrics=metrics,
            agent_state=agent_state,
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=state.opt_state.replace(
                actor=actor_opt_state, critic=critic_opt_state
            ),
        )

    def _critic_loss_with_aux(self, agent_state, batch, key):
        loss_dict = self.agent.critic_loss(agent_state, batch, key)
        return loss_dict.critic_loss, loss_dict

    def _actor_loss_with_aux(self, agent_state, batch, key):
        loss_dict = self.agent.actor_loss(agent_state, batch, key)
        return loss_dict.actor_loss, loss_dict

    def evaluate_fixed(self, agent_state):
        env_state = self.eval_env.reset(jax.random.PRNGKey(991))
        returns = jnp.zeros((self.config.eval_episodes, 2))
        finished = jnp.zeros((self.config.eval_episodes,), dtype=bool)

        def eval_step(carry):
            step, env_state, returns, finished = carry
            actions, _ = self.agent.evaluate_actions(
                agent_state,
                SampleBatch(obs=env_state.obs),
                jax.random.fold_in(jax.random.PRNGKey(992), step),
            )
            next_state = self.eval_env.step(env_state, actions)
            returns = returns + (~finished)[:, None] * next_state.reward
            finished = finished | next_state.done.astype(bool)
            return step + 1, next_state, returns, finished

        _, _, returns, _ = jax.lax.while_loop(
            lambda carry: (carry[0] < self.config.max_episode_len)
            & ~jnp.all(carry[3]),
            eval_step,
            (jnp.zeros((), dtype=jnp.uint32), env_state, returns, finished),
        )
        return returns.mean(axis=0)

    @classmethod
    def enable_jit(cls):
        super().enable_jit()
        cls.evaluate_fixed = jax.jit(cls.evaluate_fixed, static_argnums=(0,))
