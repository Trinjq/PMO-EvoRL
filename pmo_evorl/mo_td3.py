"""MO-TD3 agent, vector target, and losses."""

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.agent import Agent, AgentState
from evorl.algorithms.td3 import TD3NetworkParams
from evorl.replay_buffers import ReplayBuffer
from evorl.sample_batch import SampleBatch
from evorl.types import Action, LossDict, PolicyExtraInfo, PyTreeDict

from pmo_evorl.interpolator import (
    WALKER2D_KEY_OBJECTIVES,
    linear_rbf_project,
    normalize_objectives,
)


def select_target_vector_q(
    preference: jax.Array, target_vector_q: jax.Array
) -> jax.Array:
    """Select one whole critic vector using the smaller scalarized value."""
    scalar_q = jnp.einsum("bo,bco->bc", preference, target_vector_q)
    critic_index = jnp.argmin(scalar_q, axis=-1)
    return jnp.take_along_axis(
        target_vector_q, critic_index[:, None, None], axis=1
    ).squeeze(1)


def vector_td_target(
    reward: jax.Array,
    done: jax.Array,
    preference: jax.Array,
    target_vector_q: jax.Array,
    discount: float,
) -> jax.Array:
    """Build r + gamma * (1-done) * selected vector Q."""
    selected_q = select_target_vector_q(preference, target_vector_q)
    return reward + discount * (1.0 - done[..., None]) * selected_q


def directional_angle(
    projected_preference: jax.Array, vector_q: jax.Array
) -> jax.Array:
    """Directional angle in degrees, matching the source clamp behavior."""
    if vector_q.ndim == projected_preference.ndim + 1:
        projected_preference = projected_preference[:, None, :]
    preference_norm = jnp.maximum(
        jnp.linalg.norm(projected_preference, axis=-1), 1e-8
    )
    q_norm = jnp.maximum(jnp.linalg.norm(vector_q, axis=-1), 1e-8)
    cosine = (projected_preference * vector_q).sum(-1) / (
        preference_norm * q_norm
    )
    return jnp.rad2deg(jnp.arccos(jnp.clip(cosine, 0.0, 0.9999)))


def vector_critic_loss(
    current_vector_q: jax.Array,
    target_vector_q: jax.Array,
    projected_preference: jax.Array,
) -> jax.Array:
    """Sum both critics' directional-angle and SmoothL1 losses."""
    angle = directional_angle(projected_preference, current_vector_q)
    huber = optax.huber_loss(
        current_vector_q, target_vector_q[:, None, :], delta=1.0
    )
    return angle.sum(axis=1).mean() + huber.sum(axis=1).mean()


def preference_actor_loss(
    q1_vector: jax.Array,
    preference: jax.Array,
    projected_preference: jax.Array,
    angle_coefficient: float = 10.0,
) -> jax.Array:
    """Source actor objective: -w^T Q1 plus directional-angle penalty."""
    scalar_q = jnp.einsum("bo,bo->b", preference, q1_vector)
    angle = directional_angle(projected_preference, q1_vector)
    return -scalar_q.mean() + angle_coefficient * angle.mean()


# Backward-compatible names for the small functional API.
critic_loss = vector_critic_loss
actor_loss = preference_actor_loss


class PreferenceHERReplayBuffer(ReplayBuffer):
    """Relabel each transition with source-compatible random preferences."""

    weight_num: int = 3
    her_start_timesteps: int = 100_000
    seed: int = 0

    def add(self, buffer_state, xs, mask=None):
        if mask is not None:
            raise ValueError("PreferenceHERReplayBuffer owns its add mask")

        batch_size = xs.obs.preference.shape[0]
        key = jax.random.fold_in(
            jax.random.fold_in(
                jax.random.PRNGKey(self.seed), buffer_state.current_index
            ),
            buffer_state.buffer_size,
        )
        preferences = jnp.abs(
            jax.random.normal(key, (batch_size * self.weight_num, 2))
        )
        preferences /= preferences.sum(axis=-1, keepdims=True)
        preferences = jnp.round(preferences, decimals=3)

        relabeled = jtu.tree_map(
            lambda value: jnp.repeat(value, self.weight_num, axis=0), xs
        )
        relabeled = relabeled.replace(
            obs=relabeled.obs.replace(preference=preferences)
        )
        augmented = jtu.tree_map(
            lambda original, her: jnp.concatenate((original, her), axis=0),
            xs,
            relabeled,
        )
        use_her = (
            buffer_state.buffer_size + jnp.arange(1, batch_size + 1)
            > self.her_start_timesteps
        )
        add_mask = jnp.concatenate(
            (
                jnp.ones(batch_size, dtype=bool),
                jnp.repeat(use_her, self.weight_num),
            )
        )
        return super().add(buffer_state, augmented, add_mask)


class MOTD3Agent(Agent):
    """EvoRL-compatible, preference-conditioned vector MO-TD3 agent."""

    critic_network: nn.Module
    actor_network: nn.Module
    discount: float = 0.995
    exploration_epsilon: float = 0.1
    policy_noise: float = 0.2
    clip_policy_noise: float = 0.5
    angle_coefficient: float = 10.0

    def init(self, obs_space, action_space, key: chex.PRNGKey) -> AgentState:
        sample_key, critic_key, actor_key = jax.random.split(key, 3)
        dummy_obs = jtu.tree_map(
            lambda value: value[None], obs_space.sample(sample_key)
        )
        dummy_action = action_space.sample(sample_key)[None]
        critic_params = self.critic_network.init(
            critic_key,
            dummy_obs.state,
            dummy_obs.preference,
            dummy_action,
        )
        actor_params = self.actor_network.init(
            actor_key, dummy_obs.state, dummy_obs.preference
        )
        params = TD3NetworkParams(
            critic_params=critic_params,
            actor_params=actor_params,
            target_critic_params=critic_params,
            target_actor_params=actor_params,
        )
        return AgentState(
            params=params,
            extra_state=PyTreeDict(
                key_objectives=WALKER2D_KEY_OBJECTIVES,
                projected_key_values=normalize_objectives(
                    WALKER2D_KEY_OBJECTIVES
                ),
                interpolator_updates=jnp.zeros((), dtype=jnp.uint32),
            ),
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        actions = self.actor_network.apply(
            agent_state.params.actor_params, obs.state, obs.preference
        )
        actions += jax.random.normal(key, actions.shape) * self.exploration_epsilon
        return jnp.clip(actions, -1.0, 1.0), PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        del key
        obs = sample_batch.obs
        actions = self.actor_network.apply(
            agent_state.params.actor_params, obs.state, obs.preference
        )
        return actions, PyTreeDict()

    def critic_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs
        next_state = sample_batch.extras.env_extras.ori_obs
        next_actions = self.actor_network.apply(
            agent_state.params.target_actor_params,
            next_state,
            obs.preference,
        )
        noise = jnp.clip(
            jax.random.normal(key, next_actions.shape) * self.policy_noise,
            -self.clip_policy_noise,
            self.clip_policy_noise,
        )
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
        next_q = self.critic_network.apply(
            agent_state.params.target_critic_params,
            next_state,
            obs.preference,
            next_actions,
        )
        target = vector_td_target(
            sample_batch.rewards,
            sample_batch.extras.env_extras.termination,
            obs.preference,
            next_q,
            self.discount,
        )
        target = jax.lax.stop_gradient(target)
        current_q = self.critic_network.apply(
            agent_state.params.critic_params,
            obs.state,
            obs.preference,
            sample_batch.actions,
        )
        projected = linear_rbf_project(
            obs.preference, agent_state.extra_state.projected_key_values
        )
        loss = vector_critic_loss(current_q, target, projected)
        return PyTreeDict(critic_loss=loss, q_value=current_q.mean())

    def actor_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        del key
        obs = sample_batch.obs
        actions = self.actor_network.apply(
            agent_state.params.actor_params, obs.state, obs.preference
        )
        q1 = self.critic_network.apply(
            agent_state.params.critic_params,
            obs.state,
            obs.preference,
            actions,
        )[:, 0]
        projected = linear_rbf_project(
            obs.preference, agent_state.extra_state.projected_key_values
        )
        loss = preference_actor_loss(
            q1,
            obs.preference,
            projected,
            self.angle_coefficient,
        )
        return PyTreeDict(actor_loss=loss)
