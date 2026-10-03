"""Core vector target and losses from the original MO-TD3 implementation."""

import jax
import jax.numpy as jnp
import optax


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


def critic_loss(
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


def actor_loss(
    q1_vector: jax.Array,
    preference: jax.Array,
    projected_preference: jax.Array,
    angle_coefficient: float = 10.0,
) -> jax.Array:
    """Source actor objective: -w^T Q1 plus directional-angle penalty."""
    scalar_q = jnp.einsum("bo,bo->b", preference, q1_vector)
    angle = directional_angle(projected_preference, q1_vector)
    return -scalar_q.mean() + angle_coefficient * angle.mean()
