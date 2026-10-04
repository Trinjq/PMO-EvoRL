"""JAX equivalent of PD-MORL's three-key linear RBF interpolator."""

import jax
import jax.numpy as jnp


KEY_PREFERENCES = jnp.array(
    [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=jnp.float32
)

WALKER2D_KEY_OBJECTIVES = jnp.array(
    [
        [497.2413299560547, 2494.5884033203124],
        [1639.5962036132812, 2156.5128173828125],
        [2602.103564453125, 691.5010070800781],
    ],
    dtype=jnp.float32,
)


def normalize_objectives(objectives: jax.Array, order: int = 2) -> jax.Array:
    norm = jnp.linalg.norm(objectives, ord=order, axis=-1, keepdims=True)
    return objectives / jnp.maximum(norm, 1e-8)


def linear_rbf_project(
    preference: jax.Array,
    key_values: jax.Array,
    key_preferences: jax.Array = KEY_PREFERENCES,
) -> jax.Array:
    """Match scipy RBFInterpolator(kernel='linear', degree=0)."""
    count = key_preferences.shape[0]
    kernel = -jnp.linalg.norm(
        key_preferences[:, None, :] - key_preferences[None, :, :], axis=-1
    )
    polynomial = jnp.ones((count, 1), dtype=key_values.dtype)
    system = jnp.block(
        [[kernel, polynomial], [polynomial.T, jnp.zeros((1, 1))]]
    )
    rhs = jnp.concatenate(
        (key_values, jnp.zeros((1, key_values.shape[-1]), dtype=key_values.dtype))
    )
    coefficients = jnp.linalg.solve(system, rhs)
    query_kernel = -jnp.linalg.norm(
        preference[:, None, :] - key_preferences[None, :, :], axis=-1
    )
    return query_kernel @ coefficients[:count] + coefficients[count]
