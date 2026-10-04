"""JIT-compatible two-objective Pareto metrics."""

import jax
import jax.numpy as jnp


def nondominated_mask(objectives: jax.Array) -> jax.Array:
    """Return the maximization non-dominated mask."""
    candidate = objectives[:, None, :]
    other = objectives[None, :, :]
    dominated = jnp.any(
        jnp.all(other >= candidate, axis=-1)
        & jnp.any(other > candidate, axis=-1),
        axis=1,
    )
    return ~dominated


def hypervolume_2d(objectives: jax.Array) -> jax.Array:
    """Maximization hypervolume against reference point (0, 0)."""
    points = jnp.maximum(objectives, 0.0)
    keep = nondominated_mask(points)
    order = jnp.argsort(points[:, 0])
    points, keep = points[order], keep[order]

    def add_slice(carry, item):
        previous_x, volume = carry
        point, selected = item
        width = jnp.maximum(point[0] - previous_x, 0.0)
        volume = jnp.where(selected, volume + width * point[1], volume)
        previous_x = jnp.where(selected, point[0], previous_x)
        return (previous_x, volume), None

    (_, volume), _ = jax.lax.scan(
        add_slice, (jnp.zeros(()), jnp.zeros(())), (points, keep)
    )
    return volume


def sparsity(objectives: jax.Array) -> jax.Array:
    """Match the source's mean squared spacing over the Pareto front."""
    keep = nondominated_mask(objectives)
    count = keep.sum()

    def objective_spacing(values):
        values = jnp.sort(jnp.where(keep, values, jnp.inf))
        gaps = jnp.diff(values)
        valid = jnp.arange(gaps.shape[0]) < count - 1
        return jnp.where(valid, gaps**2, 0.0).sum()

    total = jax.vmap(objective_spacing, in_axes=1, out_axes=0)(objectives).sum()
    return jnp.where(count > 1, total / (count - 1), 0.0)
