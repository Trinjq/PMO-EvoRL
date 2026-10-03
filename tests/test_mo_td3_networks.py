"""Executable shape, conditioning, JIT, and gradient checks."""

import jax
import jax.numpy as jnp

from pmo_evorl.networks import PreferenceActor, TwinVectorCritic
from pmo_evorl.mo_td3 import (
    actor_loss,
    critic_loss,
    directional_angle,
    select_target_vector_q,
    vector_td_target,
)


def main() -> None:
    batch_size = 8
    state = jnp.zeros((batch_size, 17))
    preference = jnp.tile(jnp.array([[0.2, 0.8], [0.8, 0.2]]), (4, 1))
    actor = PreferenceActor(action_size=6)
    critic = TwinVectorCritic(reward_size=2)

    actor_params = actor.init(jax.random.PRNGKey(0), state, preference)
    action = jax.jit(actor.apply)(actor_params, state, preference)
    critic_params = critic.init(
        jax.random.PRNGKey(1), state, preference, action
    )
    vector_q = jax.jit(critic.apply)(
        critic_params, state, preference, action
    )
    scalar_q = jnp.einsum("bo,bco->bc", preference, vector_q)

    assert action.shape == (batch_size, 6)
    assert bool((jnp.abs(action) <= 1.0).all())
    assert not bool(jnp.allclose(action[0], action[1]))
    assert vector_q.shape == (batch_size, 2, 2)
    assert scalar_q.shape == (batch_size, 2)

    def actor_objective(params):
        actions = actor.apply(params, state, preference)
        qs = critic.apply(critic_params, state, preference, actions)
        return -jnp.einsum("bo,bo->b", preference, qs[:, 0]).mean()

    gradients = jax.grad(actor_objective)(actor_params)
    leaves = jax.tree.leaves(gradients)
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in leaves)
    assert float(jnp.sqrt(sum(jnp.square(x).sum() for x in leaves))) > 0.0

    target_q = jnp.array(
        [
            [[4.0, 0.0], [2.0, 2.0]],
            [[5.0, 0.0], [0.0, 2.0]],
        ]
    )
    target_preference = jnp.array([[0.75, 0.25], [0.2, 0.8]])
    selected = jax.jit(select_target_vector_q)(target_preference, target_q)
    assert bool(jnp.allclose(selected, jnp.array([[2.0, 2.0], [5.0, 0.0]])))

    reward = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    done = jnp.array([0.0, 1.0])
    td_target = jax.jit(vector_td_target, static_argnums=4)(
        reward, done, target_preference, target_q, 0.5
    )
    assert bool(jnp.allclose(td_target, jnp.array([[2.0, 3.0], [3.0, 4.0]])))

    projected = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    angles = jax.jit(directional_angle)(projected, target_q)
    assert angles.shape == (2, 2)
    assert bool(jnp.isfinite(critic_loss(target_q, selected, projected)))
    assert bool(
        jnp.isfinite(
            actor_loss(target_q[:, 0], target_preference, projected)
        )
    )
    print("MO-TD3 network checks passed:", action.shape, vector_q.shape)


if __name__ == "__main__":
    main()
