"""Executable shape, conditioning, JIT, and gradient checks."""

import jax
import jax.numpy as jnp

from pmo_evorl.networks import PreferenceActor, TwinVectorCritic


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
    print("MO-TD3 network checks passed:", action.shape, vector_q.shape)


if __name__ == "__main__":
    main()
