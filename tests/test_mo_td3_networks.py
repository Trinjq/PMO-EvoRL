"""Executable shape, conditioning, JIT, and gradient checks."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict

from pmo_evorl.interpolator import (
    WALKER2D_KEY_OBJECTIVES,
    linear_rbf_project,
    normalize_objectives,
)
from pmo_evorl.networks import PreferenceActor, TwinVectorCritic
from pmo_evorl.mo_td3 import (
    actor_loss,
    critic_loss,
    directional_angle,
    select_target_vector_q,
    vector_td_target,
    MOTD3Agent,
    PreferenceHERReplayBuffer,
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

    rbf_query = jnp.array([[0.2, 0.8], [0.7, 0.3]])
    projected_rbf = jax.jit(linear_rbf_project)(
        rbf_query, normalize_objectives(WALKER2D_KEY_OBJECTIVES)
    )
    scipy_reference = jnp.array(
        [[0.35938325, 0.90684321], [0.74972307, 0.58036140]]
    )
    assert bool(jnp.allclose(projected_rbf, scipy_reference, atol=1e-6))

    class DummySpace:
        def __init__(self, value):
            self.value = value

        def sample(self, key):
            del key
            return self.value

    agent = MOTD3Agent(critic_network=critic, actor_network=actor)
    obs_space = DummySpace(
        PyTreeDict(state=jnp.zeros(17), preference=jnp.array([0.5, 0.5]))
    )
    action_space = DummySpace(jnp.zeros(6))
    agent_state = agent.init(obs_space, action_space, jax.random.PRNGKey(2))
    batch = SampleBatch(
        obs=PyTreeDict(state=state, preference=preference),
        actions=action,
        rewards=jnp.ones((batch_size, 2)),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=state + 0.1,
                termination=jnp.zeros(batch_size),
            )
        ),
    )
    noisy_action, _ = jax.jit(agent.compute_actions)(
        agent_state, SampleBatch(obs=batch.obs), jax.random.PRNGKey(3)
    )
    critic_metrics = jax.jit(agent.critic_loss)(
        agent_state, batch, jax.random.PRNGKey(4)
    )
    actor_metrics = jax.jit(agent.actor_loss)(
        agent_state, batch, jax.random.PRNGKey(5)
    )
    assert noisy_action.shape == (batch_size, 6)
    assert bool(jnp.isfinite(critic_metrics.critic_loss))
    assert bool(jnp.isfinite(actor_metrics.actor_loss))

    replay = PreferenceHERReplayBuffer(
        capacity=64,
        sample_batch_size=8,
        learning_start_timesteps=0,
    )
    replay_state = replay.init(jtu.tree_map(lambda value: value[0], batch))
    replay_state = jax.jit(replay.add)(replay_state, batch)
    assert int(replay_state.buffer_size) == batch_size * 4
    stored_preferences = replay_state.data.obs.preference[: replay_state.buffer_size]
    assert bool(jnp.allclose(stored_preferences[:batch_size], preference))
    assert bool(
        jnp.allclose(stored_preferences[batch_size:].sum(axis=-1), 1.0)
    )
    print("MO-TD3 network checks passed:", action.shape, vector_q.shape)


if __name__ == "__main__":
    main()
