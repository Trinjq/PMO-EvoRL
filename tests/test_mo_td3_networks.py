"""Executable shape, conditioning, JIT, and gradient checks."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict

from pmo_evorl.interpolator import (
    WALKER2D_KEY_OBJECTIVES,
    evaluate_linear_rbf,
    fit_linear_rbf,
    normalize_objectives,
    update_key_objectives,
)
from pmo_evorl.metrics import hypervolume_2d, sparsity
from pmo_evorl.mo_td3 import (
    LazyPreferenceHERReplayBuffer,
    MOTD3Agent,
    PreferenceHERReplayBuffer,
    actor_loss,
    critic_loss,
    directional_angle,
    select_target_vector_q,
    vector_td_target,
)
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
    large_gradients = jtu.tree_map(lambda value: value * 1000.0, gradients)
    raw_norm = optax.global_norm(large_gradients)
    clipped_gradients, _ = optax.clip_by_global_norm(100.0).update(
        large_gradients, optax.EmptyState()
    )
    assert float(raw_norm) > 100.0
    assert bool(jnp.isclose(optax.global_norm(clipped_gradients), 100.0))

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
    initial_key_values = normalize_objectives(WALKER2D_KEY_OBJECTIVES)
    coefficients = jax.jit(fit_linear_rbf)(initial_key_values)
    projected_rbf = jax.jit(evaluate_linear_rbf)(rbf_query, coefficients)
    reference_gradient = jax.grad(
        lambda query: evaluate_linear_rbf(
            query, fit_linear_rbf(initial_key_values)
        ).sum()
    )(rbf_query)
    cached_gradient = jax.grad(
        lambda query: evaluate_linear_rbf(query, coefficients).sum()
    )(rbf_query)
    scipy_reference = jnp.array(
        [[0.35938325, 0.90684321], [0.74972307, 0.58036140]]
    )
    assert bool(jnp.allclose(projected_rbf, scipy_reference, atol=1e-6))
    assert bool(jnp.allclose(cached_gradient, reference_gradient, atol=1e-6))
    random_preferences = jax.random.uniform(
        jax.random.PRNGKey(8), (100, 2)
    )
    random_preferences /= random_preferences.sum(axis=-1, keepdims=True)
    old_coefficients = fit_linear_rbf(initial_key_values)
    old_projection = evaluate_linear_rbf(random_preferences, old_coefficients)
    cached_projection = evaluate_linear_rbf(
        random_preferences, coefficients
    )
    projection_error = jnp.max(jnp.abs(old_projection - cached_projection))
    old_projection_gradient = jax.grad(
        lambda query: evaluate_linear_rbf(query, old_coefficients).sum()
    )(random_preferences)
    cached_projection_gradient = jax.grad(
        lambda query: evaluate_linear_rbf(query, coefficients).sum()
    )(random_preferences)
    gradient_error = jnp.max(
        jnp.abs(old_projection_gradient - cached_projection_gradient)
    )
    assert float(projection_error) <= 1e-5
    assert float(gradient_error) <= 1e-5

    candidate_keys = WALKER2D_KEY_OBJECTIVES.at[0, 1].add(1.0).at[1].set(0.0)
    updated_keys, projected_keys = update_key_objectives(
        WALKER2D_KEY_OBJECTIVES, candidate_keys
    )
    assert float(updated_keys[0, 1]) == float(candidate_keys[0, 1])
    assert bool(jnp.array_equal(updated_keys[1], WALKER2D_KEY_OBJECTIVES[1]))
    assert bool(jnp.allclose(projected_keys.sum(axis=-1), 1.0))
    updated_coefficients = fit_linear_rbf(projected_keys)
    assert not bool(jnp.allclose(coefficients, updated_coefficients))

    pareto_points = jnp.array([[1.0, 5.0], [3.0, 2.0], [2.0, 1.0]])
    assert bool(jnp.isclose(hypervolume_2d(pareto_points), 9.0))
    assert bool(jnp.isclose(sparsity(pareto_points), 13.0))

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

    grouped_agent = MOTD3Agent(
        critic_network=critic,
        actor_network=actor,
        logical_group_count=2,
        random_warmup_transitions=1,
    )
    grouped_state = grouped_agent.init(
        obs_space, action_space, jax.random.PRNGKey(6)
    )
    grouped_state = grouped_state.replace(
        extra_state=grouped_state.extra_state.replace(
            logical_group_transition_count=jnp.array([0, 1], dtype=jnp.uint32)
        )
    )
    grouped_key = jax.random.PRNGKey(7)
    grouped_actions, _ = jax.jit(grouped_agent.compute_actions)(
        grouped_state, SampleBatch(obs=batch.obs), grouped_key
    )
    policy_key, random_key = jax.random.split(grouped_key)
    policy_actions = actor.apply(
        grouped_state.params.actor_params, state, preference
    )
    policy_actions = policy_actions + jax.random.normal(
        policy_key, policy_actions.shape
    ) * grouped_agent.exploration_epsilon
    random_actions = jax.random.uniform(
        random_key, policy_actions.shape, minval=-1.0, maxval=1.0
    )
    expected_grouped = jnp.concatenate(
        (random_actions[:4], jnp.clip(policy_actions[4:], -1.0, 1.0))
    )
    assert bool(jnp.allclose(grouped_actions, expected_grouped))
    assert grouped_state.extra_state.logical_group_transition_count.shape == (2,)

    replay = PreferenceHERReplayBuffer(
        capacity=64,
        sample_batch_size=8,
        her_start_timesteps=0,
    )
    replay_state = replay.init(jtu.tree_map(lambda value: value[0], batch))
    replay_state = jax.jit(replay.add)(replay_state, batch)
    assert int(replay_state.buffer_size) == batch_size * 4
    stored_preferences = replay_state.data.obs.preference[: replay_state.buffer_size]
    assert bool(jnp.allclose(stored_preferences[:batch_size], preference))
    assert bool(
        jnp.allclose(stored_preferences[batch_size:].sum(axis=-1), 1.0)
    )

    interleaved_replay = PreferenceHERReplayBuffer(
        capacity=64,
        sample_batch_size=8,
        her_start_timesteps=0,
        interleave=True,
    )
    interleaved_state = interleaved_replay.init(
        jtu.tree_map(lambda value: value[0], batch)
    )
    interleaved_state = jax.jit(interleaved_replay.add)(
        interleaved_state, batch
    )
    grouped_preferences = interleaved_state.data.obs.preference[
        : interleaved_state.buffer_size
    ].reshape(batch_size, 4, 2)
    assert bool(jnp.allclose(grouped_preferences[:, 0], preference))
    assert bool(
        jnp.allclose(grouped_preferences[:, 1:].sum(axis=-1), 1.0)
    )

    lazy_replay = LazyPreferenceHERReplayBuffer(
        capacity=64,
        sample_batch_size=8,
        her_start_timesteps=0,
    )
    lazy_state = lazy_replay.init(jtu.tree_map(lambda value: value[0], batch))
    lazy_state = jax.jit(lazy_replay.add)(lazy_state, batch)
    assert int(lazy_state.buffer_size) == batch_size
    assert lazy_state.data.extras.env_extras.her_seed.shape[0] == lazy_replay.capacity
    lazy_batch = jax.jit(lazy_replay.sample)(
        lazy_state, jax.random.PRNGKey(5)
    )
    assert lazy_batch.obs.preference.shape == (batch_size, 2)
    lazy_batches = jax.jit(lazy_replay.sample_many, static_argnums=2)(
        lazy_state, jax.random.PRNGKey(6), 3
    )
    assert lazy_batches.obs.preference.shape == (3, batch_size, 2)
    assert bool(jnp.allclose(lazy_batches.obs.preference.sum(axis=-1), 1.0))
    print("MO-TD3 network checks passed:", action.shape, vector_q.shape)


if __name__ == "__main__":
    main()
