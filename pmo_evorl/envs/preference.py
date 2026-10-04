"""Episode-level preference sampling for vectorized MORL environments."""

import jax
import jax.numpy as jnp

from evorl.envs import Box, Env, EnvState, SpaceContainer
from evorl.types import Action, PyTreeDict


class PreferenceConditionedEnv(Env):
    """Attach one subdivided-grid preference to each parallel episode."""

    def __init__(self, env: Env, num_envs: int, num_workers: int = 10):
        if num_envs % num_workers:
            raise ValueError("num_envs must be divisible by num_workers")
        self.env = env
        self.num_envs = num_envs
        self.num_workers = num_workers
        envs_per_worker = num_envs // num_workers
        self.worker_index = jnp.arange(num_envs) // envs_per_worker

    def _sample(self, keys: jax.Array) -> jax.Array:
        total_points = 1001
        points_per_worker, extra = divmod(total_points, self.num_workers)
        starts = (
            self.worker_index * points_per_worker
            + jnp.minimum(self.worker_index, extra)
        )
        sizes = points_per_worker + (self.worker_index < extra)
        indices = jax.vmap(
            lambda key, start, stop: jax.random.randint(key, (), start, stop)
        )(
            keys, starts, starts + sizes
        )
        return jnp.stack((indices, 1000 - indices), axis=-1) / 1000.0

    def reset(self, key: jax.Array) -> EnvState:
        env_key, preference_key = jax.random.split(key)
        state = self.env.reset(env_key)
        preference_keys = jax.random.split(preference_key, self.num_envs)
        preference = self._sample(preference_keys)
        info = state.info.replace(
            environment_termination=state.info.termination,
            episode_count=jnp.zeros(self.num_envs, dtype=jnp.uint32),
            preference=preference,
            preference_key=preference_keys,
        )
        obs = PyTreeDict(state=state.obs, preference=preference)
        return state.replace(obs=obs, info=info)

    def step(self, state: EnvState, action: Action) -> EnvState:
        old_preference = state.info.preference
        old_keys = state.info.preference_key
        split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(old_keys)
        next_keys, sample_keys = split_keys[:, 0], split_keys[:, 1]

        next_state = self.env.step(state, action)
        sampled_preference = self._sample(sample_keys)
        done = next_state.done[:, None].astype(bool)
        preference = jnp.where(done, sampled_preference, old_preference)
        preference_keys = jnp.where(done, next_keys, old_keys)
        info = next_state.info.replace(
            environment_termination=next_state.info.termination,
            termination=next_state.done,
            episode_count=(
                state.info.episode_count
                + next_state.done.astype(jnp.uint32)
            ),
            preference=preference,
            preference_key=preference_keys,
        )
        obs = PyTreeDict(state=next_state.obs, preference=preference)
        return next_state.replace(obs=obs, info=info)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def obs_space(self) -> SpaceContainer:
        preference_space = Box(
            low=jnp.zeros(2, dtype=jnp.float32),
            high=jnp.ones(2, dtype=jnp.float32),
        )
        return SpaceContainer(
            spaces=PyTreeDict(
                state=self.env.obs_space,
                preference=preference_space,
            )
        )
