"""Preference-conditioned networks used by continuous PD-MORL."""

from collections.abc import Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp

from evorl.networks import make_mlp, make_vmap_mlp


class PreferenceActor(nn.Module):
    """Maps state and preference to a bounded continuous action."""

    action_size: int
    hidden_layer_sizes: Sequence[int] = (400, 400)

    @nn.compact
    def __call__(self, state: jax.Array, preference: jax.Array) -> jax.Array:
        inputs = jnp.concatenate((state, preference), axis=-1)
        return make_mlp(
            layer_sizes=tuple(self.hidden_layer_sizes) + (self.action_size,),
            kernel_init=jax.nn.initializers.xavier_normal(),
            activation_final=nn.tanh,
        )(inputs)


class TwinVectorCritic(nn.Module):
    """Two independent critics that each output one vector Q-value."""

    reward_size: int
    hidden_layer_sizes: Sequence[int] = (400, 400)
    num_critics: int = 2

    @nn.compact
    def __call__(
        self,
        state: jax.Array,
        preference: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        inputs = jnp.concatenate((state, preference, action), axis=-1)
        inputs = jnp.broadcast_to(inputs, (self.num_critics,) + inputs.shape)
        return make_vmap_mlp(
            layer_sizes=tuple(self.hidden_layer_sizes) + (self.reward_size,),
            kernel_init=jax.nn.initializers.xavier_normal(),
        )(inputs)
