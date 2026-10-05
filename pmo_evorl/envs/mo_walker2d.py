"""PD-MORL Walker2d running on MuJoCo MJX-JAX."""

from pathlib import Path

import jax
import jax.numpy as jnp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import MjxEnv
from mujoco_playground._src import mjx_env


_XML_PATH = Path(__file__).with_name("assets") / "walker2d_pdmorl.xml"


class MOWalker2d(MjxEnv):
    """The original PD-MORL Walker2d contract on the MJX-JAX backend."""

    def __init__(self) -> None:
        super().__init__(
            config_dict.create(ctrl_dt=0.008, sim_dt=0.002, impl="jax")
        )
        self._mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, qpos_rng, qvel_rng = jax.random.split(rng, 3)
        low, high = -0.005, 0.005
        qpos = jnp.asarray(self.mj_model.qpos0) + jax.random.uniform(
            qpos_rng, (self.mjx_model.nq,), minval=low, maxval=high
        )
        qvel = jax.random.uniform(
            qvel_rng, (self.mjx_model.nv,), minval=low, maxval=high
        )
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self.mjx_model.impl.value,
        )
        data = mjx.forward(self.mjx_model, data)
        zero = jnp.zeros(())
        metrics = {"reward_speed": zero, "reward_energy": zero}
        return mjx_env.State(
            data,
            self._get_obs(data),
            jnp.zeros(2),
            zero,
            metrics,
            {"rng": rng},
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jnp.clip(action, -1.0, 1.0)
        x_before = state.data.qpos[0]
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        speed = (data.qpos[0] - x_before) / self.dt + 1.0
        energy = 5.0 - jnp.square(action).sum()
        reward = jnp.stack((speed, energy))
        healthy = (
            (data.qpos[1] > 0.8)
            & (data.qpos[1] < 2.0)
            & (data.qpos[2] > -1.0)
            & (data.qpos[2] < 1.0)
        )
        metrics = {"reward_speed": speed, "reward_energy": energy}
        return mjx_env.State(
            data,
            self._get_obs(data),
            reward,
            1.0 - healthy.astype(jnp.float32),
            metrics,
            state.info,
        )

    @staticmethod
    def _get_obs(data: mjx.Data) -> jax.Array:
        return jnp.concatenate((data.qpos[1:], jnp.clip(data.qvel, -10.0, 10.0)))

    @property
    def xml_path(self) -> str:
        return str(_XML_PATH)

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
