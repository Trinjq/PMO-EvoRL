"""Evaluate a saved policy with native CPU MuJoCo."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# Orbax imports JAX; pin it to CPU before that import initializes a backend.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import mujoco
import numpy as np
import orbax.checkpoint as ocp

_XML_PATH = (
    Path(__file__).parent / "pmo_evorl" / "envs" / "assets" / "walker2d_pdmorl.xml"
)
SIM_TIMESTEP = 0.002
FRAME_SKIP = 4
CONTROL_TIMESTEP = SIM_TIMESTEP * FRAME_SKIP
EPISODE_LENGTH = 500
RESET_NOISE_SCALE = 0.005


def validate_model(model: mujoco.MjModel) -> None:
    """Reject CPU models that do not match the MJX Walker2d task contract."""
    if (model.nq, model.nv, model.nu) != (9, 9, 6):
        raise ValueError("Walker2d model dimensions must be nq=9, nv=9, nu=6")
    if not np.isclose(model.opt.timestep, SIM_TIMESTEP):
        raise ValueError(f"simulation timestep must be {SIM_TIMESTEP}")
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4:
        raise ValueError("Walker2d integrator must be RK4")
    if not np.allclose(model.actuator_ctrlrange, (-1.0, 1.0)):
        raise ValueError("all Walker2d actuator ranges must be [-1, 1]")


def observation(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate((data.qpos[1:], np.clip(data.qvel, -10.0, 10.0)))


def step_environment(
    model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray
) -> tuple[np.ndarray, bool]:
    action = np.clip(action, -1.0, 1.0)
    x_before = data.qpos[0]
    data.ctrl[:] = action
    mujoco.mj_step(model, data, nstep=FRAME_SKIP)
    reward = np.array(
        [
            (data.qpos[0] - x_before) / CONTROL_TIMESTEP + 1.0,
            5.0 - np.square(action).sum(),
        ]
    )
    healthy = 0.8 < data.qpos[1] < 2.0 and -1.0 < data.qpos[2] < 1.0
    return reward, not healthy


def load_actor_layers(checkpoint: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    checkpoint = checkpoint.resolve()
    item = checkpoint / "default" if (checkpoint / "default").is_dir() else checkpoint
    with ocp.StandardCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(item).item_metadata.tree
        cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
        target = jax.tree.map(
            lambda value: jax.ShapeDtypeStruct(
                value.shape, value.dtype, sharding=cpu_sharding
            )
            if isinstance(value, ocp.metadata.ArrayMetadata)
            else value,
            metadata,
            is_leaf=lambda value: isinstance(value, ocp.metadata.ArrayMetadata),
        )
        restored = checkpointer.restore(item, target=target)
    try:
        params = restored[0]["params"]["actor_params"]["params"]["MLP_0"]
        layers = [
            (
                np.asarray(params[f"hidden_{i}"]["kernel"]),
                np.asarray(params[f"hidden_{i}"]["bias"]),
            )
            for i in range(3)
        ]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(f"unsupported checkpoint structure: {checkpoint}") from error
    expected = [
        ((19, 400), (400,)),
        ((400, 400), (400,)),
        ((400, 6), (6,)),
    ]
    if [(kernel.shape, bias.shape) for kernel, bias in layers] != expected:
        raise ValueError(
            "checkpoint actor does not match the Walker2d network contract"
        )
    return layers


def actor(
    inputs: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    values = inputs
    for kernel, bias in layers[:-1]:
        values = np.maximum(values @ kernel + bias, 0.0)
    kernel, bias = layers[-1]
    return np.tanh(values @ kernel + bias)


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    candidate = points[:, None, :]
    other = points[None, :, :]
    return ~np.any(
        np.all(other >= candidate, axis=-1) & np.any(other > candidate, axis=-1),
        axis=1,
    )


def hypervolume_2d(points: np.ndarray) -> float:
    points = np.maximum(points, 0.0)
    points = points[nondominated_mask(points)]
    points = points[np.argsort(points[:, 0])]
    previous_x = 0.0
    volume = 0.0
    for x, y in points:
        volume += max(x - previous_x, 0.0) * y
        previous_x = x
    return float(volume)


def sparsity(points: np.ndarray) -> float:
    points = points[nondominated_mask(points)]
    if len(points) <= 1:
        return 0.0
    return float(
        np.square(np.diff(np.sort(points, axis=0), axis=0)).sum() / (len(points) - 1)
    )


def pareto_count(points: np.ndarray) -> int:
    """Return the number of non-dominated points in a Pareto front."""
    return int(nondominated_mask(points).sum())


def reset(model: mujoco.MjModel, seed: int) -> mujoco.MjData:
    rng = np.random.RandomState(seed)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 + rng.uniform(
        -RESET_NOISE_SCALE, RESET_NOISE_SCALE, model.nq
    )
    data.qvel[:] = rng.uniform(
        -RESET_NOISE_SCALE, RESET_NOISE_SCALE, model.nv
    )
    mujoco.mj_forward(model, data)
    return data


def evaluate(
    layers: list[tuple[np.ndarray, np.ndarray]],
    repeats: int,
    step_size: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preference_count = round(1.0 / step_size) + 1
    preferences = np.stack(
        (
            np.linspace(0.0, 1.0, preference_count),
            np.linspace(1.0, 0.0, preference_count),
        ),
        axis=-1,
    ).astype(np.float32)
    model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    validate_model(model)
    episode_seeds = seed + np.arange(repeats * preference_count)
    data = [reset(model, int(episode_seed)) for episode_seed in episode_seeds]
    episode_preferences = np.tile(preferences, (repeats, 1))
    returns = np.zeros((len(data), 2), dtype=np.float64)
    lengths = np.zeros(len(data), dtype=np.int32)
    active = np.ones(len(data), dtype=bool)

    for _ in range(EPISODE_LENGTH):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        observations = np.stack(
            [
                observation(data[i])
                for i in indices
            ]
        ).astype(np.float32)
        actions = np.clip(
            actor(
                np.concatenate((observations, episode_preferences[indices]), axis=-1),
                layers,
            ),
            -1.0,
            1.0,
        )
        for i, action in zip(indices, actions):
            reward, done = step_environment(model, data[i], action)
            returns[i] += reward
            lengths[i] += 1
            active[i] = not done

    return (
        preferences,
        returns.reshape(repeats, preference_count, 2),
        lengths,
        episode_seeds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.step_size <= 0 or not np.isclose(
        round(1 / args.step_size) * args.step_size, 1.0
    ):
        parser.error("--step-size must divide one exactly")
    if not args.checkpoint.is_dir():
        parser.error("--checkpoint must be a checkpoint directory")
    if args.output_dir.exists():
        parser.error("--output-dir already exists")

    total_started = time.perf_counter()
    restore_started = time.perf_counter()
    layers = load_actor_layers(args.checkpoint)
    restore_seconds = time.perf_counter() - restore_started
    evaluation_started = time.perf_counter()
    preferences, returns, lengths, episode_seeds = evaluate(
        layers, args.repeats, args.step_size, args.seed
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    repeat_hypervolumes = [hypervolume_2d(repeat_returns) for repeat_returns in returns]
    repeat_sparsities = [sparsity(repeat_returns) for repeat_returns in returns]
    repeat_pareto_counts = [pareto_count(repeat_returns) for repeat_returns in returns]
    mean_returns = returns.mean(axis=0)
    source_hypervolume = hypervolume_2d(mean_returns)
    source_sparsity = sparsity(mean_returns)
    source_pareto_count = pareto_count(mean_returns)
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "preferences": len(preferences),
        "repeats": args.repeats,
        "seed": args.seed,
        "backend": "MuJoCo CPU",
        "model_sha256": hashlib.sha256(_XML_PATH.read_bytes()).hexdigest(),
        "observation_size": 17,
        "action_size": 6,
        "reward_size": 2,
        "simulation_timestep": SIM_TIMESTEP,
        "control_timestep": CONTROL_TIMESTEP,
        "frame_skip": FRAME_SKIP,
        "episode_length": EPISODE_LENGTH,
        "reset_noise_scale": RESET_NOISE_SCALE,
        "hypervolume_per_repeat": repeat_hypervolumes,
        "sparsity_per_repeat": repeat_sparsities,
        "pareto_count_per_repeat": repeat_pareto_counts,
        "hypervolume": float(np.mean(repeat_hypervolumes)),
        "sparsity": float(np.mean(repeat_sparsities)),
        "pareto_count": int(np.mean(repeat_pareto_counts)),
        "source_hv": source_hypervolume,
        "source_sparsity": source_sparsity,
        "source_pareto_count": source_pareto_count,
        "episode_length_mean": float(lengths.mean()),
        "episode_length_max": int(lengths.max()),
        "restore_seconds": restore_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": time.perf_counter() - total_started,
    }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "returns.npz",
        preferences=preferences,
        returns_per_repeat=returns,
        mean_returns=mean_returns,
        episode_lengths=lengths.reshape(args.repeats, len(preferences)),
        episode_seeds=episode_seeds.reshape(args.repeats, len(preferences)),
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
