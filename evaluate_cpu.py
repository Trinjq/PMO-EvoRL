"""Evaluate a saved policy with native CPU MuJoCo."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import orbax.checkpoint as ocp

_XML_PATH = (
    Path(__file__).parent / "pmo_evorl" / "envs" / "assets" / "walker2d_pdmorl.xml"
)


def load_actor_layers(checkpoint: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    item = checkpoint / "default" if (checkpoint / "default").is_dir() else checkpoint
    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(item)
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


def reset(model: mujoco.MjModel, seed: int) -> mujoco.MjData:
    rng = np.random.RandomState(seed)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 + rng.uniform(-0.005, 0.005, model.nq)
    data.qvel[:] = rng.uniform(-0.005, 0.005, model.nv)
    mujoco.mj_forward(model, data)
    return data


def evaluate(
    layers: list[tuple[np.ndarray, np.ndarray]], repeats: int, step_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preference_count = round(1.0 / step_size) + 1
    preferences = np.stack(
        (
            np.linspace(0.0, 1.0, preference_count),
            np.linspace(1.0, 0.0, preference_count),
        ),
        axis=-1,
    ).astype(np.float32)
    model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    data = [reset(model, repeat * 11) for repeat in range(repeats) for _ in preferences]
    episode_preferences = np.tile(preferences, (repeats, 1))
    returns = np.zeros((len(data), 2), dtype=np.float64)
    lengths = np.zeros(len(data), dtype=np.int32)
    active = np.ones(len(data), dtype=bool)

    for _ in range(500):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        observations = np.stack(
            [
                np.concatenate((data[i].qpos[1:], np.clip(data[i].qvel, -10.0, 10.0)))
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
            x_before = data[i].qpos[0]
            data[i].ctrl[:] = action
            mujoco.mj_step(model, data[i], nstep=4)
            returns[i] += (
                (data[i].qpos[0] - x_before) / 0.008 + 1.0,
                5.0 - np.square(action).sum(),
            )
            lengths[i] += 1
            active[i] = 0.8 < data[i].qpos[1] < 2.0 and -1.0 < data[i].qpos[2] < 1.0

    return preferences, returns.reshape(repeats, preference_count, 2), lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--step-size", type=float, default=0.005)
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
    preferences, returns, lengths = evaluate(layers, args.repeats, args.step_size)
    evaluation_seconds = time.perf_counter() - evaluation_started
    mean_returns = returns.mean(axis=0)
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "preferences": len(preferences),
        "repeats": args.repeats,
        "hypervolume": hypervolume_2d(mean_returns),
        "sparsity": sparsity(mean_returns),
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
        episode_lengths=lengths.reshape(args.repeats, len(preferences)),
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
