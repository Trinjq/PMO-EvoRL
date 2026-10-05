"""Run independent GPU-native PD-MORL seeds concurrently."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def train_command(args, seed: int, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "train.py",
        "--config",
        args.config,
        "--num-envs",
        str(args.num_envs_per_gpu),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    for name in (
        "total_timesteps",
        "interpolator_eval_episodes",
        "key_update_interval",
        "critic_updates_per_transition",
        "actor_updates_per_transition",
        "eval_episodes",
        "fold_iters",
        "pareto_step_size",
        "pareto_eval_batch_size",
    ):
        value = getattr(args, name)
        if value is not None:
            command.extend((f"--{name.replace('_', '-')}", str(value)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--config", default="configs/mo_td3_walker2d.yaml")
    parser.add_argument("--num-envs-per-gpu", type=int, default=320)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--output-root", default="outputs/three_gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--interpolator-eval-episodes", type=int)
    parser.add_argument("--key-update-interval", type=int)
    parser.add_argument("--critic-updates-per-transition", type=float)
    parser.add_argument("--actor-updates-per-transition", type=float)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--fold-iters", type=int)
    parser.add_argument("--pareto-step-size", type=float)
    parser.add_argument("--pareto-eval-batch-size", type=int)
    args = parser.parse_args()

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        parser.error("--devices must contain at least one GPU id")
    if args.num_envs_per_gpu % 10:
        parser.error("--num-envs-per-gpu must be divisible by 10")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    processes = []
    logs = []
    try:
        for offset, device in enumerate(devices):
            seed = args.seed + offset
            output_dir = output_root / f"seed_{seed}"
            output_dir.mkdir(parents=True, exist_ok=True)
            log = (output_dir / "console.log").open("w", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["JAX_PLATFORMS"] = "cuda"
            process = subprocess.Popen(
                train_command(args, seed, output_dir),
                cwd=Path(__file__).parent,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append((device, seed, process))
            logs.append(log)
            print(
                {
                    "gpu": device,
                    "seed": seed,
                    "pid": process.pid,
                    "log": str(output_dir / "console.log"),
                },
                flush=True,
            )

        failures = []
        for device, seed, process in processes:
            returncode = process.wait()
            if returncode:
                failures.append((device, seed, returncode))
        if failures:
            raise SystemExit(f"GPU workers failed: {failures}")
    except BaseException:
        for _, _, process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
