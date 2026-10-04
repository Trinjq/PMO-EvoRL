"""Executable check for the three-GPU launcher command contract."""

from pathlib import Path
from types import SimpleNamespace

from run_multi_gpu import train_command


def main() -> None:
    args = SimpleNamespace(
        config="configs/mo_td3_walker2d.yaml",
        num_envs_per_gpu=320,
        total_timesteps=1000,
        interpolator_eval_episodes=1,
        eval_episodes=None,
    )
    command = train_command(args, 3, Path("outputs/three_gpu/seed_3"))
    assert command[1] == "train.py"
    assert command[command.index("--num-envs") + 1] == "320"
    assert command[command.index("--seed") + 1] == "3"
    assert "--total-timesteps" in command
    assert "--interpolator-eval-episodes" in command
    assert "--eval-episodes" not in command
    print("multi-GPU launcher check passed")


if __name__ == "__main__":
    main()
