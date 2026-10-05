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
        key_update_interval=5,
        critic_updates_per_transition=0.5,
        actor_updates_per_transition=0.05,
        eval_episodes=None,
        fold_iters=1,
        pareto_step_size=0.5,
        pareto_eval_batch_size=3,
        mjx_impl="warp",
    )
    command = train_command(args, 3, Path("outputs/three_gpu/seed_3"))
    assert command[1] == "train.py"
    assert command[command.index("--num-envs") + 1] == "320"
    assert command[command.index("--seed") + 1] == "3"
    assert "--total-timesteps" in command
    assert "--interpolator-eval-episodes" in command
    assert command[command.index("--key-update-interval") + 1] == "5"
    assert command[command.index("--critic-updates-per-transition") + 1] == "0.5"
    assert command[command.index("--actor-updates-per-transition") + 1] == "0.05"
    assert "--eval-episodes" not in command
    assert command[command.index("--fold-iters") + 1] == "1"
    assert command[command.index("--pareto-step-size") + 1] == "0.5"
    assert command[command.index("--mjx-impl") + 1] == "warp"
    print("multi-GPU launcher check passed")


if __name__ == "__main__":
    main()
