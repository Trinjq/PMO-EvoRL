"""Executable checks for source-parity and speed-first update schedules."""

from omegaconf import OmegaConf

from train import configure_update_schedule


def check(num_envs: int, critic_ratio: float, actor_ratio: float, expected) -> None:
    config = OmegaConf.create(
        {
            "num_envs": num_envs,
            "critic_updates_per_transition": critic_ratio,
            "actor_updates_per_transition": actor_ratio,
        }
    )
    configure_update_schedule(config)
    assert (config.num_updates_per_iter, config.actor_update_interval) == expected


def main() -> None:
    check(320, 1.0, 0.1, (32, 10))
    check(640, 1.0, 0.1, (64, 10))
    check(640, 0.5, 0.05, (32, 10))
    try:
        check(10, 1.0, 0.03, None)
    except ValueError:
        pass
    else:
        raise AssertionError("non-integer update schedules must be rejected")
    print("training schedule checks passed")


if __name__ == "__main__":
    main()
