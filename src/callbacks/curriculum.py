"""Curriculum learning callback for move-and-shoot training.

Monitors evaluation performance and progressively increases difficulty
by adjusting the robot's movement speed during training.
"""

from dataclasses import dataclass

from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class CurriculumLevel:
    """Definition of a single curriculum difficulty level."""

    name: str
    speed_min: float  # m/s
    speed_max: float  # m/s
    threshold: float  # estimated hit rate to advance (0.0-1.0)


DEFAULT_CURRICULUM = [
    CurriculumLevel("stationary", 0.0, 0.0, 0.70),
    CurriculumLevel("slow", 0.5, 1.0, 0.60),
    CurriculumLevel("medium", 1.0, 3.0, 0.50),
    CurriculumLevel("fast", 3.0, 5.0, 0.0),  # terminal level
]


class CurriculumCallback(BaseCallback):
    """Callback that advances curriculum difficulty based on eval performance.

    Designed to be used as the ``callback_after_eval`` parameter of
    ``EvalCallback``. Reads ``self.parent.last_mean_reward`` after each
    evaluation round and estimates the hit rate. When the agent exceeds the
    current level's threshold for ``eval_window`` consecutive evaluations,
    the callback advances to the next level by updating all training and
    eval environments via ``env_method("set_curriculum_level", ...)``.

    Usage::

        curriculum_cb = CurriculumCallback(levels=DEFAULT_CURRICULUM)
        eval_cb = EvalCallback(
            eval_env,
            callback_after_eval=curriculum_cb,
            ...
        )
    """

    def __init__(
        self,
        levels: list[CurriculumLevel] | None = None,
        eval_window: int = 3,
        shots_per_episode: int = 50,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.levels = levels or list(DEFAULT_CURRICULUM)
        self.current_level_idx = 0
        self.eval_window = eval_window
        self.shots_per_episode = shots_per_episode
        self.above_threshold_count = 0

    @property
    def current_level(self) -> CurriculumLevel:
        return self.levels[self.current_level_idx]

    def _on_step(self) -> bool:
        """Called after each evaluation by the parent EvalCallback."""
        assert self.parent is not None, "CurriculumCallback must be used with EvalCallback"

        mean_reward = self.parent.last_mean_reward

        # Estimate hit rate from mean reward.
        # Each shot gives ~1.0-2.0 on hit, ~-0.5-0.0 on miss.
        # With shots_per_episode shots, a conservative estimate:
        #   estimated_hit_rate ≈ mean_reward / shots_per_episode
        # (Since max reward per shot is ~2.0 and min is ~-0.5, this underestimates
        #  but is monotonically related to true hit rate.)
        estimated_hit_rate = max(0.0, mean_reward / self.shots_per_episode)

        # Log to tensorboard
        self.logger.record("curriculum/level", self.current_level_idx)
        self.logger.record("curriculum/level_name", self.current_level.name)
        self.logger.record("curriculum/estimated_hit_rate", estimated_hit_rate)
        self.logger.record("curriculum/speed_min", self.current_level.speed_min)
        self.logger.record("curriculum/speed_max", self.current_level.speed_max)

        if self.verbose >= 1:
            print(
                f"[Curriculum] Level {self.current_level_idx} "
                f"({self.current_level.name}), "
                f"est. hit rate: {estimated_hit_rate:.1%}, "
                f"threshold: {self.current_level.threshold:.1%}"
            )

        # Check if we should advance
        if self.current_level_idx < len(self.levels) - 1:
            if estimated_hit_rate >= self.current_level.threshold:
                self.above_threshold_count += 1
            else:
                self.above_threshold_count = 0

            if self.above_threshold_count >= self.eval_window:
                self._advance_level()

        return True

    def _advance_level(self):
        """Move to the next curriculum level."""
        self.current_level_idx += 1
        self.above_threshold_count = 0
        level = self.current_level

        if self.verbose >= 1:
            print(
                f"[Curriculum] ADVANCING to level {self.current_level_idx}: "
                f"{level.name} (speed {level.speed_min}-{level.speed_max} m/s)"
            )

        # Update all training environments
        training_env = self.model.get_env()
        training_env.env_method("set_curriculum_level", level.speed_min, level.speed_max)

        # Also update eval environment
        if hasattr(self.parent, "eval_env") and self.parent.eval_env is not None:
            self.parent.eval_env.env_method(
                "set_curriculum_level", level.speed_min, level.speed_max
            )

    def _on_training_start(self) -> None:
        """Initialize all envs to level 0."""
        level = self.current_level
        training_env = self.model.get_env()
        training_env.env_method("set_curriculum_level", level.speed_min, level.speed_max)
