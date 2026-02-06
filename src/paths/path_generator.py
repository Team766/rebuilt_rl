"""Path generation for move-and-shoot training.

Provides robot motion paths that stay within the alliance zone.
Each path gives position and velocity at any time t.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..config import (
    ALLIANCE_ZONE_DEPTH,
    ALLIANCE_ZONE_WIDTH,
    HUB_DISTANCE_FROM_WALL,
    MIN_DISTANCE_FROM_HUB,
)

HUB_X = HUB_DISTANCE_FROM_WALL  # 4.03m
HUB_Y = ALLIANCE_ZONE_WIDTH / 2  # 4.035m

# Default path duration covers a full 50-shot episode at 0.5s intervals.
DEFAULT_PATH_DURATION = 25.0


@dataclass
class PathState:
    """Robot state at a point along the path."""

    x: float  # field X position (meters)
    y: float  # field Y position (meters)
    vx: float  # X velocity (m/s)
    vy: float  # Y velocity (m/s)
    speed: float  # magnitude of velocity (m/s)


class RobotPath(ABC):
    """Abstract base class for robot motion paths."""

    @abstractmethod
    def state_at(self, t: float) -> PathState:
        """Return the robot state at time t seconds from path start."""
        ...

    @abstractmethod
    def duration(self) -> float:
        """Total duration of the path in seconds."""
        ...

    def is_valid_at(self, t: float) -> bool:
        """Check if position at time t is within field bounds and far enough from hub."""
        s = self.state_at(t)
        if not (0 <= s.x <= ALLIANCE_ZONE_DEPTH and 0 <= s.y <= ALLIANCE_ZONE_WIDTH):
            return False
        dx = HUB_X - s.x
        dy = HUB_Y - s.y
        dist = np.sqrt(dx * dx + dy * dy)
        return dist >= MIN_DISTANCE_FROM_HUB


class StraightLinePath(RobotPath):
    """Robot moves in a straight line at constant speed."""

    def __init__(
        self,
        start_x: float,
        start_y: float,
        heading: float,  # radians, 0 = +X direction
        speed: float,  # m/s
        path_duration: float,  # seconds
    ):
        self.start_x = start_x
        self.start_y = start_y
        self.heading = heading
        self._speed = speed
        self._duration = path_duration
        self._vx = speed * np.cos(heading)
        self._vy = speed * np.sin(heading)

    def state_at(self, t: float) -> PathState:
        t = np.clip(t, 0.0, self._duration)
        return PathState(
            x=self.start_x + self._vx * t,
            y=self.start_y + self._vy * t,
            vx=self._vx,
            vy=self._vy,
            speed=self._speed,
        )

    def duration(self) -> float:
        return self._duration


def _fold(raw: float, length: float) -> tuple[float, int]:
    """Fold a value into [0, length] with elastic reflections.

    Returns (position, direction) where direction is +1 (original) or -1 (reflected).
    """
    cycle = 2 * length
    phase = raw % cycle
    if phase < 0:
        phase += cycle
    if phase <= length:
        return phase, 1
    else:
        return cycle - phase, -1


class BouncingLinePath(RobotPath):
    """Robot moves in a straight line, bouncing elastically off zone walls.

    Uses triangle-wave folding so the robot stays within the alliance zone
    for any duration. Velocity direction flips on each wall bounce but
    speed magnitude stays constant.
    """

    def __init__(
        self,
        start_x: float,
        start_y: float,
        heading: float,  # radians, 0 = +X direction
        speed: float,  # m/s
        path_duration: float,  # seconds
    ):
        self.start_x = start_x
        self.start_y = start_y
        self.heading = heading
        self._speed = speed
        self._duration = path_duration
        self._vx = speed * np.cos(heading)
        self._vy = speed * np.sin(heading)

    def state_at(self, t: float) -> PathState:
        t = np.clip(t, 0.0, self._duration)
        raw_x = self.start_x + self._vx * t
        raw_y = self.start_y + self._vy * t
        x, sx = _fold(raw_x, ALLIANCE_ZONE_DEPTH)
        y, sy = _fold(raw_y, ALLIANCE_ZONE_WIDTH)
        return PathState(
            x=x,
            y=y,
            vx=self._vx * sx,
            vy=self._vy * sy,
            speed=self._speed,
        )

    def duration(self) -> float:
        return self._duration


def _distance_to_hub(x: float, y: float) -> float:
    """Compute distance from a point to the HUB center."""
    dx = HUB_X - x
    dy = HUB_Y - y
    return np.sqrt(dx * dx + dy * dy)


def generate_path(
    np_random: np.random.Generator,
    speed_min: float,
    speed_max: float,
    path_duration: float = DEFAULT_PATH_DURATION,
) -> RobotPath:
    """Generate a random path within the alliance zone.

    For moving robots, returns a BouncingLinePath that elastically reflects
    off zone walls, keeping the robot in motion for the full duration.
    For stationary robots (speed_max <= 0), returns a StraightLinePath at
    a fixed position.

    Args:
        np_random: Seeded random number generator (from gymnasium env).
        speed_min: Minimum robot speed in m/s.
        speed_max: Maximum robot speed in m/s.
        path_duration: Total path duration in seconds.

    Returns:
        A RobotPath valid for the requested duration.
    """
    # Pick a random start position far enough from hub
    margin = 0.3
    start_x = np_random.uniform(margin, ALLIANCE_ZONE_DEPTH - margin)
    start_y = np_random.uniform(margin, ALLIANCE_ZONE_WIDTH - margin)
    while _distance_to_hub(start_x, start_y) < MIN_DISTANCE_FROM_HUB:
        start_x = np_random.uniform(margin, ALLIANCE_ZONE_DEPTH - margin)
        start_y = np_random.uniform(margin, ALLIANCE_ZONE_WIDTH - margin)

    # Stationary case
    if speed_max <= 0:
        return StraightLinePath(
            start_x, start_y, heading=0.0, speed=0.0, path_duration=path_duration
        )

    # Moving case: bouncing path stays in bounds for any duration
    heading = np_random.uniform(-np.pi, np.pi)
    speed = np_random.uniform(speed_min, speed_max)
    return BouncingLinePath(start_x, start_y, heading, speed, path_duration)


# Keep old name as alias for backwards compatibility with tests/callers
generate_straight_line_path = generate_path
