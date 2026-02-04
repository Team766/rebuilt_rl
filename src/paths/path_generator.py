"""Path generation for move-and-shoot training.

Provides robot motion paths that stay within the alliance zone and avoid the HUB.
Each path gives position and velocity at any time t.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..config import (
    ALLIANCE_ZONE_DEPTH,
    ALLIANCE_ZONE_WIDTH,
    DEFAULT_PATH_MIN_DURATION,
    HUB_DISTANCE_FROM_WALL,
    MIN_DISTANCE_FROM_HUB,
)

HUB_X = HUB_DISTANCE_FROM_WALL  # 4.03m
HUB_Y = ALLIANCE_ZONE_WIDTH / 2  # 4.035m


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


def _distance_to_hub(x: float, y: float) -> float:
    """Compute distance from a point to the HUB center."""
    dx = HUB_X - x
    dy = HUB_Y - y
    return np.sqrt(dx * dx + dy * dy)


def generate_straight_line_path(
    np_random: np.random.Generator,
    speed_min: float,
    speed_max: float,
    min_duration: float = DEFAULT_PATH_MIN_DURATION,
) -> StraightLinePath:
    """Generate a random straight-line path within the alliance zone.

    Picks a random start position, heading, and speed, then computes the maximum
    duration before the robot exits the zone or gets too close to the HUB. Retries
    until a path meeting the minimum duration requirement is found.

    Args:
        np_random: Seeded random number generator (from gymnasium env).
        speed_min: Minimum robot speed in m/s.
        speed_max: Maximum robot speed in m/s.
        min_duration: Minimum path duration in seconds.

    Returns:
        A valid StraightLinePath.
    """
    # Handle stationary case
    if speed_max <= 0:
        start_x = np_random.uniform(0.5, ALLIANCE_ZONE_DEPTH - 0.5)
        start_y = np_random.uniform(0.5, ALLIANCE_ZONE_WIDTH - 0.5)
        # Ensure far enough from hub
        while _distance_to_hub(start_x, start_y) < MIN_DISTANCE_FROM_HUB:
            start_x = np_random.uniform(0.5, ALLIANCE_ZONE_DEPTH - 0.5)
            start_y = np_random.uniform(0.5, ALLIANCE_ZONE_WIDTH - 0.5)
        return StraightLinePath(
            start_x, start_y, heading=0.0, speed=0.0, path_duration=min_duration
        )

    max_attempts = 200
    for _ in range(max_attempts):
        # Random start with margin from walls
        margin = 0.3
        start_x = np_random.uniform(margin, ALLIANCE_ZONE_DEPTH - margin)
        start_y = np_random.uniform(margin, ALLIANCE_ZONE_WIDTH - margin)

        if _distance_to_hub(start_x, start_y) < MIN_DISTANCE_FROM_HUB:
            continue

        heading = np_random.uniform(-np.pi, np.pi)
        speed = np_random.uniform(speed_min, speed_max)

        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)

        # Compute max time before exiting alliance zone bounds
        t_max_candidates = []
        if vx > 0:
            t_max_candidates.append((ALLIANCE_ZONE_DEPTH - start_x) / vx)
        elif vx < 0:
            t_max_candidates.append(-start_x / vx)
        # If vx == 0, no X constraint

        if vy > 0:
            t_max_candidates.append((ALLIANCE_ZONE_WIDTH - start_y) / vy)
        elif vy < 0:
            t_max_candidates.append(-start_y / vy)
        # If vy == 0, no Y constraint

        if not t_max_candidates:
            # speed > 0 but both vx and vy are 0 (shouldn't happen)
            continue

        t_bound = min(t_max_candidates)

        # Check hub proximity along the path at 0.1s intervals
        t_hub_limit = t_bound
        for t_check in np.arange(0, t_bound + 0.1, 0.1):
            t_check = min(t_check, t_bound)
            px = start_x + vx * t_check
            py = start_y + vy * t_check
            if _distance_to_hub(px, py) < MIN_DISTANCE_FROM_HUB:
                t_hub_limit = max(0, t_check - 0.1)
                break

        duration = min(t_bound, t_hub_limit)
        if duration >= min_duration:
            return StraightLinePath(start_x, start_y, heading, speed, duration)

    # Fallback: stationary path at a safe position
    return StraightLinePath(
        start_x=1.0,
        start_y=ALLIANCE_ZONE_WIDTH / 2,
        heading=0.0,
        speed=0.0,
        path_duration=min_duration,
    )
