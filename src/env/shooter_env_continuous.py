"""Gymnasium environment for FRC ball shooter with continuous action space."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..config import (
    VELOCITY_MIN,
    VELOCITY_MAX,
    ANGLE_MIN_DEG,
    ANGLE_MAX_DEG,
    AZIMUTH_MIN_DEG,
    AZIMUTH_MAX_DEG,
    HUB_DISTANCE_FROM_WALL,
    ALLIANCE_ZONE_DEPTH,
    ALLIANCE_ZONE_WIDTH,
    MIN_DISTANCE_FROM_HUB,
    HUB_OPENING_HEIGHT,
    HUB_OPENING_HALF_WIDTH,
    REWARD_HIT_BASE,
    REWARD_HIT_CENTER,
    REWARD_MISS_SCALE,
)
from ..physics.projectile import compute_trajectory_3d


class ShooterEnvContinuous(gym.Env):
    """FRC Ball Shooter Environment with Continuous Actions.

    Uses continuous action space for velocity, elevation, and azimuth.
    This is more natural for the problem and allows finer control.

    State: (range, bearing) - distance and angle to target
    Action: Continuous (velocity, elevation, azimuth) normalized to [-1, 1]
    Reward: Shaped based on hit/miss and distance from center

    Each episode simulates a full match with multiple shots from different positions.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, shots_per_episode: int = 50, air_resistance: bool = False):
        super().__init__()

        self.shots_per_episode = shots_per_episode
        self.air_resistance = air_resistance

        self.render_mode = render_mode

        # Continuous action space: 3 values in [-1, 1]
        # Will be scaled to actual ranges
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # Observation space: (range, bearing)
        max_dist = np.sqrt(
            (ALLIANCE_ZONE_DEPTH + HUB_DISTANCE_FROM_WALL) ** 2
            + (ALLIANCE_ZONE_WIDTH / 2) ** 2
        )
        self.observation_space = spaces.Box(
            low=np.array([MIN_DISTANCE_FROM_HUB, -np.pi], dtype=np.float32),
            high=np.array([max_dist, np.pi], dtype=np.float32),
            dtype=np.float32,
        )

        # Action scaling parameters
        self.velocity_range = (VELOCITY_MIN, VELOCITY_MAX)
        self.elevation_range = (np.deg2rad(ANGLE_MIN_DEG), np.deg2rad(ANGLE_MAX_DEG))
        self.azimuth_range = (np.deg2rad(AZIMUTH_MIN_DEG), np.deg2rad(AZIMUTH_MAX_DEG))

        # HUB position
        self.hub_x = HUB_DISTANCE_FROM_WALL
        self.hub_y = ALLIANCE_ZONE_WIDTH / 2

        # Robot position (set in reset)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.distance_to_hub = 0.0
        self.bearing_to_hub = 0.0

        # Episode tracking
        self.current_shot = 0
        self.episode_hits = 0

        self.last_trajectory = None

    def _scale_action(self, action: np.ndarray) -> tuple[float, float, float]:
        """Scale normalized action [-1, 1] to actual values."""
        # action[0] -> velocity
        # action[1] -> elevation
        # action[2] -> azimuth
        velocity = (action[0] + 1) / 2 * (self.velocity_range[1] - self.velocity_range[0]) + self.velocity_range[0]
        elevation = (action[1] + 1) / 2 * (self.elevation_range[1] - self.elevation_range[0]) + self.elevation_range[0]
        azimuth = (action[2] + 1) / 2 * (self.azimuth_range[1] - self.azimuth_range[0]) + self.azimuth_range[0]
        return velocity, elevation, azimuth

    def _generate_new_position(self):
        """Generate a new random robot position."""
        valid_position = False
        while not valid_position:
            self.robot_x = self.np_random.uniform(0, ALLIANCE_ZONE_DEPTH)
            self.robot_y = self.np_random.uniform(0, ALLIANCE_ZONE_WIDTH)

            dx = self.hub_x - self.robot_x
            dy = self.hub_y - self.robot_y
            self.distance_to_hub = np.sqrt(dx * dx + dy * dy)
            self.bearing_to_hub = np.arctan2(dy, dx)

            valid_position = self.distance_to_hub >= MIN_DISTANCE_FROM_HUB

    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset episode tracking
        self.current_shot = 0
        self.episode_hits = 0

        # Generate first position
        self._generate_new_position()

        observation = np.array(
            [self.distance_to_hub, self.bearing_to_hub], dtype=np.float32
        )
        info = {
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "distance": self.distance_to_hub,
            "bearing": self.bearing_to_hub,
            "bearing_deg": np.rad2deg(self.bearing_to_hub),
            "shot": self.current_shot,
        }

        return observation, info

    def step(self, action):
        """Execute action and return result."""
        # Scale action to actual values
        velocity, elevation, azimuth = self._scale_action(action)

        # Compute 3D trajectory
        result = compute_trajectory_3d(
            velocity=velocity,
            elevation=elevation,
            azimuth=azimuth,
            target_distance=self.distance_to_hub,
            target_bearing=self.bearing_to_hub,
            air_resistance=self.air_resistance,
        )

        self.last_trajectory = result

        # Track hits
        if result.hit:
            self.episode_hits += 1

        # Compute reward
        reward = self._compute_reward(result)

        # Increment shot counter
        self.current_shot += 1

        # Episode ends after all shots taken
        terminated = self.current_shot >= self.shots_per_episode
        truncated = False

        # Move to new position for next shot (if not terminated)
        if not terminated:
            self._generate_new_position()

        observation = np.array(
            [self.distance_to_hub, self.bearing_to_hub], dtype=np.float32
        )

        info = {
            "hit": result.hit,
            "height_at_target": result.height_at_target,
            "velocity_y_at_target": result.velocity_y_at_target,
            "lateral_offset": result.lateral_offset,
            "vertical_miss": result.vertical_miss,
            "lateral_miss": result.lateral_miss,
            "total_miss_distance": result.total_miss_distance,
            "center_distance": result.center_distance,
            "velocity": velocity,
            "elevation_deg": np.rad2deg(elevation),
            "azimuth_deg": np.rad2deg(azimuth),
            "target_bearing_deg": np.rad2deg(self.bearing_to_hub),
            "azimuth_error_deg": np.rad2deg(azimuth - self.bearing_to_hub),
            "distance": self.distance_to_hub,
            "shot": self.current_shot,
            "episode_hits": self.episode_hits,
            "episode_hit_rate": self.episode_hits / self.current_shot,
        }

        return observation, reward, terminated, truncated, info

    def _compute_reward(self, result) -> float:
        """Compute shaped reward based on 3D trajectory result."""
        if result.hit:
            max_center_dist = np.sqrt(2) * HUB_OPENING_HALF_WIDTH
            center_bonus = 1.0 - (result.center_distance / max_center_dist)
            center_bonus = max(0.0, center_bonus)
            reward = REWARD_HIT_BASE + REWARD_HIT_CENTER * center_bonus
        else:
            max_miss = 5.0
            normalized_miss = min(result.total_miss_distance / max_miss, 1.0)
            reward = REWARD_MISS_SCALE * normalized_miss

        return reward

    def render(self):
        if self.render_mode == "human":
            pass
        return None

    def close(self):
        pass


# Register the environment
gym.register(
    id="FRCShooterContinuous-v0",
    entry_point="src.env.shooter_env_continuous:ShooterEnvContinuous",
)
