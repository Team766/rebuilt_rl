"""Gymnasium environment for FRC ball shooter training."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..config import (
    VELOCITY_BINS,
    ANGLE_BINS,
    AZIMUTH_BINS,
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
    TOTAL_ACTIONS_3D,
    action_to_velocity_angle,
    action_to_velocity_elevation_azimuth,
)
from ..physics.projectile import compute_trajectory, compute_trajectory_3d


class ShooterEnv(gym.Env):
    """FRC Ball Shooter Environment.

    The robot spawns at a random position in the alliance zone and must
    choose a velocity and angle to shoot the ball into the HUB.

    State: Distance to HUB (scalar)
    Action: Discrete (velocity_bin * ANGLE_BINS + angle_bin)
    Reward: Shaped based on hit/miss and distance from center
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Action space: discrete (velocity x angle grid)
        self.action_space = spaces.Discrete(VELOCITY_BINS * ANGLE_BINS)

        # Observation space: distance to target
        # Min distance is MIN_DISTANCE_FROM_HUB (0.5m)
        # Max distance is roughly diagonal across alliance zone to HUB
        max_dist = np.sqrt(
            (ALLIANCE_ZONE_DEPTH + HUB_DISTANCE_FROM_WALL) ** 2
            + (ALLIANCE_ZONE_WIDTH / 2) ** 2
        )
        self.observation_space = spaces.Box(
            low=np.array([MIN_DISTANCE_FROM_HUB], dtype=np.float32),
            high=np.array([max_dist], dtype=np.float32),
            dtype=np.float32,
        )

        # HUB position (fixed at center of field, distance from alliance wall)
        self.hub_x = HUB_DISTANCE_FROM_WALL
        self.hub_y = ALLIANCE_ZONE_WIDTH / 2  # Center of field width

        # Robot position (set in reset)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.distance_to_hub = 0.0

        # For rendering (optional, future use)
        self.last_trajectory = None

    def reset(self, seed=None, options=None):
        """Reset environment with random robot position."""
        super().reset(seed=seed)

        # Random position in alliance zone
        # x: 0 to ALLIANCE_ZONE_DEPTH (distance from alliance wall)
        # y: 0 to ALLIANCE_ZONE_WIDTH (lateral position)
        valid_position = False
        while not valid_position:
            self.robot_x = self.np_random.uniform(0, ALLIANCE_ZONE_DEPTH)
            self.robot_y = self.np_random.uniform(0, ALLIANCE_ZONE_WIDTH)

            # Compute distance to HUB
            dx = self.hub_x - self.robot_x
            dy = self.hub_y - self.robot_y
            self.distance_to_hub = np.sqrt(dx * dx + dy * dy)

            # Must be at least MIN_DISTANCE_FROM_HUB away
            valid_position = self.distance_to_hub >= MIN_DISTANCE_FROM_HUB

        observation = np.array([self.distance_to_hub], dtype=np.float32)
        info = {
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "distance": self.distance_to_hub,
        }

        return observation, info

    def step(self, action):
        """Execute action and return result.

        Args:
            action: Discrete action index

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode action
        velocity, angle = action_to_velocity_angle(action)

        # Compute trajectory
        result = compute_trajectory(velocity, angle, self.distance_to_hub)

        # Store for rendering
        self.last_trajectory = result

        # Compute reward
        reward = self._compute_reward(result)

        # Episode always ends after one shot
        terminated = True
        truncated = False

        # Next observation (doesn't matter since terminated)
        observation = np.array([self.distance_to_hub], dtype=np.float32)

        info = {
            "hit": result.hit,
            "height_at_target": result.height_at_target,
            "velocity_y_at_target": result.velocity_y_at_target,
            "center_distance": result.center_distance,
            "miss_distance": result.miss_distance,
            "velocity": velocity,
            "angle_deg": np.rad2deg(angle),
            "distance": self.distance_to_hub,
        }

        return observation, reward, terminated, truncated, info

    def _compute_reward(self, result) -> float:
        """Compute shaped reward based on trajectory result."""
        if result.hit:
            # Base reward + bonus for center shots
            center_bonus = 1.0 - (result.center_distance / HUB_OPENING_HALF_WIDTH)
            center_bonus = max(0.0, center_bonus)  # Clamp to [0, 1]
            reward = REWARD_HIT_BASE + REWARD_HIT_CENTER * center_bonus
        else:
            # Penalty based on miss distance
            # Normalize by a reasonable max miss distance
            max_miss = 5.0  # meters
            normalized_miss = min(result.miss_distance / max_miss, 1.0)
            reward = REWARD_MISS_SCALE * normalized_miss

        return reward

    def render(self):
        """Render the environment (placeholder for future visualization)."""
        if self.render_mode == "human":
            # Future: implement matplotlib or pygame visualization
            pass
        return None

    def close(self):
        """Clean up resources."""
        pass


class ShooterEnv3D(gym.Env):
    """FRC Ball Shooter Environment with Turret Aiming (3D).

    The robot spawns at a random position in the alliance zone and must
    choose a velocity, elevation angle, AND azimuth (turret) angle to
    shoot the ball into the HUB.

    State: (range, bearing) - distance and angle to target
    Action: Discrete (velocity_bin * ANGLE_BINS * AZIMUTH_BINS + elev_bin * AZIMUTH_BINS + azim_bin)
    Reward: Shaped based on hit/miss and distance from center
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Action space: discrete (velocity x elevation x azimuth grid)
        # 10 x 15 x 180 = 27,000 actions
        self.action_space = spaces.Discrete(TOTAL_ACTIONS_3D)

        # Observation space: (range, bearing)
        # Range: MIN_DISTANCE_FROM_HUB to max diagonal distance
        # Bearing: -pi to pi radians
        max_dist = np.sqrt(
            (ALLIANCE_ZONE_DEPTH + HUB_DISTANCE_FROM_WALL) ** 2
            + (ALLIANCE_ZONE_WIDTH / 2) ** 2
        )
        self.observation_space = spaces.Box(
            low=np.array([MIN_DISTANCE_FROM_HUB, -np.pi], dtype=np.float32),
            high=np.array([max_dist, np.pi], dtype=np.float32),
            dtype=np.float32,
        )

        # HUB position (fixed at center of field, distance from alliance wall)
        self.hub_x = HUB_DISTANCE_FROM_WALL
        self.hub_y = ALLIANCE_ZONE_WIDTH / 2  # Center of field width

        # Robot position (set in reset)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.distance_to_hub = 0.0
        self.bearing_to_hub = 0.0

        # For rendering (optional, future use)
        self.last_trajectory = None

    def reset(self, seed=None, options=None):
        """Reset environment with random robot position."""
        super().reset(seed=seed)

        # Random position in alliance zone
        valid_position = False
        while not valid_position:
            self.robot_x = self.np_random.uniform(0, ALLIANCE_ZONE_DEPTH)
            self.robot_y = self.np_random.uniform(0, ALLIANCE_ZONE_WIDTH)

            # Compute distance and bearing to HUB
            dx = self.hub_x - self.robot_x
            dy = self.hub_y - self.robot_y
            self.distance_to_hub = np.sqrt(dx * dx + dy * dy)
            self.bearing_to_hub = np.arctan2(dy, dx)

            # Must be at least MIN_DISTANCE_FROM_HUB away
            valid_position = self.distance_to_hub >= MIN_DISTANCE_FROM_HUB

        observation = np.array(
            [self.distance_to_hub, self.bearing_to_hub], dtype=np.float32
        )
        info = {
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "distance": self.distance_to_hub,
            "bearing": self.bearing_to_hub,
            "bearing_deg": np.rad2deg(self.bearing_to_hub),
        }

        return observation, info

    def step(self, action):
        """Execute action and return result.

        Args:
            action: Discrete action index

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode action
        velocity, elevation, azimuth = action_to_velocity_elevation_azimuth(action)

        # Compute 3D trajectory
        result = compute_trajectory_3d(
            velocity=velocity,
            elevation=elevation,
            azimuth=azimuth,
            target_distance=self.distance_to_hub,
            target_bearing=self.bearing_to_hub,
        )

        # Store for rendering
        self.last_trajectory = result

        # Compute reward
        reward = self._compute_reward(result)

        # Episode always ends after one shot
        terminated = True
        truncated = False

        # Next observation (doesn't matter since terminated)
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
        }

        return observation, reward, terminated, truncated, info

    def _compute_reward(self, result) -> float:
        """Compute shaped reward based on 3D trajectory result."""
        if result.hit:
            # Base reward + bonus for center shots
            # Use 2D center distance (vertical + lateral)
            max_center_dist = np.sqrt(2) * HUB_OPENING_HALF_WIDTH
            center_bonus = 1.0 - (result.center_distance / max_center_dist)
            center_bonus = max(0.0, center_bonus)  # Clamp to [0, 1]
            reward = REWARD_HIT_BASE + REWARD_HIT_CENTER * center_bonus
        else:
            # Penalty based on total miss distance
            max_miss = 5.0  # meters
            normalized_miss = min(result.total_miss_distance / max_miss, 1.0)
            reward = REWARD_MISS_SCALE * normalized_miss

        return reward

    def render(self):
        """Render the environment (placeholder for future visualization)."""
        if self.render_mode == "human":
            pass
        return None

    def close(self):
        """Clean up resources."""
        pass


# Register the environments
gym.register(
    id="FRCShooter-v0",
    entry_point="src.env.shooter_env:ShooterEnv",
)

gym.register(
    id="FRCShooter3D-v0",
    entry_point="src.env.shooter_env:ShooterEnv3D",
)
