"""Gymnasium environment for FRC ball shooter with continuous action space."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import (
    ALLIANCE_ZONE_DEPTH,
    ALLIANCE_ZONE_WIDTH,
    ANGLE_MAX_DEG,
    ANGLE_MIN_DEG,
    AZIMUTH_MAX_DEG,
    AZIMUTH_MIN_DEG,
    BALL_DIAMETER,
    DEFAULT_SHOT_INTERVAL,
    HUB_DISTANCE_FROM_WALL,
    HUB_OPENING_HALF_WIDTH,
    HUB_OPENING_HEIGHT,
    MIN_DISTANCE_FROM_HUB,
    REWARD_HIT_BASE,
    REWARD_HIT_CENTER,
    REWARD_MISS_SCALE,
    ROBOT_MAX_SPEED,
    VELOCITY_MAX,
    VELOCITY_MIN,
)
from ..physics.projectile import compute_trajectory_3d, compute_trajectory_3d_moving


class ShooterEnvContinuous(gym.Env):
    """FRC Ball Shooter Environment with Continuous Actions.

    Uses continuous action space for velocity, elevation, and azimuth.
    This is more natural for the problem and allows finer control.

    State: (range, bearing) - distance and angle to target
           (range, bearing, vx, vy) - when move_and_shoot is enabled
    Action: Continuous (velocity, elevation, azimuth) normalized to [-1, 1]
    Reward: Shaped based on hit/miss and distance from center

    Each episode simulates a full match with multiple shots from different positions.
    When move_and_shoot is enabled, the robot follows a path during the episode
    and the ball inherits the robot's velocity at launch.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        shots_per_episode: int = 50,
        air_resistance: bool = False,
        move_and_shoot: bool = False,
        shot_interval: float = DEFAULT_SHOT_INTERVAL,
        speed_min: float = 0.0,
        speed_max: float = 0.0,
    ):
        super().__init__()

        self.shots_per_episode = shots_per_episode
        self.air_resistance = air_resistance
        self.move_and_shoot = move_and_shoot
        self.shot_interval = shot_interval
        self.speed_min = speed_min
        self.speed_max = speed_max

        self.render_mode = render_mode

        # Continuous action space: 3 values in [-1, 1]
        # Will be scaled to actual ranges
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # Observation space depends on mode
        max_dist = np.sqrt(
            (ALLIANCE_ZONE_DEPTH + HUB_DISTANCE_FROM_WALL) ** 2
            + (ALLIANCE_ZONE_WIDTH / 2) ** 2
        )

        if self.move_and_shoot:
            # [distance, bearing, robot_vx, robot_vy]
            self.observation_space = spaces.Box(
                low=np.array(
                    [MIN_DISTANCE_FROM_HUB, -np.pi, -ROBOT_MAX_SPEED, -ROBOT_MAX_SPEED],
                    dtype=np.float32,
                ),
                high=np.array(
                    [max_dist, np.pi, ROBOT_MAX_SPEED, ROBOT_MAX_SPEED],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
        else:
            # [distance, bearing]
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

        # Robot state (set in reset)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        self.distance_to_hub = 0.0
        self.bearing_to_hub = 0.0

        # Episode tracking
        self.current_shot = 0
        self.episode_hits = 0

        # Path state (for move_and_shoot mode)
        self.current_path = None
        self.path_time = 0.0

        self.last_trajectory = None

    def _scale_action(self, action: np.ndarray) -> tuple[float, float, float]:
        """Scale normalized action [-1, 1] to actual values."""
        # action[0] -> velocity
        # action[1] -> elevation
        # action[2] -> azimuth
        v_lo, v_hi = self.velocity_range
        e_lo, e_hi = self.elevation_range
        a_lo, a_hi = self.azimuth_range
        velocity = (action[0] + 1) / 2 * (v_hi - v_lo) + v_lo
        elevation = (action[1] + 1) / 2 * (e_hi - e_lo) + e_lo
        azimuth = (action[2] + 1) / 2 * (a_hi - a_lo) + a_lo
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

        self.robot_vx = 0.0
        self.robot_vy = 0.0

    def _update_position_from_path(self):
        """Update robot position, bearing, and velocity from current path state."""
        state = self.current_path.state_at(self.path_time)
        self.robot_x = state.x
        self.robot_y = state.y
        self.robot_vx = state.vx
        self.robot_vy = state.vy

        dx = self.hub_x - self.robot_x
        dy = self.hub_y - self.robot_y
        self.distance_to_hub = np.sqrt(dx * dx + dy * dy)
        self.bearing_to_hub = np.arctan2(dy, dx)

    def _make_observation(self) -> np.ndarray:
        """Build observation array based on mode."""
        if self.move_and_shoot:
            return np.array(
                [self.distance_to_hub, self.bearing_to_hub, self.robot_vx, self.robot_vy],
                dtype=np.float32,
            )
        else:
            return np.array(
                [self.distance_to_hub, self.bearing_to_hub],
                dtype=np.float32,
            )

    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset episode tracking
        self.current_shot = 0
        self.episode_hits = 0

        # Generate initial position
        if self.move_and_shoot and self.speed_max > 0:
            from ..paths.path_generator import generate_straight_line_path

            self.current_path = generate_straight_line_path(
                self.np_random,
                speed_min=self.speed_min,
                speed_max=self.speed_max,
                path_duration=self.shots_per_episode * self.shot_interval,
            )
            self.path_time = 0.0
            self._update_position_from_path()
        else:
            self.current_path = None
            self._generate_new_position()

        observation = self._make_observation()
        info = {
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "distance": self.distance_to_hub,
            "bearing": self.bearing_to_hub,
            "bearing_deg": np.rad2deg(self.bearing_to_hub),
            "shot": self.current_shot,
        }
        if self.move_and_shoot:
            info["robot_vx"] = self.robot_vx
            info["robot_vy"] = self.robot_vy

        return observation, info

    def step(self, action):
        """Execute action and return result."""
        # Scale action to actual values
        velocity, elevation, azimuth = self._scale_action(action)

        # Compute trajectory — use moving physics when robot has velocity
        if self.move_and_shoot and self.current_path is not None:
            result = compute_trajectory_3d_moving(
                launch_velocity=velocity,
                elevation=elevation,
                azimuth=azimuth,
                target_distance=self.distance_to_hub,
                target_bearing=self.bearing_to_hub,
                robot_vx=self.robot_vx,
                robot_vy=self.robot_vy,
                air_resistance=self.air_resistance,
            )
        else:
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

        # Move to next position for next shot (if not terminated)
        if not terminated:
            if self.move_and_shoot and self.current_path is not None:
                self.path_time += self.shot_interval
                self._update_position_from_path()
            else:
                self._generate_new_position()

        observation = self._make_observation()

        # velocity_y_at_target field name differs between result types
        vy_at_target = getattr(
            result, "velocity_z_at_target", getattr(result, "velocity_y_at_target", 0.0)
        )

        info = {
            "hit": result.hit,
            "height_at_target": result.height_at_target,
            "velocity_y_at_target": vy_at_target,
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
        if self.move_and_shoot:
            info["robot_vx"] = self.robot_vx
            info["robot_vy"] = self.robot_vy

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

        # Penalize peak height above hub opening + 2 ball diameters buffer.
        # Continuous gradient: flatter is always better, but close-range lobs
        # are fine since their excess is small. -0.1 per meter of excess.
        traj_z = getattr(result, "trajectory_z", result.trajectory_y)
        peak_height = float(np.max(traj_z))
        height_baseline = HUB_OPENING_HEIGHT + 2 * BALL_DIAMETER
        excess_height = max(0.0, peak_height - height_baseline)
        reward -= 0.1 * excess_height

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
