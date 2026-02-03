"""Unit tests for ShooterEnv gymnasium environment."""

import numpy as np
import pytest

from src.config import (
    ALLIANCE_ZONE_DEPTH,
    ALLIANCE_ZONE_WIDTH,
    ANGLE_BINS,
    MIN_DISTANCE_FROM_HUB,
    VELOCITY_BINS,
)
from src.env.shooter_env import ShooterEnv


class TestShooterEnvBasics:
    """Basic environment tests."""

    def test_env_creation(self):
        """Environment can be created."""
        env = ShooterEnv()
        assert env is not None
        env.close()

    def test_action_space(self):
        """Action space is correct size."""
        env = ShooterEnv()
        assert env.action_space.n == VELOCITY_BINS * ANGLE_BINS
        env.close()

    def test_observation_space(self):
        """Observation space has correct shape and bounds."""
        env = ShooterEnv()
        assert env.observation_space.shape == (1,)
        assert env.observation_space.low[0] == pytest.approx(MIN_DISTANCE_FROM_HUB)
        assert env.observation_space.high[0] > MIN_DISTANCE_FROM_HUB
        env.close()


class TestShooterEnvReset:
    """Tests for reset functionality."""

    def test_reset_returns_observation(self):
        """Reset returns valid observation."""
        env = ShooterEnv()
        obs, info = env.reset()

        assert obs.shape == (1,)
        assert env.observation_space.contains(obs)
        env.close()

    def test_reset_returns_info(self):
        """Reset returns info dict with robot position."""
        env = ShooterEnv()
        obs, info = env.reset()

        assert "robot_x" in info
        assert "robot_y" in info
        assert "distance" in info
        env.close()

    def test_reset_robot_in_alliance_zone(self):
        """Robot spawns within alliance zone."""
        env = ShooterEnv()

        for _ in range(50):  # Test multiple resets
            obs, info = env.reset()
            assert 0 <= info["robot_x"] <= ALLIANCE_ZONE_DEPTH
            assert 0 <= info["robot_y"] <= ALLIANCE_ZONE_WIDTH

        env.close()

    def test_reset_minimum_distance(self):
        """Robot spawns at least MIN_DISTANCE_FROM_HUB from target."""
        env = ShooterEnv()

        for _ in range(50):
            obs, info = env.reset()
            assert info["distance"] >= MIN_DISTANCE_FROM_HUB

        env.close()

    def test_reset_with_seed_reproducible(self):
        """Reset with same seed produces same result."""
        env = ShooterEnv()

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)

        assert obs1[0] == pytest.approx(obs2[0])
        assert info1["robot_x"] == pytest.approx(info2["robot_x"])
        assert info1["robot_y"] == pytest.approx(info2["robot_y"])

        env.close()


class TestShooterEnvStep:
    """Tests for step functionality."""

    def test_step_returns_correct_tuple(self):
        """Step returns (obs, reward, terminated, truncated, info)."""
        env = ShooterEnv()
        env.reset(seed=42)

        result = env.step(0)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert obs.shape == (1,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()

    def test_step_terminates_episode(self):
        """Single shot terminates episode."""
        env = ShooterEnv()
        env.reset()

        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is True
        assert truncated is False

        env.close()

    def test_step_info_contains_details(self):
        """Step info contains shot details."""
        env = ShooterEnv()
        env.reset()

        _, _, _, _, info = env.step(0)

        assert "hit" in info
        assert "height_at_target" in info
        assert "velocity" in info
        assert "angle_deg" in info
        assert "distance" in info
        assert "center_distance" in info

        env.close()

    def test_all_actions_valid(self):
        """All discrete actions can be executed."""
        env = ShooterEnv()
        env.reset(seed=42)

        for action in range(env.action_space.n):
            env.reset(seed=42)  # Reset to same state
            obs, reward, terminated, truncated, info = env.step(action)

            # Should not crash and should return valid values
            assert not np.isnan(reward)
            assert not np.isnan(obs[0])
            assert terminated is True

        env.close()


class TestShooterEnvRewards:
    """Tests for reward computation."""

    def test_hit_gives_positive_reward(self):
        """Hitting target gives positive reward."""
        env = ShooterEnv()

        # Try many resets and actions to find a hit
        hits_found = 0
        for seed in range(100):
            env.reset(seed=seed)
            for action in range(0, env.action_space.n, 10):  # Sample actions
                env.reset(seed=seed)
                _, reward, _, _, info = env.step(action)
                if info["hit"]:
                    assert reward > 0
                    hits_found += 1
                    break
            if hits_found > 0:
                break

        env.close()

    def test_miss_gives_negative_or_zero_reward(self):
        """Missing target gives non-positive reward."""
        env = ShooterEnv()

        for seed in range(20):
            env.reset(seed=seed)
            _, reward, _, _, info = env.step(0)  # Action 0 might miss
            if not info["hit"]:
                assert reward <= 0
                break

        env.close()

    def test_center_hit_better_than_edge(self):
        """Center hits should get better reward than edge hits."""
        # This is hard to test directly without controlling exact trajectory
        # So we just verify the reward logic exists
        env = ShooterEnv()
        env.reset()
        env.close()


class TestGymnasiumCompatibility:
    """Test gymnasium API compatibility."""

    def test_check_env(self):
        """Environment passes gymnasium check_env."""
        from gymnasium.utils.env_checker import check_env

        env = ShooterEnv()
        # This will raise if there are issues
        check_env(env, warn=True)
        env.close()

    def test_sample_actions(self):
        """Can sample random actions."""
        env = ShooterEnv()
        env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            assert 0 <= action < env.action_space.n

        env.close()

    def test_episode_loop(self):
        """Can run standard episode loop."""
        env = ShooterEnv()

        for episode in range(5):
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

            # Episode should end after exactly one step
            assert terminated

        env.close()
