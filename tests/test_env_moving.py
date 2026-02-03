"""Tests for move-and-shoot environment mode."""

import numpy as np
import pytest

from src.env.shooter_env_continuous import ShooterEnvContinuous


class TestMoveAndShootEnvCreation:
    def test_stationary_observation_space(self):
        """Default env has 2D observation."""
        env = ShooterEnvContinuous()
        assert env.observation_space.shape == (2,)
        env.close()

    def test_moving_observation_space(self):
        """Move-and-shoot env has 4D observation."""
        env = ShooterEnvContinuous(move_and_shoot=True)
        assert env.observation_space.shape == (4,)
        env.close()

    def test_action_space_unchanged(self):
        """Action space is still 3D regardless of mode."""
        env = ShooterEnvContinuous(move_and_shoot=True)
        assert env.action_space.shape == (3,)
        env.close()

    def test_default_stationary_backward_compatible(self):
        """Default env without move_and_shoot works identically to before."""
        env = ShooterEnvContinuous()
        obs, info = env.reset(seed=42)
        assert obs.shape == (2,)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (2,)
        assert "robot_vx" not in info
        env.close()


class TestMoveAndShootReset:
    def test_reset_returns_4d_observation(self):
        env = ShooterEnvContinuous(move_and_shoot=True, speed_min=1.0, speed_max=2.0)
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,)
        assert env.observation_space.contains(obs)
        env.close()

    def test_reset_with_zero_speed_is_stationary(self):
        """When speed_max=0, robot velocity in obs should be 0."""
        env = ShooterEnvContinuous(move_and_shoot=True, speed_min=0.0, speed_max=0.0)
        obs, info = env.reset(seed=42)
        assert obs[2] == pytest.approx(0.0, abs=0.01)
        assert obs[3] == pytest.approx(0.0, abs=0.01)
        env.close()

    def test_reset_with_speed_has_nonzero_velocity(self):
        """When speed_max > 0, obs should contain nonzero velocity."""
        env = ShooterEnvContinuous(move_and_shoot=True, speed_min=2.0, speed_max=3.0)
        obs, info = env.reset(seed=42)
        speed = np.sqrt(obs[2] ** 2 + obs[3] ** 2)
        assert speed > 0.1
        env.close()

    def test_info_contains_velocity(self):
        """Info dict should contain robot_vx and robot_vy when move_and_shoot."""
        env = ShooterEnvContinuous(move_and_shoot=True, speed_min=1.0, speed_max=2.0)
        obs, info = env.reset(seed=42)
        assert "robot_vx" in info
        assert "robot_vy" in info
        env.close()

    def test_reset_seed_reproducible(self):
        """Same seed produces identical state."""
        env = ShooterEnvContinuous(move_and_shoot=True, speed_min=1.0, speed_max=2.0)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_almost_equal(obs1, obs2)
        env.close()


class TestMoveAndShootStep:
    def test_step_returns_valid_result(self):
        env = ShooterEnvContinuous(move_and_shoot=True, speed_min=1.0, speed_max=2.0)
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2.shape == (4,)
        assert not np.isnan(reward)
        assert isinstance(info["hit"], (bool, np.bool_))
        env.close()

    def test_position_changes_between_shots(self):
        """Distance to hub should change because robot is moving."""
        env = ShooterEnvContinuous(
            move_and_shoot=True, speed_min=2.0, speed_max=3.0, shots_per_episode=10
        )
        obs1, _ = env.reset(seed=42)
        dist1 = obs1[0]
        action = env.action_space.sample()
        obs2, _, _, _, _ = env.step(action)
        dist2 = obs2[0]
        assert dist1 != pytest.approx(dist2, abs=0.001)
        env.close()

    def test_episode_completes(self):
        """Episode terminates after shots_per_episode shots."""
        n_shots = 5
        env = ShooterEnvContinuous(
            move_and_shoot=True, speed_min=1.0, speed_max=2.0, shots_per_episode=n_shots
        )
        obs, _ = env.reset(seed=42)
        for i in range(n_shots):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        env.close()

    def test_not_terminated_before_episode_end(self):
        """Episode should not terminate before shots_per_episode."""
        env = ShooterEnvContinuous(
            move_and_shoot=True, speed_min=1.0, speed_max=2.0, shots_per_episode=10
        )
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is False
        env.close()

    def test_step_info_contains_velocity(self):
        """Step info dict should contain robot_vx and robot_vy."""
        env = ShooterEnvContinuous(move_and_shoot=True, speed_min=1.0, speed_max=2.0)
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "robot_vx" in info
        assert "robot_vy" in info
        env.close()

    def test_all_actions_produce_valid_results(self):
        """Random actions should not produce NaN or crash."""
        env = ShooterEnvContinuous(
            move_and_shoot=True, speed_min=1.0, speed_max=3.0, shots_per_episode=20
        )
        obs, _ = env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert not np.any(np.isnan(obs))
            assert not np.isnan(reward)
            if terminated:
                break
        env.close()


class TestSetCurriculumLevel:
    def test_set_curriculum_level(self):
        env = ShooterEnvContinuous(move_and_shoot=True)
        env.set_curriculum_level(2.0, 4.0)
        assert env.speed_min == 2.0
        assert env.speed_max == 4.0
        env.close()

    def test_curriculum_level_takes_effect_on_reset(self):
        """After setting speed, reset should generate a moving path."""
        env = ShooterEnvContinuous(move_and_shoot=True, speed_min=0.0, speed_max=0.0)
        obs1, _ = env.reset(seed=42)
        assert obs1[2] == pytest.approx(0.0, abs=0.01)
        assert obs1[3] == pytest.approx(0.0, abs=0.01)

        env.set_curriculum_level(2.0, 3.0)
        obs2, _ = env.reset(seed=43)
        speed = np.sqrt(obs2[2] ** 2 + obs2[3] ** 2)
        assert speed > 0.1
        env.close()


class TestGymnasiumCompatibilityMoving:
    def test_check_env_stationary(self):
        """Gymnasium check_env passes for move_and_shoot with zero speed."""
        from gymnasium.utils.env_checker import check_env

        env = ShooterEnvContinuous(move_and_shoot=True, speed_min=0.0, speed_max=0.0)
        check_env(env, warn=True)
        env.close()

    def test_check_env_moving(self):
        """Gymnasium check_env passes for move_and_shoot with nonzero speed."""
        from gymnasium.utils.env_checker import check_env

        env = ShooterEnvContinuous(move_and_shoot=True, speed_min=1.0, speed_max=2.0)
        check_env(env, warn=True)
        env.close()

    def test_episode_loop(self):
        """Standard RL episode loop works correctly."""
        env = ShooterEnvContinuous(
            move_and_shoot=True, speed_min=1.0, speed_max=2.0, shots_per_episode=5
        )
        obs, info = env.reset(seed=42)
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        assert isinstance(total_reward, float)
        env.close()
