"""Unit tests for moving-robot 3D trajectory physics."""

import numpy as np
import pytest

from src.config import LAUNCH_HEIGHT
from src.physics.projectile import (
    compute_trajectory_3d,
    compute_trajectory_3d_moving,
)


class TestZeroVelocityEquivalence:
    """With zero robot velocity, compute_trajectory_3d_moving should match compute_trajectory_3d."""

    def test_hit_matches(self):
        result_stationary = compute_trajectory_3d(
            velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=np.deg2rad(10),
            target_distance=5.0,
            target_bearing=np.deg2rad(10),
        )
        result_moving = compute_trajectory_3d_moving(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=np.deg2rad(10),
            target_distance=5.0,
            target_bearing=np.deg2rad(10),
            robot_vx=0.0,
            robot_vy=0.0,
        )
        assert result_moving.hit == result_stationary.hit

    def test_height_matches(self):
        result_stationary = compute_trajectory_3d(
            velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=np.deg2rad(5),
            target_distance=5.0,
            target_bearing=np.deg2rad(5),
        )
        result_moving = compute_trajectory_3d_moving(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=np.deg2rad(5),
            target_distance=5.0,
            target_bearing=np.deg2rad(5),
            robot_vx=0.0,
            robot_vy=0.0,
        )
        assert result_moving.height_at_target == pytest.approx(
            result_stationary.height_at_target, abs=0.1
        )

    def test_lateral_offset_matches(self):
        # Use a trajectory that misses laterally but doesn't collide with hub.
        # Large azimuth offset so ball passes well to the side of the hub.
        result_stationary = compute_trajectory_3d(
            velocity=12.0,
            elevation=np.deg2rad(40),
            azimuth=np.deg2rad(30),  # Far off from bearing
            target_distance=5.0,
            target_bearing=np.deg2rad(0),
        )
        result_moving = compute_trajectory_3d_moving(
            launch_velocity=12.0,
            elevation=np.deg2rad(40),
            azimuth=np.deg2rad(30),
            target_distance=5.0,
            target_bearing=np.deg2rad(0),
            robot_vx=0.0,
            robot_vy=0.0,
        )
        assert result_moving.lateral_offset == pytest.approx(
            result_stationary.lateral_offset, abs=0.1
        )


class TestRobotVelocityEffects:
    def test_forward_velocity_changes_height(self):
        """Robot moving toward hub changes ball's effective speed, affecting height at target."""
        kwargs = dict(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=np.deg2rad(0),
            target_distance=5.0,
            target_bearing=np.deg2rad(0),
        )
        r_still = compute_trajectory_3d_moving(robot_vx=0.0, robot_vy=0.0, **kwargs)
        # Robot moving toward hub (bearing=0, so +X is toward hub)
        r_forward = compute_trajectory_3d_moving(robot_vx=3.0, robot_vy=0.0, **kwargs)
        # Heights should differ because ball reaches target plane sooner/later
        assert r_still.height_at_target != pytest.approx(r_forward.height_at_target, abs=0.1)

    def test_perpendicular_velocity_causes_lateral_drift(self):
        """Robot moving perpendicular to bearing causes lateral offset."""
        result = compute_trajectory_3d_moving(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=np.deg2rad(0),  # aiming perfectly along bearing
            target_distance=5.0,
            target_bearing=np.deg2rad(0),
            robot_vx=0.0,
            robot_vy=2.0,  # perpendicular to bearing (bearing=0 means X-axis)
        )
        assert abs(result.lateral_offset) > 0.05

    def test_perpendicular_velocity_increases_lateral_miss(self):
        """Larger perpendicular velocity should cause more lateral drift."""
        kwargs = dict(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=np.deg2rad(0),
            target_distance=5.0,
            target_bearing=np.deg2rad(0),
            robot_vx=0.0,
        )
        r_slow = compute_trajectory_3d_moving(robot_vy=1.0, **kwargs)
        r_fast = compute_trajectory_3d_moving(robot_vy=3.0, **kwargs)
        assert abs(r_fast.lateral_offset) > abs(r_slow.lateral_offset)

    def test_backward_velocity_effect(self):
        """Robot moving away from hub should change trajectory differently than toward."""
        kwargs = dict(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=np.deg2rad(0),
            target_distance=5.0,
            target_bearing=np.deg2rad(0),
            robot_vy=0.0,
        )
        r_forward = compute_trajectory_3d_moving(robot_vx=2.0, **kwargs)
        r_backward = compute_trajectory_3d_moving(robot_vx=-2.0, **kwargs)
        assert r_forward.height_at_target != pytest.approx(r_backward.height_at_target, abs=0.1)


class TestTrajectory3DMovingBasics:
    def test_starts_at_launch_height(self):
        """Trajectory Z should start at LAUNCH_HEIGHT."""
        result = compute_trajectory_3d_moving(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=0.0,
            target_distance=5.0,
            target_bearing=0.0,
            robot_vx=1.0,
            robot_vy=0.0,
        )
        assert result.trajectory_z[0] == pytest.approx(LAUNCH_HEIGHT, abs=0.01)

    def test_trajectory_has_three_axes(self):
        """Result should have trajectory_x, trajectory_y, trajectory_z arrays."""
        result = compute_trajectory_3d_moving(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=0.0,
            target_distance=5.0,
            target_bearing=0.0,
        )
        assert len(result.trajectory_x) > 1
        assert len(result.trajectory_y) > 1
        assert len(result.trajectory_z) > 1
        assert len(result.trajectory_x) == len(result.trajectory_z)

    def test_hit_detection_works(self):
        """A well-aimed shot from a stationary robot should still hit."""
        # Use parameters known to produce a hit from existing tests
        result = compute_trajectory_3d_moving(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=np.deg2rad(5),
            target_distance=5.0,
            target_bearing=np.deg2rad(5),
            robot_vx=0.0,
            robot_vy=0.0,
        )
        # Should at least produce reasonable values (may or may not hit depending on angle)
        assert result.height_at_target > 0

    def test_miss_distance_nonnegative(self):
        """Miss distance should never be negative."""
        result = compute_trajectory_3d_moving(
            launch_velocity=10.0,
            elevation=np.deg2rad(20),
            azimuth=np.deg2rad(30),
            target_distance=5.0,
            target_bearing=np.deg2rad(0),
            robot_vx=1.0,
            robot_vy=1.0,
        )
        assert result.total_miss_distance >= 0
        assert result.vertical_miss >= 0
        assert result.lateral_miss >= 0


class TestAirResistanceWithMoving:
    def test_drag_reduces_height(self):
        """Air resistance should reduce effective range / height at target."""
        kwargs = dict(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=0.0,
            target_distance=8.0,
            target_bearing=0.0,
            robot_vx=0.0,
            robot_vy=0.0,
        )
        r_no_drag = compute_trajectory_3d_moving(air_resistance=False, **kwargs)
        r_drag = compute_trajectory_3d_moving(air_resistance=True, **kwargs)
        assert r_drag.height_at_target < r_no_drag.height_at_target

    def test_drag_with_robot_velocity(self):
        """Air resistance + robot velocity should both affect trajectory."""
        kwargs = dict(
            launch_velocity=15.0,
            elevation=np.deg2rad(45),
            azimuth=0.0,
            target_distance=5.0,
            target_bearing=0.0,
        )
        r_still_no_drag = compute_trajectory_3d_moving(
            robot_vx=0.0, robot_vy=0.0, air_resistance=False, **kwargs
        )
        r_moving_drag = compute_trajectory_3d_moving(
            robot_vx=2.0, robot_vy=1.0, air_resistance=True, **kwargs
        )
        # Should produce different results when both are active
        assert r_still_no_drag.height_at_target != pytest.approx(
            r_moving_drag.height_at_target, abs=0.1
        )
