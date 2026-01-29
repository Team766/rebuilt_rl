"""Unit tests for physics module."""

import numpy as np
import pytest

from src.physics.projectile import (
    compute_trajectory,
    check_hub_entry,
    compute_optimal_angle,
)
from src.config import (
    LAUNCH_HEIGHT,
    HUB_OPENING_HEIGHT,
    HUB_ENTRY_MIN,
    HUB_ENTRY_MAX,
    GRAVITY,
)


class TestCheckHubEntry:
    """Tests for hub entry detection."""

    def test_hit_center(self):
        """Ball at center height, descending."""
        hit, miss = check_hub_entry(HUB_OPENING_HEIGHT, -5.0)
        assert hit is True
        assert miss == 0.0

    def test_hit_low_edge(self):
        """Ball at lower edge, descending."""
        hit, miss = check_hub_entry(HUB_ENTRY_MIN + 0.01, -3.0)
        assert hit is True
        assert miss == 0.0

    def test_hit_high_edge(self):
        """Ball at upper edge, descending."""
        hit, miss = check_hub_entry(HUB_ENTRY_MAX - 0.01, -3.0)
        assert hit is True
        assert miss == 0.0

    def test_miss_too_low(self):
        """Ball below opening."""
        hit, miss = check_hub_entry(HUB_ENTRY_MIN - 0.5, -3.0)
        assert hit is False
        assert miss == pytest.approx(0.5, abs=0.01)

    def test_miss_too_high(self):
        """Ball above opening."""
        hit, miss = check_hub_entry(HUB_ENTRY_MAX + 0.3, -3.0)
        assert hit is False
        assert miss == pytest.approx(0.3, abs=0.01)

    def test_miss_ascending(self):
        """Ball at correct height but ascending (wrong direction)."""
        hit, miss = check_hub_entry(HUB_OPENING_HEIGHT, 2.0)
        assert hit is False
        assert miss > 0  # Should have penalty


class TestComputeTrajectory:
    """Tests for trajectory computation."""

    def test_basic_trajectory(self):
        """Basic trajectory computation."""
        result = compute_trajectory(
            velocity=15.0,
            angle=np.deg2rad(45),
            target_distance=5.0,
        )
        assert result.trajectory_x[0] == 0.0
        assert result.trajectory_y[0] == pytest.approx(LAUNCH_HEIGHT, abs=0.01)
        assert len(result.trajectory_x) == len(result.trajectory_y)
        assert len(result.trajectory_x) > 10  # Should have multiple points

    def test_trajectory_starts_at_launch_height(self):
        """Trajectory should start at launch height."""
        result = compute_trajectory(
            velocity=10.0,
            angle=np.deg2rad(30),
            target_distance=3.0,
        )
        assert result.trajectory_y[0] == pytest.approx(LAUNCH_HEIGHT, abs=0.001)

    def test_short_distance_hit(self):
        """Should be able to hit at short distance with right params."""
        # At 2m distance, need enough velocity to reach target
        result = compute_trajectory(
            velocity=10.0,
            angle=np.deg2rad(60),
            target_distance=2.0,
        )
        # Ball should reach target and have some height
        assert result.height_at_target > 0.5  # Should be above ground
        assert result.height_at_target < 5.0  # Shouldn't be absurdly high

    def test_trajectory_descends(self):
        """Trajectory should eventually descend."""
        result = compute_trajectory(
            velocity=15.0,
            angle=np.deg2rad(45),
            target_distance=10.0,
        )
        # Find max height index
        max_idx = np.argmax(result.trajectory_y)
        # Should have points after max (descending portion)
        assert max_idx < len(result.trajectory_y) - 1

    def test_miss_short_shot(self):
        """Very weak shot should miss (too low)."""
        result = compute_trajectory(
            velocity=5.0,
            angle=np.deg2rad(20),
            target_distance=10.0,
        )
        # Weak shot at long distance should fall short
        assert result.hit is False
        assert result.miss_distance > 0

    def test_hit_detection_requires_descending(self):
        """Ball must be descending to count as hit."""
        # Very high angle, short distance - ball might be ascending at target
        result = compute_trajectory(
            velocity=20.0,
            angle=np.deg2rad(80),
            target_distance=2.0,
        )
        # If ball is ascending at target, should not count as hit
        if result.velocity_y_at_target > 0:
            assert result.hit is False


class TestComputeOptimalAngle:
    """Tests for optimal angle computation."""

    def test_optimal_angle_exists(self):
        """Should find optimal angle for reasonable params."""
        angle = compute_optimal_angle(velocity=15.0, target_distance=5.0)
        assert angle is not None
        assert 0 < angle < np.pi / 2  # Between 0 and 90 degrees

    def test_optimal_angle_hits_target(self):
        """Optimal angle should result in a hit."""
        velocity = 12.0
        distance = 4.0
        angle = compute_optimal_angle(velocity, distance)

        if angle is not None:
            result = compute_trajectory(velocity, angle, distance)
            # Should hit or be very close
            assert result.hit or result.miss_distance < 0.2

    def test_impossible_shot(self):
        """Should return None for impossible shots."""
        # Very low velocity, very long distance
        angle = compute_optimal_angle(velocity=3.0, target_distance=50.0)
        assert angle is None

    def test_optimal_angle_reasonable_range(self):
        """Optimal angles should be valid (0-90 degrees)."""
        for dist in [3.0, 5.0, 8.0]:
            angle = compute_optimal_angle(velocity=15.0, target_distance=dist)
            if angle is not None:
                angle_deg = np.rad2deg(angle)
                # Just check it's a valid angle (0-90 degrees)
                # Short distances may require high arcs to hit descending
                assert 0 < angle_deg < 90


class TestIntegration:
    """Integration tests combining components."""

    def test_range_of_distances(self):
        """Test hits possible across typical shooting distances."""
        hits = 0
        total = 0

        for distance in [2.0, 4.0, 6.0, 8.0]:
            for velocity in [10.0, 15.0, 20.0]:
                angle = compute_optimal_angle(velocity, distance)
                if angle is not None:
                    result = compute_trajectory(velocity, angle, distance)
                    if result.hit:
                        hits += 1
                    total += 1

        # Should be able to hit most combinations
        assert total > 0
        hit_rate = hits / total
        assert hit_rate > 0.3  # At least 30% of optimal shots should hit

    def test_trajectory_physics_sanity(self):
        """Verify basic physics relationships."""
        # Higher velocity should reach further - use shorter distance
        # that both can reach
        result_slow = compute_trajectory(10.0, np.deg2rad(45), 5.0)
        result_fast = compute_trajectory(20.0, np.deg2rad(45), 5.0)

        # Fast shot should have higher height at same distance
        # (both should reach 5m)
        assert result_fast.height_at_target > result_slow.height_at_target

    def test_angle_affects_arc(self):
        """Higher angle should create higher arc."""
        result_low = compute_trajectory(15.0, np.deg2rad(30), 5.0)
        result_high = compute_trajectory(15.0, np.deg2rad(60), 5.0)

        max_height_low = np.max(result_low.trajectory_y)
        max_height_high = np.max(result_high.trajectory_y)

        assert max_height_high > max_height_low
