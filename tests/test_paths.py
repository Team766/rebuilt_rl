"""Unit tests for path generation module."""

import numpy as np
import pytest

from src.config import (
    ALLIANCE_ZONE_WIDTH,
    HUB_DISTANCE_FROM_WALL,
    MIN_DISTANCE_FROM_HUB,
)
from src.paths.path_generator import (
    StraightLinePath,
    generate_straight_line_path,
)

HUB_X = HUB_DISTANCE_FROM_WALL
HUB_Y = ALLIANCE_ZONE_WIDTH / 2


class TestStraightLinePath:
    def test_initial_position(self):
        """Path starts at specified position."""
        path = StraightLinePath(1.0, 2.0, heading=0.0, speed=1.0, path_duration=5.0)
        state = path.state_at(0.0)
        assert state.x == pytest.approx(1.0)
        assert state.y == pytest.approx(2.0)

    def test_position_at_time(self):
        """Position advances linearly with time along heading."""
        path = StraightLinePath(0.0, 0.0, heading=0.0, speed=2.0, path_duration=5.0)
        state = path.state_at(1.0)
        assert state.x == pytest.approx(2.0, abs=0.01)
        assert state.y == pytest.approx(0.0, abs=0.01)

    def test_velocity_constant(self):
        """Velocity is constant along the entire path."""
        path = StraightLinePath(0.0, 0.0, heading=np.pi / 4, speed=3.0, path_duration=5.0)
        s0 = path.state_at(0.0)
        s1 = path.state_at(2.5)
        s2 = path.state_at(5.0)
        assert s0.vx == pytest.approx(s1.vx)
        assert s0.vy == pytest.approx(s1.vy)
        assert s1.vx == pytest.approx(s2.vx)

    def test_time_clamped_to_bounds(self):
        """Time is clamped to [0, duration]."""
        path = StraightLinePath(0.0, 0.0, heading=0.0, speed=1.0, path_duration=2.0)
        s_before = path.state_at(-1.0)
        s_after = path.state_at(10.0)
        assert s_before.x == pytest.approx(0.0)
        assert s_after.x == pytest.approx(2.0)

    def test_diagonal_heading(self):
        """45-degree heading moves equally in X and Y."""
        path = StraightLinePath(
            0.0, 0.0, heading=np.pi / 4, speed=np.sqrt(2), path_duration=5.0
        )
        state = path.state_at(1.0)
        assert state.x == pytest.approx(1.0, abs=0.01)
        assert state.y == pytest.approx(1.0, abs=0.01)

    def test_duration(self):
        path = StraightLinePath(0.0, 0.0, heading=0.0, speed=1.0, path_duration=3.5)
        assert path.duration() == pytest.approx(3.5)

    def test_speed_stored(self):
        path = StraightLinePath(0.0, 0.0, heading=0.0, speed=2.5, path_duration=3.0)
        state = path.state_at(0.0)
        assert state.speed == pytest.approx(2.5)

    def test_negative_heading(self):
        """Negative heading moves in -Y direction."""
        path = StraightLinePath(1.0, 4.0, heading=-np.pi / 2, speed=1.0, path_duration=5.0)
        state = path.state_at(2.0)
        assert state.x == pytest.approx(1.0, abs=0.01)
        assert state.y == pytest.approx(2.0, abs=0.01)


class TestStraightLinePathValidity:
    def test_is_valid_inside_zone(self):
        """Position inside zone and far from hub is valid."""
        path = StraightLinePath(1.0, 1.0, heading=0.0, speed=0.0, path_duration=5.0)
        assert path.is_valid_at(0.0)

    def test_is_invalid_outside_zone(self):
        """Position outside alliance zone is invalid."""
        path = StraightLinePath(-1.0, 1.0, heading=0.0, speed=0.0, path_duration=5.0)
        assert not path.is_valid_at(0.0)

    def test_is_invalid_near_hub(self):
        """Position too close to hub is invalid."""
        path = StraightLinePath(
            HUB_X - 0.1, HUB_Y, heading=0.0, speed=0.0, path_duration=5.0
        )
        assert not path.is_valid_at(0.0)


class TestGenerateStraightLinePath:
    def test_returns_valid_path(self):
        """Generated path stays within bounds for its duration."""
        rng = np.random.default_rng(42)
        path = generate_straight_line_path(rng, speed_min=1.0, speed_max=2.0)
        for t in np.arange(0, path.duration(), 0.2):
            assert path.is_valid_at(t), f"Path invalid at t={t}"

    def test_speed_range_respected(self):
        """Generated path speed is within requested range."""
        rng = np.random.default_rng(42)
        path = generate_straight_line_path(rng, speed_min=1.0, speed_max=2.0)
        state = path.state_at(0.0)
        assert 1.0 <= state.speed <= 2.0

    def test_zero_speed_produces_stationary(self):
        """Speed range [0, 0] produces a stationary path."""
        rng = np.random.default_rng(42)
        path = generate_straight_line_path(rng, speed_min=0.0, speed_max=0.0)
        s0 = path.state_at(0.0)
        s1 = path.state_at(1.0)
        assert s0.x == pytest.approx(s1.x)
        assert s0.y == pytest.approx(s1.y)
        assert s0.vx == pytest.approx(0.0)
        assert s0.vy == pytest.approx(0.0)

    def test_multiple_seeds_produce_different_paths(self):
        """Different seeds produce different paths."""
        p1 = generate_straight_line_path(np.random.default_rng(1), 1.0, 3.0)
        p2 = generate_straight_line_path(np.random.default_rng(2), 1.0, 3.0)
        s1 = p1.state_at(0.0)
        s2 = p2.state_at(0.0)
        # At least one of position or heading should differ
        assert not (
            s1.x == pytest.approx(s2.x, abs=0.01)
            and s1.y == pytest.approx(s2.y, abs=0.01)
            and s1.vx == pytest.approx(s2.vx, abs=0.01)
        )

    def test_minimum_duration_met(self):
        """Path meets the minimum duration requirement at low speed."""
        rng = np.random.default_rng(42)
        # Use low speed where 3-second paths fit easily in the zone
        path = generate_straight_line_path(rng, speed_min=0.5, speed_max=1.0, min_duration=3.0)
        assert path.duration() >= 3.0
        assert path._speed > 0, "Path should not be a stationary fallback"

    def test_high_speed_path_is_not_stationary(self):
        """High-speed paths should move (not fall back to stationary)."""
        rng = np.random.default_rng(42)
        path = generate_straight_line_path(rng, speed_min=3.0, speed_max=5.0)
        assert path._speed > 0, "Path should not be a stationary fallback"
        assert path.duration() > 0

    def test_hub_avoidance(self):
        """Path never passes within MIN_DISTANCE_FROM_HUB of the hub."""
        rng = np.random.default_rng(42)
        for seed in range(10):
            rng = np.random.default_rng(seed)
            path = generate_straight_line_path(rng, speed_min=1.0, speed_max=3.0)
            for t in np.arange(0, path.duration(), 0.1):
                state = path.state_at(t)
                dx = HUB_X - state.x
                dy = HUB_Y - state.y
                dist = np.sqrt(dx * dx + dy * dy)
                assert dist >= MIN_DISTANCE_FROM_HUB - 0.15, (
                    f"Seed {seed}, t={t}: dist to hub = {dist:.3f}"
                )
