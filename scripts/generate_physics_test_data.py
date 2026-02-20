#!/usr/bin/env python3
"""Generate reference physics test data for validating the JavaScript port.

Runs comprehensive test cases through the Python physics engine and outputs
results to physics_test_data.json. The shot_simulator.html built-in test
runner loads this file to verify the JS physics matches.

Usage:
    python scripts/generate_physics_test_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    BALL_MASS,
    BALL_RADIUS,
    DRAG_COEFFICIENT,
    HUB_ENTRY_MIN,
    HUB_OPENING_HALF_WIDTH,
    HUB_OPENING_HEIGHT,
    LAUNCH_HEIGHT,
)
from src.physics.projectile import (
    check_hub_entry,
    compute_optimal_angle,
    compute_trajectory,
    compute_trajectory_3d_moving,
)


def generate_test_cases():
    """Generate all test cases with expected results from Python physics."""
    cases = []

    # --- Hub entry detection ---
    cases.append(
        {
            "name": "hub_entry_hit_above_rim",
            "function": "checkHubEntry",
            "params": {"height": HUB_ENTRY_MIN + 0.1, "vy": -5.0},
            "expected": {"hit": True, "miss_distance": 0.0},
        }
    )
    cases.append(
        {
            "name": "hub_entry_hit_edge",
            "function": "checkHubEntry",
            "params": {"height": HUB_ENTRY_MIN + 0.01, "vy": -3.0},
            "expected": {"hit": True, "miss_distance": 0.0},
        }
    )
    cases.append(
        {
            "name": "hub_entry_miss_too_low",
            "function": "checkHubEntry",
            "params": {"height": HUB_ENTRY_MIN - 0.5, "vy": -3.0},
            "expected": {"hit": False, "miss_distance": 0.5},
            "tolerance": {"miss_distance": 0.01},
        }
    )
    cases.append(
        {
            "name": "hub_entry_miss_ascending",
            "function": "checkHubEntry",
            "params": {"height": HUB_OPENING_HEIGHT, "vy": 2.0},
            "expected": {"hit": False},
        }
    )

    # --- Optimal angle ---
    for vel, dist, name_suffix in [
        (15.0, 5.0, "v15_d5"),
        (12.0, 4.0, "v12_d4"),
        (10.0, 3.0, "v10_d3"),
        (20.0, 8.0, "v20_d8"),
    ]:
        angle = compute_optimal_angle(vel, dist)
        expected = {}
        if angle is not None:
            expected["angle_deg"] = float(np.rad2deg(angle))
            expected["reachable"] = True
            # Verify it produces a hit via trajectory
            result = compute_trajectory(vel, angle, dist)
            expected["hit_with_optimal"] = result.hit
            expected["height_at_target"] = float(result.height_at_target)
        else:
            expected["reachable"] = False

        cases.append(
            {
                "name": f"optimal_angle_{name_suffix}",
                "function": "computeOptimalAngle",
                "params": {"velocity": vel, "target_distance": dist},
                "expected": expected,
                "tolerance": {"angle_deg": 0.5, "height_at_target": 0.05},
            }
        )

    # Impossible shot
    angle = compute_optimal_angle(3.0, 50.0)
    cases.append(
        {
            "name": "optimal_angle_impossible",
            "function": "computeOptimalAngle",
            "params": {"velocity": 3.0, "target_distance": 50.0},
            "expected": {"reachable": False},
        }
    )

    # --- 3D Moving trajectory: stationary shots ---
    trajectory_cases = [
        {
            "name": "stationary_hit_bearing30",
            "params": {
                "launch_velocity": 8.3,
                "elevation_deg": 70,
                "azimuth_deg": 30,
                "target_distance": 4.0,
                "target_bearing_deg": 30,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        {
            "name": "stationary_hit_bearing45",
            "params": {
                "launch_velocity": 8.3,
                "elevation_deg": 70,
                "azimuth_deg": 45,
                "target_distance": 4.0,
                "target_bearing_deg": 45,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        {
            "name": "stationary_miss_wrong_azimuth",
            "params": {
                "launch_velocity": 15.0,
                "elevation_deg": 50,
                "azimuth_deg": 45,
                "target_distance": 4.0,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        {
            "name": "stationary_miss_weak_shot",
            "params": {
                "launch_velocity": 5.0,
                "elevation_deg": 15,
                "azimuth_deg": 0,
                "target_distance": 5.0,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        # Wall collision cases
        {
            "name": "wall_collision_very_close",
            "params": {
                "launch_velocity": 5.66,
                "elevation_deg": 78.88,
                "azimuth_deg": 0,
                "target_distance": 0.5,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        {
            "name": "wall_collision_close",
            "params": {
                "launch_velocity": 5.66,
                "elevation_deg": 78.88,
                "azimuth_deg": 0,
                "target_distance": 0.7,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        {
            "name": "entry_from_above_hit",
            "params": {
                "launch_velocity": 5.66,
                "elevation_deg": 78.88,
                "azimuth_deg": 0,
                "target_distance": 0.92,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        # Moving robot cases
        {
            "name": "moving_lateral_drift",
            "params": {
                "launch_velocity": 15.0,
                "elevation_deg": 45,
                "azimuth_deg": 0,
                "target_distance": 5.0,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 2.0,
                "air_resistance": False,
            },
        },
        {
            "name": "moving_forward",
            "params": {
                "launch_velocity": 15.0,
                "elevation_deg": 45,
                "azimuth_deg": 0,
                "target_distance": 5.0,
                "target_bearing_deg": 0,
                "robot_vx": 3.0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        {
            "name": "moving_backward",
            "params": {
                "launch_velocity": 15.0,
                "elevation_deg": 45,
                "azimuth_deg": 0,
                "target_distance": 5.0,
                "target_bearing_deg": 0,
                "robot_vx": -2.0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        # Air resistance cases
        {
            "name": "air_resistance_on",
            "params": {
                "launch_velocity": 15.0,
                "elevation_deg": 45,
                "azimuth_deg": 0,
                "target_distance": 8.0,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": True,
            },
        },
        {
            "name": "air_resistance_off",
            "params": {
                "launch_velocity": 15.0,
                "elevation_deg": 45,
                "azimuth_deg": 0,
                "target_distance": 8.0,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        {
            "name": "air_resistance_moving",
            "params": {
                "launch_velocity": 15.0,
                "elevation_deg": 45,
                "azimuth_deg": 0,
                "target_distance": 5.0,
                "target_bearing_deg": 0,
                "robot_vx": 2.0,
                "robot_vy": 1.0,
                "air_resistance": True,
            },
        },
        # Physics sanity checks
        {
            "name": "sanity_slow_v10_d5",
            "params": {
                "launch_velocity": 10.0,
                "elevation_deg": 45,
                "azimuth_deg": 0,
                "target_distance": 5.0,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        {
            "name": "sanity_fast_v20_d5",
            "params": {
                "launch_velocity": 20.0,
                "elevation_deg": 45,
                "azimuth_deg": 0,
                "target_distance": 5.0,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        {
            "name": "sanity_low_angle_30",
            "params": {
                "launch_velocity": 15.0,
                "elevation_deg": 30,
                "azimuth_deg": 0,
                "target_distance": 5.0,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        {
            "name": "sanity_high_angle_60",
            "params": {
                "launch_velocity": 15.0,
                "elevation_deg": 60,
                "azimuth_deg": 0,
                "target_distance": 5.0,
                "target_bearing_deg": 0,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        # Various bearings
        {
            "name": "bearing_neg30_hit",
            "params": {
                "launch_velocity": 8.3,
                "elevation_deg": 70,
                "azimuth_deg": -30,
                "target_distance": 4.0,
                "target_bearing_deg": -30,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
        # Zero-velocity equivalence (match test_physics_moving.py)
        {
            "name": "zero_vel_equiv_v15_e45_d5",
            "params": {
                "launch_velocity": 15.0,
                "elevation_deg": 45,
                "azimuth_deg": 10,
                "target_distance": 5.0,
                "target_bearing_deg": 10,
                "robot_vx": 0,
                "robot_vy": 0,
                "air_resistance": False,
            },
        },
    ]

    for tc in trajectory_cases:
        p = tc["params"]
        result = compute_trajectory_3d_moving(
            launch_velocity=p["launch_velocity"],
            elevation=np.deg2rad(p["elevation_deg"]),
            azimuth=np.deg2rad(p["azimuth_deg"]),
            target_distance=p["target_distance"],
            target_bearing=np.deg2rad(p["target_bearing_deg"]),
            robot_vx=p["robot_vx"],
            robot_vy=p["robot_vy"],
            air_resistance=p["air_resistance"],
        )
        tc["function"] = "computeTrajectory3DMoving"
        tc["expected"] = {
            "hit": bool(result.hit),
            "height_at_target": float(result.height_at_target),
            "velocity_z_at_target": float(result.velocity_z_at_target),
            "lateral_offset": float(result.lateral_offset),
            "vertical_miss": float(result.vertical_miss),
            "lateral_miss": float(result.lateral_miss),
            "total_miss_distance": float(result.total_miss_distance),
            "center_distance": float(result.center_distance),
            "trajectory_length": len(result.trajectory_x),
            "start_z": float(result.trajectory_z[0]),
            "max_z": float(np.max(result.trajectory_z)),
        }
        tc["tolerance"] = {
            "height_at_target": 0.05,
            "velocity_z_at_target": 0.1,
            "lateral_offset": 0.05,
            "vertical_miss": 0.05,
            "lateral_miss": 0.05,
            "total_miss_distance": 0.1,
            "center_distance": 0.05,
            "start_z": 0.01,
            "max_z": 0.05,
        }
        cases.append(tc)

    # --- Comparison test pairs (for relational assertions) ---
    comparisons = [
        {
            "name": "air_resistance_reduces_height",
            "type": "less_than",
            "field": "height_at_target",
            "case_a": "air_resistance_on",
            "case_b": "air_resistance_off",
            "description": "Air resistance should reduce height at target",
        },
        {
            "name": "faster_shot_higher_at_target",
            "type": "greater_than",
            "field": "height_at_target",
            "case_a": "sanity_fast_v20_d5",
            "case_b": "sanity_slow_v10_d5",
            "description": "Faster shot should have higher height at same distance",
        },
        {
            "name": "higher_angle_higher_arc",
            "type": "greater_than",
            "field": "max_z",
            "case_a": "sanity_high_angle_60",
            "case_b": "sanity_low_angle_30",
            "description": "Higher angle should produce higher arc",
        },
    ]

    return {"test_cases": cases, "comparisons": comparisons}


def main():
    data = generate_test_cases()

    output_path = Path(__file__).parent / "physics_test_data.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    n_cases = len(data["test_cases"])
    n_comparisons = len(data["comparisons"])
    print(f"Generated {n_cases} test cases and {n_comparisons} comparisons")
    print(f"Written to {output_path}")

    # Print summary
    hits = sum(
        1
        for tc in data["test_cases"]
        if tc["function"] == "computeTrajectory3DMoving" and tc["expected"].get("hit")
    )
    trajectories = sum(
        1 for tc in data["test_cases"] if tc["function"] == "computeTrajectory3DMoving"
    )
    print(f"Trajectory tests: {trajectories} ({hits} hits, {trajectories - hits} misses)")


if __name__ == "__main__":
    main()
