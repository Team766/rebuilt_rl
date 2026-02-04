#!/usr/bin/env python3
"""Generate browser-based 3D visualization of trained FRC shooter RL agent."""

import argparse
import json
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import DQN, PPO, SAC

from src.config import (
    ALLIANCE_ZONE_DEPTH,
    ALLIANCE_ZONE_WIDTH,
    FIELD_LENGTH,
    FIELD_WIDTH,
    HUB_DISTANCE_FROM_WALL,
    HUB_ENTRY_MAX,
    HUB_ENTRY_MIN,
    HUB_OPENING_HEIGHT,
    HUB_OPENING_WIDTH,
    LAUNCH_HEIGHT,
)
from src.env.shooter_env_continuous import ShooterEnvContinuous
from src.sac_logging import LoggingSAC


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays and types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super().default(obj)


def downsample_trajectory(x, y, z, target_points=50):
    """Downsample trajectory to approximately target_points, keeping first and last."""
    n = len(x)
    if n <= target_points:
        return x, y, z

    # Keep first and last, evenly space the rest
    indices = [0]  # First point
    step = (n - 1) / (target_points - 1)
    for i in range(1, target_points - 1):
        indices.append(int(i * step))
    indices.append(n - 1)  # Last point

    return x[indices], y[indices], z[indices]


def load_model(model_path):
    """Load a trained model, trying multiple algorithm types."""
    model_path = Path(model_path)

    # Try loading with different algorithms
    for algo_class, algo_name in [
        (LoggingSAC, "LoggingSAC"),
        (SAC, "SAC"),
        (PPO, "PPO"),
        (DQN, "DQN"),
    ]:
        try:
            print(f"Attempting to load model with {algo_name}...")
            model = algo_class.load(str(model_path))
            print(f"Successfully loaded model with {algo_name}")
            return model, algo_name
        except Exception:
            continue

    raise ValueError(
        f"Could not load model from {model_path}. "
        f"Tried: LoggingSAC, SAC, PPO, DQN. "
        f"Make sure the model file exists and is compatible."
    )


def collect_episode_data(env, model, episode_id):
    """Collect data for a single episode."""
    obs, info = env.reset()

    # Get path info if in move-and-shoot mode
    path_info = None
    if env.move_and_shoot and env.current_path is not None:
        duration = env.current_path.duration()
        n_points = max(2, int(duration / 0.1))
        times = np.linspace(0, duration, n_points)
        path_xs = []
        path_ys = []
        for t in times:
            st = env.current_path.state_at(t)
            path_xs.append(float(st.x))
            path_ys.append(float(st.y))
        path_info = {"x": path_xs, "y": path_ys}

    shots = []
    total_reward = 0.0
    hits = 0

    terminated = False
    shot_id = 0

    while not terminated:
        # Save robot state BEFORE step
        robot_x = float(env.robot_x)
        robot_y = float(env.robot_y)
        robot_vx = float(env.robot_vx)
        robot_vy = float(env.robot_vy)
        distance = float(env.distance_to_hub)
        bearing = float(env.bearing_to_hub)

        # Predict action
        action, _ = model.predict(obs, deterministic=True)

        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        if info["hit"]:
            hits += 1

        # Get trajectory from environment
        traj = env.last_trajectory

        # Compute field coordinates from trajectory
        if hasattr(traj, "trajectory_z"):
            # TrajectoryResult3DMoving - already in field coordinates
            field_x = robot_x + traj.trajectory_x
            field_y = robot_y + traj.trajectory_y
            field_z = traj.trajectory_z
        else:
            # TrajectoryResult3D - need to project using azimuth
            azimuth_rad = np.deg2rad(info["azimuth_deg"])

            # Project trajectory_x (range) and trajectory_y (height) into 3D
            field_x = robot_x + traj.trajectory_x * np.cos(azimuth_rad)
            field_y = robot_y + traj.trajectory_x * np.sin(azimuth_rad)
            field_z = traj.trajectory_y

        # Downsample trajectory
        field_x, field_y, field_z = downsample_trajectory(field_x, field_y, field_z)

        shot_data = {
            "shot_id": shot_id,
            "robot_x": robot_x,
            "robot_y": robot_y,
            "robot_vx": robot_vx,
            "robot_vy": robot_vy,
            "distance": distance,
            "bearing_deg": float(np.rad2deg(bearing)),
            "velocity": float(info["velocity"]),
            "elevation_deg": float(info["elevation_deg"]),
            "azimuth_deg": float(info["azimuth_deg"]),
            "hit": bool(info["hit"]),
            "reward": float(reward),
            "height_at_target": float(info["height_at_target"]),
            "center_distance": float(info["center_distance"]),
            "trajectory": {
                "x": field_x.tolist() if isinstance(field_x, np.ndarray) else field_x,
                "y": field_y.tolist() if isinstance(field_y, np.ndarray) else field_y,
                "z": field_z.tolist() if isinstance(field_z, np.ndarray) else field_z,
            },
        }

        shots.append(shot_data)
        shot_id += 1

    hit_rate = hits / len(shots) if shots else 0.0

    episode_data = {
        "episode_id": episode_id,
        "hit_rate": hit_rate,
        "total_reward": total_reward,
        "path": path_info,
        "shots": shots,
    }

    return episode_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D visualization of trained FRC shooter RL agent"
    )
    parser.add_argument("model_path", type=str, help="Path to trained model (.zip file)")
    parser.add_argument(
        "--env-type",
        type=str,
        default="continuous",
        choices=["continuous"],
        help="Environment type (only continuous supported)",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--air-resistance", action="store_true", help="Enable air resistance")
    parser.add_argument("--move-and-shoot", action="store_true", help="Enable move-and-shoot mode")
    parser.add_argument(
        "--speed-min",
        type=float,
        default=0.0,
        help="Min robot speed in m/s (default: 0.0)",
    )
    parser.add_argument(
        "--speed-max",
        type=float,
        default=5.0,
        help="Max robot speed in m/s (default: 5.0)",
    )
    parser.add_argument(
        "--output", type=str, default="visualization.html", help="Output HTML file path"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open in browser"
    )

    args = parser.parse_args()

    # Load model
    model, algo_name = load_model(args.model_path)

    # Create environment (no wrappers)
    env = ShooterEnvContinuous(
        air_resistance=args.air_resistance,
        move_and_shoot=args.move_and_shoot,
        shots_per_episode=50,
        speed_min=args.speed_min,
        speed_max=args.speed_max if args.move_and_shoot else 0.0,
    )
    env.reset(seed=args.seed)

    # Collect episode data
    episodes = []
    total_hits = 0
    total_shots = 0

    for ep in range(args.episodes):
        print(f"Collecting episode {ep + 1}/{args.episodes}...")
        episode_data = collect_episode_data(env, model, ep)
        episodes.append(episode_data)

        total_hits += sum(1 for shot in episode_data["shots"] if shot["hit"])
        total_shots += len(episode_data["shots"])

        print(
            f"  Episode {ep + 1}: {len(episode_data['shots'])} shots, "
            f"{episode_data['hit_rate']:.1%} hit rate, "
            f"reward: {episode_data['total_reward']:.2f}"
        )

    overall_hit_rate = total_hits / total_shots if total_shots > 0 else 0.0

    # Prepare JSON data
    data = {
        "metadata": {
            "model_path": str(args.model_path),
            "algorithm": algo_name,
            "move_and_shoot": args.move_and_shoot,
            "air_resistance": args.air_resistance,
            "total_episodes": args.episodes,
            "overall_hit_rate": overall_hit_rate,
        },
        "field": {
            "length": float(FIELD_LENGTH),
            "width": float(FIELD_WIDTH),
            "alliance_zone_depth": float(ALLIANCE_ZONE_DEPTH),
            "alliance_zone_width": float(ALLIANCE_ZONE_WIDTH),
            "hub_x": float(HUB_DISTANCE_FROM_WALL),
            "hub_y": float(ALLIANCE_ZONE_WIDTH / 2),
            "hub_opening_height": float(HUB_OPENING_HEIGHT),
            "hub_opening_diameter": float(HUB_OPENING_WIDTH),
            "hub_entry_min": float(HUB_ENTRY_MIN),
            "hub_entry_max": float(HUB_ENTRY_MAX),
            "launch_height": float(LAUNCH_HEIGHT),
        },
        "episodes": episodes,
    }

    # Load template
    template_path = Path(__file__).parent / "visualize_template.html"
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template file not found: {template_path}. "
            f"Make sure visualize_template.html exists in the scripts directory."
        )

    template = template_path.read_text()

    # Replace placeholder with data
    html = template.replace("/*DATA_PLACEHOLDER*/null", json.dumps(data, cls=NumpyEncoder))

    # Write output
    output_path = Path(args.output)
    output_path.write_text(html)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Visualization generated: {output_path.absolute()}")
    print(f"Total episodes: {args.episodes}")
    print(f"Total shots: {total_shots}")
    print(f"Overall hit rate: {overall_hit_rate:.1%}")
    print(f"Model: {args.model_path}")
    print(f"Algorithm: {algo_name}")
    print(f"Move-and-shoot: {args.move_and_shoot}")
    print(f"Air resistance: {args.air_resistance}")
    print("=" * 60)

    # Open in browser
    if not args.no_browser:
        try:
            webbrowser.open(f"file://{output_path.absolute()}")
            print("\nOpening visualization in browser...")
        except Exception as e:
            print(f"\nCould not auto-open browser: {e}")
            print(f"Please open {output_path.absolute()} manually.")


if __name__ == "__main__":
    main()
