#!/usr/bin/env python3
"""Evaluation script for trained FRC ball shooter RL agent."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import DQN, PPO, SAC

from src.config import action_to_velocity_angle, action_to_velocity_elevation_azimuth
from src.env.shooter_env import ShooterEnv, ShooterEnv3D
from src.env.shooter_env_continuous import ShooterEnvContinuous


def evaluate(
    model_path: str,
    n_episodes: int = 100,
    seed: int = 42,
    verbose: bool = True,
    analyze_by_distance: bool = True,
    env_type: str = "2d",
    air_resistance: bool = False,
    move_and_shoot: bool = False,
):
    """Evaluate a trained model.

    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        seed: Random seed
        verbose: Print detailed results
        analyze_by_distance: Analyze performance by distance bins
        env_type: Environment type ("2d", "3d", or "continuous")

    Returns:
        Dictionary with evaluation metrics
    """
    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        # Try adding .zip extension
        if model_path.with_suffix(".zip").exists():
            model_path = model_path.with_suffix(".zip")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    # Detect algorithm from model
    algorithm = None
    model = None
    for algo_cls, algo_name in [(SAC, "SAC"), (PPO, "PPO"), (DQN, "DQN")]:
        try:
            model = algo_cls.load(str(model_path))
            algorithm = algo_name
            break
        except Exception:
            continue

    if model is None:
        raise ValueError(f"Could not load model from {model_path}")

    if verbose:
        print(f"Loaded {algorithm} model from: {model_path}")

    # Create environment based on type
    if env_type == "3d":
        env = ShooterEnv3D(air_resistance=air_resistance)
    elif env_type == "continuous":
        env = ShooterEnvContinuous(
            air_resistance=air_resistance,
            move_and_shoot=move_and_shoot,
        )
    else:
        env = ShooterEnv(air_resistance=air_resistance)

    # Track results
    hits = 0
    total_reward = 0
    results = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        distance = info["distance"]

        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Execute action
        if env_type != "continuous":
            action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)

        # Record result based on env type
        if env_type == "continuous":
            results.append({
                "episode": episode,
                "distance": distance,
                "velocity": info.get("velocity", 0),
                "elevation_deg": info.get("elevation_deg", 0),
                "azimuth_deg": info.get("azimuth_deg", 0),
                "hit": info["hit"],
                "reward": reward,
                "height_at_target": info["height_at_target"],
                "center_distance": info["center_distance"],
            })
        elif env_type == "3d":
            velocity, elevation, azimuth = action_to_velocity_elevation_azimuth(action)
            results.append({
                "episode": episode,
                "distance": distance,
                "velocity": velocity,
                "elevation_deg": np.rad2deg(elevation),
                "azimuth_deg": np.rad2deg(azimuth),
                "hit": info["hit"],
                "reward": reward,
                "height_at_target": info["height_at_target"],
                "center_distance": info["center_distance"],
            })
        else:
            velocity, angle = action_to_velocity_angle(action)
            results.append({
                "episode": episode,
                "distance": distance,
                "velocity": velocity,
                "angle_deg": np.rad2deg(angle),
                "hit": info["hit"],
                "reward": reward,
                "height_at_target": info["height_at_target"],
                "center_distance": info["center_distance"],
            })

        if info["hit"]:
            hits += 1
        total_reward += reward

    env.close()

    # Compute metrics
    hit_rate = hits / n_episodes
    avg_reward = total_reward / n_episodes

    if verbose:
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({n_episodes} episodes)")
        print(f"{'='*50}")
        print(f"Hit Rate: {hit_rate:.1%} ({hits}/{n_episodes})")
        print(f"Average Reward: {avg_reward:.3f}")

    # Analyze by distance bins
    if analyze_by_distance:
        distance_bins = [0, 2, 4, 6, 8, 10, 15]
        bin_results = {f"{distance_bins[i]}-{distance_bins[i+1]}m": {"hits": 0, "total": 0}
                      for i in range(len(distance_bins) - 1)}

        for r in results:
            d = r["distance"]
            for i in range(len(distance_bins) - 1):
                if distance_bins[i] <= d < distance_bins[i + 1]:
                    bin_key = f"{distance_bins[i]}-{distance_bins[i+1]}m"
                    bin_results[bin_key]["total"] += 1
                    if r["hit"]:
                        bin_results[bin_key]["hits"] += 1
                    break

        if verbose:
            print("\nHit Rate by Distance:")
            print(f"{'-'*30}")
            for bin_key, data in bin_results.items():
                if data["total"] > 0:
                    rate = data["hits"] / data["total"]
                    print(f"  {bin_key}: {rate:.1%} ({data['hits']}/{data['total']})")

    # Sample some specific results
    if verbose and len(results) > 0:
        print("\nSample Shots:")
        print(f"{'-'*70}")
        if env_type in ["3d", "continuous"]:
            print(f"{'Dist':>6} {'Vel':>6} {'Elev':>6} {'Azim':>6} {'Height':>7} {'Hit':>5}")
            print(f"{'-'*70}")
            for r in results[:10]:
                hit_str = "Yes" if r["hit"] else "No"
                print(f"{r['distance']:6.2f} {r['velocity']:6.1f} {r['elevation_deg']:6.1f} "
                      f"{r['azimuth_deg']:6.1f} {r['height_at_target']:7.2f} {hit_str:>5}")
        else:
            print(f"{'Dist':>6} {'Vel':>6} {'Angle':>6} {'Height':>7} {'Hit':>5}")
            print(f"{'-'*70}")
            for r in results[:10]:
                hit_str = "Yes" if r["hit"] else "No"
                print(f"{r['distance']:6.2f} {r['velocity']:6.1f} {r['angle_deg']:6.1f} "
                      f"{r['height_at_target']:7.2f} {hit_str:>5}")

    return {
        "hit_rate": hit_rate,
        "avg_reward": avg_reward,
        "total_hits": hits,
        "total_episodes": n_episodes,
        "results": results,
    }


def compare_to_random(n_episodes: int = 100, seed: int = 42):
    """Compare random policy to see baseline performance."""
    env = ShooterEnv()
    hits = 0
    total_reward = 0

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if info["hit"]:
            hits += 1
        total_reward += reward

    env.close()

    hit_rate = hits / n_episodes
    avg_reward = total_reward / n_episodes

    print(f"\nRandom Policy Baseline ({n_episodes} episodes)")
    print(f"{'='*40}")
    print(f"Hit Rate: {hit_rate:.1%} ({hits}/{n_episodes})")
    print(f"Average Reward: {avg_reward:.3f}")

    return hit_rate, avg_reward


def main():
    parser = argparse.ArgumentParser(description="Evaluate FRC shooter RL agent")
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to saved model (or 'random' for baseline)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Also show random baseline",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="2d",
        choices=["2d", "3d", "continuous"],
        help="Environment type",
    )
    parser.add_argument(
        "--air-resistance",
        action="store_true",
        help="Enable air resistance (should match training config)",
    )
    parser.add_argument(
        "--move-and-shoot",
        action="store_true",
        help="Enable move-and-shoot mode (should match training config)",
    )

    args = parser.parse_args()

    if args.model_path is None or args.model_path.lower() == "random":
        compare_to_random(args.episodes, args.seed)
    else:
        evaluate(
            args.model_path,
            n_episodes=args.episodes,
            seed=args.seed,
            env_type=args.env_type,
            air_resistance=args.air_resistance,
            move_and_shoot=args.move_and_shoot,
        )

        if args.random_baseline:
            compare_to_random(args.episodes, args.seed)


if __name__ == "__main__":
    main()
