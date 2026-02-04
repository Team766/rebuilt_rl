#!/usr/bin/env python3
"""Training script for FRC ball shooter RL agent."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from stable_baselines3 import DQN, PPO, SAC

from src.sac_logging import LoggingSAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.env.shooter_env import ShooterEnv, ShooterEnv3D
from src.env.shooter_env_continuous import ShooterEnvContinuous


def make_env(
    seed: int = 0,
    env_type: str = "2d",
    air_resistance: bool = False,
    move_and_shoot: bool = False,
    shot_interval: float = 0.5,
):
    """Create a single environment instance."""
    def _init():
        if env_type == "3d":
            env = ShooterEnv3D(air_resistance=air_resistance)
        elif env_type == "continuous":
            env = ShooterEnvContinuous(
                air_resistance=air_resistance,
                move_and_shoot=move_and_shoot,
                shot_interval=shot_interval,
            )
        else:
            env = ShooterEnv(air_resistance=air_resistance)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(
    algorithm: str = "PPO",
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    seed: int = 42,
    save_dir: str = "models",
    log_dir: str = "logs",
    eval_freq: int = 5000,
    n_eval_episodes: int = 50,
    checkpoint_freq: int = 5000,
    learning_rate: float = 3e-4,
    verbose: int = 1,
    env_type: str = "2d",
    air_resistance: bool = False,
    move_and_shoot: bool = False,
    shot_interval: float = 0.5,
    resume: str | None = None,
):
    """Train the RL agent.

    Args:
        algorithm: RL algorithm ("PPO", "DQN", or "SAC")
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        seed: Random seed
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        eval_freq: Evaluation frequency (steps)
        n_eval_episodes: Episodes per evaluation
        checkpoint_freq: Checkpoint save frequency
        learning_rate: Learning rate
        verbose: Verbosity level
        env_type: "2d" for original env, "3d" for turret aiming, "continuous" for SAC
        air_resistance: Whether to enable air resistance in physics
        move_and_shoot: Whether to enable move-and-shoot training with curriculum
        shot_interval: Seconds between shots in move-and-shoot mode
        resume: Path to a saved model zip to resume training from
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_suffix = {"2d": "", "3d": "_3D", "continuous": "_CONT"}[env_type]
    if move_and_shoot:
        env_suffix += "_MAS"
    run_name = f"{algorithm}{env_suffix}_{timestamp}"
    save_path = Path(save_dir) / run_name
    log_path = Path(log_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    print(f"Training {algorithm} for {total_timesteps:,} timesteps")
    print(f"Environment: {env_type.upper()}")
    print(f"Air resistance: {'ON' if air_resistance else 'OFF'}")
    print(f"Move and shoot: {'ON' if move_and_shoot else 'OFF'}")
    if move_and_shoot:
        print(f"Shot interval: {shot_interval}s")
    print(f"Save path: {save_path}")
    print(f"Log path: {log_path}")

    # Create vectorized training environment
    env_kwargs = dict(
        env_type=env_type,
        air_resistance=air_resistance,
        move_and_shoot=move_and_shoot,
        shot_interval=shot_interval,
    )
    if n_envs > 1:
        env = SubprocVecEnv([make_env(seed + i, **env_kwargs) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(seed, **env_kwargs)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(seed + 1000, **env_kwargs)])

    # Create model - check for available GPU memory
    device = "cpu"
    if torch.cuda.is_available():
        try:
            # Try to allocate a small tensor to check if GPU is usable
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            torch.cuda.empty_cache()

            # Check available memory
            free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # GB
            if free_memory > 1.0:  # Need at least 1GB free
                device = "cuda"
            else:
                print(f"Warning: Only {free_memory:.1f}GB GPU memory available, using CPU")
        except Exception as e:
            print(f"Warning: GPU error ({e}), falling back to CPU")

    print(f"Using device: {device}")

    # Resume from checkpoint or create new model
    algo_cls = {"PPO": PPO, "SAC": LoggingSAC, "DQN": DQN}[algorithm.upper()]
    if resume:
        print(f"Resuming from: {resume}")
        model = algo_cls.load(
            resume,
            env=env,
            device=device,
            tensorboard_log=str(log_path),
        )
        # Load replay buffer if it exists (SAC/DQN)
        replay_path = resume.replace(".zip", "").replace(
            "shooter_", "shooter_replay_buffer_"
        ) + ".pkl"
        if Path(replay_path).exists() and algorithm.upper() in ["SAC", "DQN"]:
            print(f"Loading replay buffer: {replay_path}")
            model.load_replay_buffer(replay_path)
    else:
        # Policy network architecture - large networks for GPU utilization
        policy_kwargs = dict(
            net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])
        )
        sac_policy_kwargs = dict(
            net_arch=[512, 512, 256]
        )

        if algorithm.upper() == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(log_path),
                verbose=verbose,
                seed=seed,
                device=device,
            )
        elif algorithm.upper() == "SAC":
            # target_entropy=-6 (2x default of -dim(A)=-3) because the optimal
            # policy is nearly deterministic: given distance+bearing there is one
            # correct launch configuration.  The default target drives entropy
            # back up after convergence, degrading the learned policy.
            model = LoggingSAC(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                buffer_size=500_000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                target_entropy=-6.0,
                policy_kwargs=sac_policy_kwargs,
                tensorboard_log=str(log_path),
                verbose=verbose,
                seed=seed,
                device=device,
            )
        elif algorithm.upper() == "DQN":
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                buffer_size=100_000,
                learning_starts=1000,
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_final_eps=0.05,
                tensorboard_log=str(log_path),
                verbose=verbose,
                seed=seed,
                device=device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    # Create callbacks
    # Curriculum callback for move-and-shoot
    if move_and_shoot:
        from src.callbacks.curriculum import CurriculumCallback

        curriculum_callback = CurriculumCallback(
            shots_per_episode=50,
            eval_window=3,
            verbose=verbose,
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_after_eval=curriculum_callback,
            best_model_save_path=str(save_path / "best"),
            log_path=str(log_path),
            eval_freq=eval_freq // n_envs,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=verbose,
        )
    else:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(save_path / "best"),
            log_path=str(log_path),
            eval_freq=eval_freq // n_envs,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=verbose,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(save_path / "checkpoints"),
        name_prefix="shooter",
        save_replay_buffer=algorithm.upper() in ["DQN", "SAC"],
        save_vecnormalize=False,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = save_path / "final_model"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}")

    # Cleanup
    env.close()
    eval_env.close()

    return model, str(save_path)


def main():
    parser = argparse.ArgumentParser(description="Train FRC shooter RL agent")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "DQN", "SAC"],
        help="RL algorithm to use (SAC recommended for continuous)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=30_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for tensorboard logs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1000,
        help="Evaluation frequency (in timesteps)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="2d",
        choices=["2d", "3d", "continuous"],
        help="Environment type: 2d, 3d (discrete turret), or continuous (continuous actions)",
    )
    parser.add_argument(
        "--air-resistance",
        action="store_true",
        help="Enable air resistance in physics simulation",
    )
    parser.add_argument(
        "--move-and-shoot",
        action="store_true",
        help="Enable move-and-shoot training with curriculum learning (continuous env only)",
    )
    parser.add_argument(
        "--shot-interval",
        type=float,
        default=0.5,
        help="Seconds between shots in move-and-shoot mode (default: 0.5)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to saved model checkpoint to resume training from",
    )

    args = parser.parse_args()

    if args.move_and_shoot and args.env_type != "continuous":
        parser.error("--move-and-shoot requires --env-type continuous")

    train(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        learning_rate=args.learning_rate,
        eval_freq=args.eval_freq,
        verbose=args.verbose,
        env_type=args.env_type,
        air_resistance=args.air_resistance,
        move_and_shoot=args.move_and_shoot,
        shot_interval=args.shot_interval,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
