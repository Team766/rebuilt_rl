# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning for FRC 2026 REBUILT ball shooting. Trains an agent to output optimal launch velocity, elevation angle, and azimuth angle for shooting balls into the HUB target from any position in the alliance zone. Built by Team 766 (M-A Bears).

## Common Commands

```bash
# Install (standard, CUDA 12.6)
poetry install --with dev

# Install on DGX Spark (CUDA 13.0) - override PyTorch after install
# poetry install --with dev
# poetry run pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall

# Run all tests
poetry run pytest tests/ -v --tb=short

# Run a single test file
poetry run pytest tests/test_physics.py -v
poetry run pytest tests/test_env.py -v

# Run a specific test
poetry run pytest tests/test_physics.py::test_hub_entry_valid -v

# Coverage (matches CI)
poetry run pytest tests/ --cov=src --cov-report=xml

# Format
poetry run black src/ scripts/ tests/

# Lint
poetry run ruff check src/ scripts/ tests/
poetry run ruff check --fix src/ scripts/ tests/

# Train (recommended configuration)
python scripts/train.py --algorithm SAC --env-type continuous --timesteps 50000

# Train move-and-shoot with curriculum learning
python scripts/train.py --algorithm SAC --env-type continuous --move-and-shoot --timesteps 150000

# Train move-and-shoot with air resistance
python scripts/train.py --algorithm SAC --env-type continuous --move-and-shoot --air-resistance --timesteps 150000

# Resume training from checkpoint
python scripts/train.py --algorithm SAC --env-type continuous --resume models/SAC_CONT_*/checkpoints/shooter_50000_steps.zip

# Evaluate
python scripts/evaluate.py models/SAC_CONT_*/best/best_model.zip --env-type continuous --episodes 100 --analyze-by-distance

# Monitor training
tensorboard --logdir logs/
```

## Code Style

- Black formatter, 100-char line length, Python 3.10+ target
- Ruff linter with E, F, W, I, N rule sets
- Python 3.10-3.12 compatibility required (CI tests all three)

## Architecture

The codebase has four layers with strict dependency direction: Physics -> Environments -> Scripts.

**`src/config.py`** - Single source of truth for all game constants (field dimensions, HUB specs, ball properties), action space definitions, reward parameters, and bin-to-value conversion functions. All action space encoding/decoding lives here.

**`src/physics/projectile.py`** - Trajectory simulation engine with no RL dependencies. Uses Euler integration (dt=0.001s) for 2D and 3D trajectories. Supports optional quadratic air drag. Returns `TrajectoryResult`/`TrajectoryResult3D`/`TrajectoryResult3DMoving` dataclass-style objects. Hub entry validation requires the ball to be descending (vy < 0) and within the opening bounds. `compute_trajectory_3d_moving()` does full 3D Euler integration with robot velocity inheritance for move-and-shoot mode.

**`src/env/`** - Three Gymnasium environment variants sharing the same physics:
- `ShooterEnv` (2D discrete): obs=[distance], 150 actions, single-shot episodes
- `ShooterEnv3D` (3D discrete): obs=[distance, bearing], 27,000 actions, single-shot episodes
- `ShooterEnvContinuous` (3D continuous): obs=[distance, bearing], 3D Box[-1,1] actions, 50-shot episodes. Supports `move_and_shoot` mode: obs=[distance, bearing, vx, vy], robot follows paths and ball inherits robot velocity.

All environments use minimal observations (distance + bearing, not absolute position) so the agent generalizes across field positions. Reward shaping: +1.0 to +2.0 for hits (with center accuracy bonus), -0.5 to 0 for misses (scaled by distance).

**`src/paths/`** - Path generation for move-and-shoot mode. `RobotPath` ABC with `StraightLinePath` implementation. Paths stay within alliance zone and maintain distance from hub. `generate_straight_line_path()` creates random valid paths.

**`src/sac_logging.py`** - `LoggingSAC` subclass of SB3's SAC. Logs actor/critic gradient norms per training step to TensorBoard for diagnostics. Applies gradient clipping (max norm 1.0) on both actor and critic to prevent Q-value divergence.

**`src/callbacks/`** - `CurriculumCallback` for automatic difficulty progression during move-and-shoot training. Monitors eval hit rate and advances through levels (crawl -> slow -> medium -> fast) by updating robot speed via `env_method("set_curriculum_level", ...)`. Clears the replay buffer on level advancement to prevent stale transitions from destabilizing Q-value estimates.

**`scripts/train.py`** - Training orchestration using Stable-Baselines3 (PPO/DQN/SAC). Uses SubprocVecEnv for parallel environments, auto-detects GPU, saves best model via EvalCallback and periodic checkpoints. Models save to `models/`, logs to `logs/`.

**`scripts/evaluate.py`** - Loads trained models and runs evaluation episodes with hit rate metrics. Supports `--analyze-by-distance` for performance breakdown.

## Key Design Decisions

- SAC with continuous actions is the recommended algorithm/env combo (98.4% hit rate with air resistance)
- The continuous env runs 50 shots per episode (match-length), while discrete envs are single-shot
- Air resistance is optional (`--air-resistance` flag) and uses realistic quadratic drag
- Best models are saved at `models/*/best/best_model.zip` (not final_model.zip, which may be degraded)
- CI replaces GPU PyTorch with CPU-only version since GitHub Actions has no GPU
- Move-and-shoot (`--move-and-shoot`) requires `--env-type continuous`. Uses 4D observations [distance, bearing, vx, vy] even at curriculum level 0 (crawl) since SB3 can't change obs shape mid-training. Robot velocity is added to ball launch velocity (realistic physics). Path durations are limited by alliance zone geometry; the robot stops when the path ends.
- SAC hyperparameters: `batch_size=256`, `gradient_steps=1`, `target_entropy=-6.0`. Larger batch sizes and more gradient steps cause critic Q-value divergence, especially in move-and-shoot mode. The low target entropy allows the policy to stay deterministic after convergence (default of -3 drives entropy back up, degrading a converged policy). Gradient clipping (max norm 1.0) is applied via `LoggingSAC`.
- Imports use `from src.xxx` (Poetry `packages = [{ include = "src" }]`), e.g. `from src.config import ...`, `from src.physics.projectile import compute_trajectory_3d`.
