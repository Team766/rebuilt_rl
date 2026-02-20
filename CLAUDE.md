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

# Train move-and-shoot
python scripts/train.py --algorithm SAC --env-type continuous --move-and-shoot --speed-min 0.1 --speed-max 0.5 --timesteps 150000

# Train move-and-shoot with air resistance
python scripts/train.py --algorithm SAC --env-type continuous --move-and-shoot --air-resistance --timesteps 150000

# Resume training from checkpoint
python scripts/train.py --algorithm SAC --env-type continuous --resume models/SAC_CONT_*/checkpoints/shooter_50000_steps.zip

# Evaluate
python scripts/evaluate.py models/SAC_CONT_*/best/best_model.zip --env-type continuous --episodes 100 --analyze-by-distance

# Monitor training
tensorboard --logdir logs/

# Visualize trained model (generates self-contained HTML with Three.js)
python scripts/visualize.py models/SAC_CONT_*/best/best_model.zip --env-type continuous --episodes 3
python scripts/visualize.py models/SAC_CONT_MAS_*/best/best_model.zip --env-type continuous --move-and-shoot --air-resistance --speed-min 3.0 --speed-max 5.0 --episodes 3
```

## Code Style

- Black formatter, 100-char line length, Python 3.10+ target
- Ruff linter with E, F, W, I, N rule sets
- Python 3.10-3.12 compatibility required (CI tests all three)

## Architecture

The codebase has four layers with strict dependency direction: Physics -> Environments -> Scripts.

**`src/config.py`** - Single source of truth for all game constants (field dimensions, HUB specs, ball properties), action space definitions, reward parameters, and bin-to-value conversion functions. All action space encoding/decoding lives here.

**`src/physics/projectile.py`** - Trajectory simulation engine with no RL dependencies. Uses Euler integration (dt=0.001s) for 2D and 3D trajectories. Supports optional quadratic air drag. Returns `TrajectoryResult`/`TrajectoryResult3D`/`TrajectoryResult3DMoving` dataclass-style objects. Hit detection models the hub as a cylinder with a top scoring surface and solid walls: balls must enter the cylinder from above (z >= HUB_OPENING_HEIGHT) and descend through the top to score. Balls entering from the side (z < hub height) trigger wall collision and miss. `compute_trajectory_3d_moving()` does full 3D Euler integration with robot velocity inheritance for move-and-shoot mode.

**`src/env/`** - Three Gymnasium environment variants sharing the same physics:
- `ShooterEnv` (2D discrete): obs=[distance], 150 actions, single-shot episodes
- `ShooterEnv3D` (3D discrete): obs=[distance, bearing], 27,000 actions, single-shot episodes
- `ShooterEnvContinuous` (3D continuous): obs=[distance, bearing], 3D Box[-1,1] actions, 50-shot episodes. Supports `move_and_shoot` mode: obs=[distance, bearing, vx, vy], robot follows paths and ball inherits robot velocity.

All environments use minimal observations (distance + bearing, not absolute position) so the agent generalizes across field positions. Reward shaping: +1.0 to +2.0 for hits (with center accuracy bonus), -0.5 to 0 for misses (scaled by distance).

**`src/paths/`** - Path generation for move-and-shoot mode. `RobotPath` ABC with `BouncingLinePath` (elastically bounces off zone walls) and `StraightLinePath` (stationary only). `generate_path()` creates random valid paths that keep the robot moving for the full episode.

**`src/sac_logging.py`** - `LoggingSAC` subclass of SB3's SAC. Logs actor/critic gradient norms per training step to TensorBoard for diagnostics. Applies gradient clipping (max norm 1.0) on both actor and critic to prevent Q-value divergence.

**`scripts/train.py`** - Training orchestration using Stable-Baselines3 (PPO/DQN/SAC). Uses SubprocVecEnv for parallel environments, auto-detects GPU, saves best model via EvalCallback and periodic checkpoints. Models save to `models/`, logs to `logs/`.

**`scripts/evaluate.py`** - Loads trained models and runs evaluation episodes with hit rate metrics. Supports `--analyze-by-distance` for performance breakdown.

**`scripts/visualize.py`** + **`scripts/visualize_template.html`** - Browser-based 3D visualization using Three.js (CDN). `visualize.py` loads a trained model, runs episodes, captures trajectories in field coordinates, and embeds the JSON data into the HTML template. The resulting self-contained HTML file shows animated ball trajectories, robot position, field, and HUB with playback controls. Supports both stationary and move-and-shoot modes. Use `--speed-min`/`--speed-max` to match the speed range the model was trained on.

## Key Design Decisions

- SAC with continuous actions is the recommended algorithm/env combo (98.4% hit rate with air resistance)
- The continuous env runs 50 shots per episode (match-length), while discrete envs are single-shot
- Air resistance is optional (`--air-resistance` flag) and uses realistic quadratic drag
- Best models are saved at `models/*/best/best_model.zip` (not final_model.zip, which may be degraded)
- CI replaces GPU PyTorch with CPU-only version since GitHub Actions has no GPU
- Move-and-shoot (`--move-and-shoot`) requires `--env-type continuous`. Uses 4D observations [distance, bearing, vx, vy]. Robot velocity is added to ball launch velocity (realistic physics). The robot bounces elastically off zone walls (triangle-wave folding), staying in motion for the entire episode. Use `--speed-min` and `--speed-max` to set the robot speed range.
- SAC hyperparameters: `batch_size=256`, `gradient_steps=1`, `target_entropy=-6.0`. Larger batch sizes and more gradient steps cause critic Q-value divergence, especially in move-and-shoot mode. The low target entropy allows the policy to stay deterministic after convergence (default of -3 drives entropy back up, degrading a converged policy). Gradient clipping (max norm 1.0) is applied via `LoggingSAC`.
- Imports use `from src.xxx` (Poetry `packages = [{ include = "src" }]`), e.g. `from src.config import ...`, `from src.physics.projectile import compute_trajectory_3d`.
