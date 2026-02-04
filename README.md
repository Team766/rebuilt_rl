# FRC 2026 REBUILT Ball Shooter RL

[![Tests](https://github.com/Team766/rebuilt_rl/actions/workflows/tests.yml/badge.svg)](https://github.com/Team766/rebuilt_rl/actions/workflows/tests.yml)

Reinforcement learning model for FRC 2026 REBUILT game ball shooting. Trains an agent to output optimal launch velocity, elevation angle, and azimuth (turret) angle given the robot's position relative to the target HUB.

## Overview

The agent learns to shoot balls into the HUB target from any position in the alliance zone. Given the range and bearing to the target, it outputs:
- **Velocity**: Launch speed (5-25 m/s)
- **Elevation**: Vertical angle (10-80 degrees)
- **Azimuth**: Horizontal turret angle (-90 to +90 degrees)

## Game Specifications (REBUILT 2026)

- HUB opening: 1.06m diameter hexagonal
- HUB height: 1.83m
- Ball (FUEL): 15.0cm diameter, 0.215kg (AndyMark am-5801)
- Ball launch height: 0.5m
- Field: 16.5m x 8.2m
- Alliance zone depth: 5m

## Documentation

- [Training Guide](docs/training.md) - How to train models, CLI options, troubleshooting
- [Environment Design](docs/gym.md) - Gymnasium environment architecture and usage
- [RL Algorithms](docs/rl.md) - Beginner-friendly guide to reinforcement learning and PPO/DQN/SAC

## Installation

### Standard Install (most systems, CUDA 12.6)

```bash
poetry install

# With dev dependencies (pytest, black, ruff)
poetry install --with dev
```

### DGX Spark / Blackwell GPUs (CUDA 13.0)

```bash
# Install base dependencies
poetry install

# Override PyTorch with cu130 version
poetry run pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
```

### Using pip

```bash
# Requires Python 3.10+ with CUDA support
pip install torch  # Install PyTorch for your CUDA version first
pip install -e .
```

## Training

```bash
# Train SAC with continuous actions (recommended)
python scripts/train.py --algorithm SAC --env-type continuous --timesteps 50000

# With air resistance for realistic physics
python scripts/train.py --algorithm SAC --env-type continuous --timesteps 50000 --air-resistance

# Move-and-shoot (trains robot to shoot while moving)
python scripts/train.py --algorithm SAC --env-type continuous --move-and-shoot --speed-min 0.1 --speed-max 0.5 --timesteps 150000

# Move-and-shoot with air resistance
python scripts/train.py --algorithm SAC --env-type continuous --move-and-shoot --air-resistance --speed-min 0.1 --speed-max 0.5 --timesteps 150000

# Resume from a checkpoint
python scripts/train.py --algorithm SAC --env-type continuous --resume models/SAC_CONT_*/checkpoints/shooter_50000_steps.zip

# Options:
#   --algorithm: PPO, DQN, or SAC (SAC recommended for continuous)
#   --env-type: 2d, 3d, or continuous
#   --timesteps: Total training steps (default: 30000)
#   --eval-freq: Evaluation frequency (default: 1000)
#   --n-envs: Parallel environments (default: 4)
#   --air-resistance: Enable air resistance in physics simulation
#   --move-and-shoot: Enable move-and-shoot (continuous only)
#   --speed-min: Min robot speed in m/s (default: 0.0)
#   --speed-max: Max robot speed in m/s (default: 0.0)
#   --shot-interval: Seconds between shots in move-and-shoot mode (default: 0.5)
#   --learning-rate: Learning rate (default: 3e-4)
#   --resume: Path to saved model checkpoint to resume training from
```

## Evaluation

```bash
python scripts/evaluate.py models/SAC_CONT_*/best/best_model.zip --env-type continuous --episodes 100
```

## Project Structure

```
rebuilt_rl/
├── pyproject.toml             # Package configuration (Poetry)
├── src/
│   ├── config.py              # Game constants and action encoding
│   ├── sac_logging.py         # LoggingSAC: SAC with gradient norm logging and clipping
│   ├── env/
│   │   ├── shooter_env.py     # Discrete action environments (2D, 3D)
│   │   └── shooter_env_continuous.py  # Continuous action environment (supports move-and-shoot)
│   ├── paths/                 # Path generation for move-and-shoot mode
│   ├── callbacks/
│   └── physics/
│       └── projectile.py      # 2D/3D trajectory simulation with air resistance
├── scripts/
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── docs/                      # Documentation
└── tests/                     # Unit tests
```

## Environment Details

### Observation Space

**Stationary mode**: `[distance, bearing]`
- `distance`: Range to target (0.5m - ~12m)
- `bearing`: Angle to target (-π to π radians)

**Move-and-shoot mode**: `[distance, bearing, vx, vy]`
- Adds robot velocity components so the agent can compensate for movement

### Action Space (Continuous)
- 3D Box [-1, 1] scaled to actual ranges
- `[0]`: Velocity (5-25 m/s)
- `[1]`: Elevation (10-80 degrees)
- `[2]`: Azimuth (-90 to +90 degrees)

### Reward
- **Hit**: +1.0 to +2.0 (bonus for center shots)
- **Miss**: -0.5 to 0 (scaled by miss distance)

### Episode
- 50 shots per episode (simulates a match)
- Robot moves to new random position after each shot

## Results

### Stationary Shooting

SAC with continuous actions achieves **98.4% hit rate** with air resistance enabled:

| Configuration | Hit Rate |
|---------------|----------|
| Without air resistance | 99.7% |
| With air resistance (realistic) | 98.4% |

### SAC Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 3e-4 | |
| Batch size | 256 | Larger values cause critic divergence |
| Gradient steps | 1 | Per environment step |
| Buffer size | 500,000 | |
| Target entropy | -6.0 | 2x default; optimal policy is nearly deterministic |
| Gradient clipping | 1.0 | Max norm on actor and critic |
| Network | [512, 512, 256] | Shared architecture for actor and critic |

## Team

Team 766 - M-A Bears
