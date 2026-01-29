# FRC 2026 REBUILT Ball Shooter RL

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

### Using Poetry (recommended)

```bash
# Install with Poetry (includes PyTorch with CUDA 13.0 for Blackwell/GB10)
poetry install

# With dev dependencies (pytest, black, ruff)
poetry install --with dev
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

# Options:
#   --algorithm: PPO, DQN, or SAC (SAC recommended for continuous)
#   --env-type: 2d, 3d, or continuous
#   --timesteps: Total training steps (default: 30000)
#   --eval-freq: Evaluation frequency (default: 1000)
#   --n-envs: Parallel environments (default: 4)
#   --air-resistance: Enable air resistance in physics simulation
#   --learning-rate: Learning rate (default: 3e-4, try 1e-4 for stability)
```

## Evaluation

```bash
python scripts/evaluate.py models/SAC_CONT_*/best/best_model.zip --env-type continuous --episodes 100
```

## Project Structure

```
frc_rl/
├── pyproject.toml             # Package configuration (Poetry)
├── src/
│   ├── config.py              # Game constants and action encoding
│   ├── env/
│   │   ├── shooter_env.py     # Discrete action environments (2D, 3D)
│   │   └── shooter_env_continuous.py  # Continuous action environment
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
- `range`: Distance to target (0.5m - ~12m)
- `bearing`: Angle to target (-π to π radians)

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

SAC with continuous actions achieves **98.4% hit rate** with air resistance enabled:

| Configuration | Hit Rate |
|---------------|----------|
| Without air resistance | 99.7% |
| With air resistance (realistic) | 98.4% |

Best results achieved with:
- Learning rate: 1e-4
- Batch size: 4096
- ~32k training steps (best checkpoint)

## Team

Team 766 - M-A Bears
