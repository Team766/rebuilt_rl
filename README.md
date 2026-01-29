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
- Ball launch height: 0.5m
- Field: 16.5m x 8.2m
- Alliance zone depth: 5m

## Installation

```bash
# Requires Python 3.10+ with CUDA support
pip install gymnasium stable-baselines3 numpy torch
```

## Training

```bash
# Train SAC with continuous actions (recommended)
python scripts/train.py --algorithm SAC --env-type continuous --timesteps 30000

# Options:
#   --algorithm: PPO, DQN, or SAC (SAC recommended for continuous)
#   --env-type: 2d, 3d, or continuous
#   --timesteps: Total training steps (default: 30000)
#   --eval-freq: Evaluation frequency (default: 1000)
#   --n-envs: Parallel environments (default: 4)
```

## Evaluation

```bash
python scripts/evaluate.py models/SAC_CONT_*/best/best_model.zip --env-type continuous --episodes 100
```

## Project Structure

```
frc_rl/
├── src/
│   ├── config.py              # Game constants and action encoding
│   ├── env/
│   │   ├── shooter_env.py     # Discrete action environments (2D, 3D)
│   │   └── shooter_env_continuous.py  # Continuous action environment
│   └── physics/
│       └── projectile.py      # 2D/3D trajectory simulation
├── scripts/
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
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

SAC with continuous actions achieves ~97% hit rate after 20k training steps:
- Close range (0-2m): 87%
- Medium range (2-4m): 100%
- Long range (4-6m): 100%

## Team

Team 766 - M-A Bears
