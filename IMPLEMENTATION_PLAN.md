# FRC 2026 REBUILT Ball Shooter RL Implementation Plan

## Overview

Train a reinforcement learning model to output optimal launch velocity and angle for shooting FUEL (foam balls) into the HUB, given the robot's distance from the target.

## Game Specifications (from REBUILT 2026 Manual)

| Parameter | Value |
|-----------|-------|
| Field size | 16.5m x 8.2m |
| HUB opening height | 1.83m (72") |
| HUB opening width | 1.06m (41.7" hexagonal) |
| HUB distance from alliance wall | 4.03m |
| Alliance zone | 4.03m x 8.07m |
| FUEL diameter | 0.15m (5.91") |
| Robot launch height | 0.5m (fixed) |

## Environment Design

### State Space
- **Single continuous value**: Range to target (distance from robot to HUB center)
- Range bounds: [0.5m, ~12m] (min 50cm from hub, max is diagonal across field)

### Action Space (Discrete)
- **Velocity**: 10 discrete bins from 5 to 25 m/s (2 m/s increments)
- **Angle**: 15 discrete bins from 10° to 80° (5° increments)
- **Total actions**: 10 × 15 = 150 discrete actions

### Physics Model (2D Projectile Motion)
```
x(t) = v × cos(θ) × t
y(t) = h₀ + v × sin(θ) × t - 0.5 × g × t²

Where:
- v = launch velocity (m/s)
- θ = launch angle (radians)
- h₀ = 0.5m (launch height)
- g = 9.81 m/s²
```

### Scoring Logic (Basketball-style entry from above)
1. Compute full trajectory until ball hits ground or passes target
2. Check if ball passes through HUB opening **while descending** (vy < 0)
3. At the moment ball crosses x = target_distance:
   - Ball must be descending (negative vertical velocity)
   - Ball height must be within HUB opening [1.30m, 2.36m] (accounting for ball radius)
4. HUB opening center: 1.83m, half-width: 0.53m, ball radius: 0.075m

### Episode Structure
- **Single shot per episode**: Robot spawns, takes one shot, receives reward, episode ends
- Rationale: Each shot is independent; learning optimal (v, θ) for each distance

### Reward Function (Shaped)
```python
if ball_hits_target:
    # Distance from center of opening (1.83m)
    center_distance = abs(ball_height_at_target - 1.83)
    max_distance = 0.53  # half opening width
    reward = 1.0 + (1.0 - center_distance / max_distance)  # Range: [1.0, 2.0]
else:
    # Negative reward based on how far off the shot was
    miss_distance = compute_miss_distance()
    reward = -0.5 * (miss_distance / max_miss)  # Range: [-0.5, 0]
```

## Project Structure

```
frc_rl/
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── env/
│   │   ├── __init__.py
│   │   └── shooter_env.py      # Gym environment
│   ├── physics/
│   │   ├── __init__.py
│   │   └── projectile.py       # Projectile motion calculations
│   └── config.py               # Game constants and hyperparameters
├── scripts/
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation and testing
└── tests/
    ├── test_physics.py         # Unit tests for physics
    └── test_env.py             # Unit tests for environment
```

## Implementation Phases

### Phase 1: Core Physics Engine ✓
- [x] Implement 2D projectile motion calculations
- [x] Implement hit detection (does trajectory pass through HUB opening?)
- [x] Unit tests for physics calculations (19 tests passing)

### Phase 2: Gym Environment ✓
- [x] Create `ShooterEnv` class inheriting from `gymnasium.Env`
- [x] Implement state space (range to target)
- [x] Implement discrete action space (velocity × angle grid)
- [x] Implement `reset()`: random robot spawn in alliance zone
- [x] Implement `step()`: compute trajectory, check hit, return reward
- [x] Unit tests for environment (18 tests passing)

### Phase 3: Training Pipeline ✓
- [x] Set up stable-baselines3 with PPO algorithm
- [x] Configure hyperparameters (learning rate, batch size, etc.)
- [x] Implement training script with logging
- [x] Add checkpointing and model saving
- [x] Support for DQN as alternative algorithm

### Phase 4: Evaluation ✓
- [x] Create evaluation script
- [x] Compute success rate across distance ranges
- [x] Random baseline comparison

## Future Enhancements (Not in Initial Scope)
- [ ] Trajectory visualization / rendering
- [ ] Launch noise/uncertainty modeling
- [ ] 3D physics with azimuth angle
- [ ] Air resistance
- [ ] Moving targets / defense scenarios

## Environment Setup

Using the existing ml-frameworks CUDA 13.0 stack for DGX Spark:

```bash
# Activate the environment
cd /home/cpadwick/code/ml-frameworks/stacks/pytorch-cu130
source .venv/bin/activate

# Run from frc_rl directory
cd /home/cpadwick/code/frc_rl
```

**Verified Environment:**
- PyTorch 2.10.0+cu130
- CUDA 13.0 (NVIDIA GB10)
- Gymnasium 1.2.3
- Stable-Baselines3 2.7.1

## Dependencies

Already installed in the cu130 environment:
```
torch==2.10.0+cu130
gymnasium==1.2.3
stable-baselines3==2.7.1
numpy==1.26.4
matplotlib==3.10.8
pytest==7.4.4  # from dev dependencies
```

## Key Assumptions

1. **Auto-aim**: Robot always faces the HUB center (2D simplification)
2. **Perfect knowledge**: Robot knows exact range to target
3. **No obstacles**: Clear line of sight to HUB
4. **Instant launch**: No mechanical delays or spin-up time
5. **Point mass**: Ball treated as point (with radius for hit detection)

## Success Criteria

- Model achieves >90% hit rate across all valid shooting distances
- Model learns to prefer center shots (higher rewards)
- Training converges within reasonable time (~100k timesteps)

## Resolved Design Decisions

1. **HUB geometry**: Ball must enter from above (descending) - basketball-style ✓
2. **Episode structure**: Single shot per episode ✓
3. **Action discretization**: 150 actions (10 vel × 15 angle) - start here, tune if needed ✓
