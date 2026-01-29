# Gymnasium Environment Design

This document describes the design of the FRC ball shooter Gymnasium environments.

## What is Gymnasium?

[Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI Gym) is a standard API for reinforcement learning environments. It provides a consistent interface that RL algorithms can use to interact with any environment.

The key concept is the **agent-environment loop**:

```
┌─────────┐     action      ┌─────────────┐
│  Agent  │ ───────────────►│ Environment │
│  (RL)   │                 │  (Shooter)  │
│         │◄─────────────── │             │
└─────────┘  observation,   └─────────────┘
             reward, done
```

1. The agent observes the current state
2. The agent chooses an action
3. The environment executes the action and returns:
   - New observation (next state)
   - Reward (how good was that action?)
   - Done flag (is the episode over?)

## Environment Overview

We provide three environment variants:

| Environment | Action Space | Observation | Use Case |
|-------------|--------------|-------------|----------|
| `ShooterEnv` | Discrete (150) | Range only | Simple 2D, auto-aim |
| `ShooterEnv3D` | Discrete (27,000) | Range + Bearing | Turret aiming, discrete |
| `ShooterEnvContinuous` | Continuous (3D) | Range + Bearing | Turret aiming, precise |

## Observation Space

### What the Agent Sees

The agent receives information about its position relative to the target HUB.

**2D Environment (`ShooterEnv`)**:
```python
observation = [distance]  # Just the range to target (meters)
```

**3D Environments (`ShooterEnv3D`, `ShooterEnvContinuous`)**:
```python
observation = [distance, bearing]
# distance: Range to target (0.5m to ~12m)
# bearing: Angle to target (-π to +π radians)
```

### Why This Design?

We give the agent minimal information - just what it needs to make the shot:
- **Distance**: Determines required velocity and elevation
- **Bearing**: Determines turret azimuth angle

We don't give the agent:
- Absolute position (x, y) - not needed for the shot
- Previous shots - each shot is independent
- Target location - implicit in range/bearing

This keeps the state space small, making learning faster.

## Action Space

### Discrete Actions (2D and 3D)

For discrete environments, actions are indices into a grid:

**2D Environment**:
```python
# 10 velocity bins × 15 angle bins = 150 actions
action = velocity_idx * 15 + angle_idx

# Velocity: 5-25 m/s in 10 steps
# Elevation: 10-80° in 15 steps
```

**3D Environment**:
```python
# 10 velocity × 15 elevation × 180 azimuth = 27,000 actions
action = vel_idx * (15 * 180) + elev_idx * 180 + azim_idx

# Azimuth: -90° to +90° in 1° steps
```

### Continuous Actions

For `ShooterEnvContinuous`, actions are a 3D vector:

```python
action = [velocity_norm, elevation_norm, azimuth_norm]
# Each value in range [-1, 1], scaled to actual ranges:
#   velocity:  [-1, 1] → [5, 25] m/s
#   elevation: [-1, 1] → [10, 80] degrees
#   azimuth:   [-1, 1] → [-90, +90] degrees
```

### Why Continuous is Better

Discrete actions have limitations:
- 27,000 actions is a huge space to explore
- 1° resolution may not be precise enough
- Learning takes longer

Continuous actions allow:
- Infinite precision
- Smaller effective search space
- Faster learning with appropriate algorithms (SAC)

## Reward Function

The reward function shapes what the agent learns.

### Hit Reward
```python
if hit:
    # Base reward for hitting + bonus for center shots
    center_bonus = 1.0 - (distance_from_center / max_center_distance)
    reward = 1.0 + 1.0 * center_bonus  # Range: [1.0, 2.0]
```

### Miss Penalty
```python
if miss:
    # Penalty scaled by how far off the shot was
    normalized_miss = min(miss_distance / 5.0, 1.0)
    reward = -0.5 * normalized_miss  # Range: [-0.5, 0]
```

### Why This Design?

1. **Positive for hits, negative for misses**: Clear signal for success/failure
2. **Center bonus**: Encourages precise shots, not just "good enough"
3. **Scaled miss penalty**: Tells the agent "you were close" vs "way off"
4. **Bounded range**: Prevents extreme values that destabilize training

## Episode Structure

### Single-Shot Episodes (Original)
```python
def step(self, action):
    # Execute shot
    result = compute_trajectory(...)
    reward = self._compute_reward(result)
    terminated = True  # Episode ends after one shot
    return observation, reward, terminated, truncated, info
```

### Multi-Shot Episodes (Current)
```python
def step(self, action):
    # Execute shot
    result = compute_trajectory(...)
    reward = self._compute_reward(result)

    self.current_shot += 1
    terminated = self.current_shot >= 50  # 50 shots per episode

    if not terminated:
        self._generate_new_position()  # Move to new spot

    return observation, reward, terminated, truncated, info
```

### Why 50 Shots Per Episode?

- Matches a real FRC match (~50 shots)
- Provides better learning signal (more data per episode)
- Allows the agent to experience many positions per episode
- More stable training metrics

## Physics Integration

The environment uses a physics engine (`src/physics/projectile.py`) to simulate ball trajectories:

```python
def step(self, action):
    velocity, elevation, azimuth = self._scale_action(action)

    result = compute_trajectory_3d(
        velocity=velocity,
        elevation=elevation,
        azimuth=azimuth,
        target_distance=self.distance_to_hub,
        target_bearing=self.bearing_to_hub,
    )

    # result contains:
    # - hit: bool (did it go in?)
    # - height_at_target: where the ball crossed the target plane
    # - lateral_offset: horizontal miss distance
    # - center_distance: how far from center (for bonus)
```

### Hit Detection

A shot is a "hit" if:
1. Ball crosses the target plane at correct height (within HUB opening)
2. Ball crosses at correct lateral position (within HUB width)
3. Ball is **descending** (basketball-style entry)

The descending requirement prevents unrealistic flat shots.

## Reset Behavior

```python
def reset(self, seed=None):
    # Random position in alliance zone
    self.robot_x = random.uniform(0, ALLIANCE_ZONE_DEPTH)  # 0-5m
    self.robot_y = random.uniform(0, ALLIANCE_ZONE_WIDTH)  # 0-8.2m

    # Ensure minimum distance from target
    while distance_to_hub < 0.5:
        regenerate_position()

    # Compute range and bearing
    self.distance_to_hub = sqrt(dx² + dy²)
    self.bearing_to_hub = atan2(dy, dx)

    return [self.distance_to_hub, self.bearing_to_hub], info
```

## Configuration

Key constants are in `src/config.py`:

```python
# Field dimensions (meters)
ALLIANCE_ZONE_DEPTH = 5.0
ALLIANCE_ZONE_WIDTH = 8.23
HUB_DISTANCE_FROM_WALL = 4.03

# Target (HUB)
HUB_OPENING_HEIGHT = 1.83
HUB_OPENING_HALF_WIDTH = 0.53

# Action ranges
VELOCITY_MIN, VELOCITY_MAX = 5.0, 25.0
ANGLE_MIN_DEG, ANGLE_MAX_DEG = 10.0, 80.0
AZIMUTH_MIN_DEG, AZIMUTH_MAX_DEG = -90.0, 90.0

# Rewards
REWARD_HIT_BASE = 1.0
REWARD_HIT_CENTER = 1.0
REWARD_MISS_SCALE = -0.5
```

## Using the Environment

### Basic Usage
```python
from src.env.shooter_env_continuous import ShooterEnvContinuous

env = ShooterEnvContinuous()
obs, info = env.reset(seed=42)

for _ in range(50):  # One episode
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Shot {info['shot']}: {'HIT' if info['hit'] else 'MISS'}, reward={reward:.2f}")

    if terminated:
        break

env.close()
```

### With Stable-Baselines3
```python
from stable_baselines3 import SAC
from src.env.shooter_env_continuous import ShooterEnvContinuous

env = ShooterEnvContinuous()
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30000)

# Evaluate
obs, info = env.reset()
action, _ = model.predict(obs, deterministic=True)
obs, reward, terminated, truncated, info = env.step(action)
print(f"Hit: {info['hit']}")
```

## Gymnasium Registration

The environments are registered with Gymnasium:

```python
# In shooter_env.py
gym.register(id="FRCShooter-v0", entry_point="src.env.shooter_env:ShooterEnv")
gym.register(id="FRCShooter3D-v0", entry_point="src.env.shooter_env:ShooterEnv3D")

# In shooter_env_continuous.py
gym.register(id="FRCShooterContinuous-v0", entry_point="src.env.shooter_env_continuous:ShooterEnvContinuous")
```

You can also create environments via:
```python
import gymnasium as gym
env = gym.make("FRCShooterContinuous-v0")
```
