# Reinforcement Learning Guide

This guide explains reinforcement learning (RL) concepts and the algorithms used in this project. It's written for beginners with no prior RL experience.

## What is Reinforcement Learning?

Reinforcement Learning is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. Unlike traditional programming where you tell the computer exactly what to do, in RL the agent learns from experience.

### The Key Idea

Imagine teaching a dog a new trick:
1. The dog tries something (action)
2. You give it a treat if it did well (reward)
3. The dog learns to repeat actions that got treats

RL works the same way:
1. The agent observes the current situation (state)
2. The agent takes an action
3. The environment gives a reward (positive or negative)
4. The agent updates its strategy to get more rewards

### Our Ball Shooter Example

```
State:   "I'm 3.5 meters from the target at a 45° angle"
Action:  "Shoot at 12 m/s, 50° elevation, 45° azimuth"
Reward:  +1.9 (hit near center!) or -0.3 (missed by 0.5m)
```

The agent tries thousands of shots, gradually learning which velocity/angle combinations work for each distance.

## Key RL Concepts

### Policy (π)

The **policy** is the agent's strategy - a mapping from states to actions.

```
Policy: state → action
Example: distance=3.5m, bearing=45° → velocity=12, elevation=50°, azimuth=45°
```

A good policy consistently picks actions that lead to high rewards.

### Value Function (V)

The **value function** estimates how good a state is - the expected total reward from that state onward.

```
V(state) = expected future reward starting from this state
```

### Q-Function (Q)

The **Q-function** estimates how good a state-action pair is.

```
Q(state, action) = expected future reward after taking this action in this state
```

If you know Q, you can find the best action: pick the one with highest Q value.

### Exploration vs Exploitation

A fundamental challenge in RL:
- **Exploitation**: Use what you know works (shoot the way that's been successful)
- **Exploration**: Try new things (maybe there's an even better way?)

Too much exploitation = stuck in suboptimal strategies
Too much exploration = never use what you've learned

Good RL algorithms balance both.

## Neural Networks in RL

Modern RL uses neural networks to represent policies and value functions. Instead of storing a table of all possible states and actions (impossible for continuous states), a neural network learns to generalize.

```
Neural Network:
  Input:  [distance, bearing] = [3.5, 0.785]
  Hidden: Several layers of neurons
  Output: [velocity, elevation, azimuth] = [12.1, 50.3, 44.8]
```

The network learns patterns like "for longer distances, use higher velocity" without being explicitly programmed.

## Algorithms We Use

### PPO (Proximal Policy Optimization)

**Type**: Policy Gradient (directly optimizes the policy)

**How it works**:
1. Collect a batch of experiences (many shots)
2. Calculate which actions were better than expected
3. Update the policy to make good actions more likely
4. Limit how much the policy can change (the "proximal" part)

**Strengths**:
- Stable training
- Works with discrete or continuous actions
- Good default choice

**Weaknesses**:
- Sample inefficient (needs lots of data)
- May not find optimal solution for complex problems

**When to use**: Good starting point, especially for discrete actions.

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, learning_rate=3e-4)
```

### DQN (Deep Q-Network)

**Type**: Value-based (learns Q-function, derives policy from it)

**How it works**:
1. Maintain a neural network that estimates Q(state, action)
2. For each experience, update Q towards the actual reward received
3. Pick actions with highest Q value (with some exploration)

**Key innovations**:
- **Experience Replay**: Store past experiences and learn from random samples
- **Target Network**: Use a separate, slowly-updated network for stability

**Strengths**:
- Sample efficient (reuses past experiences)
- Stable with proper tuning

**Weaknesses**:
- Only works with discrete actions
- Can overestimate Q values

**When to use**: Discrete action spaces with limited actions.

```python
from stable_baselines3 import DQN
model = DQN("MlpPolicy", env, learning_rate=3e-4)
```

### SAC (Soft Actor-Critic)

**Type**: Actor-Critic (learns both policy and value function)

**How it works**:
1. **Actor**: Neural network that outputs actions (the policy)
2. **Critic**: Neural network that estimates Q values
3. **Entropy bonus**: Rewards the agent for being "uncertain" (encourages exploration)

The "soft" in SAC means it maximizes reward AND entropy:
```
objective = expected_reward + α × entropy
```

This prevents the policy from becoming too deterministic too quickly.

**Strengths**:
- State-of-the-art for continuous control
- Automatic exploration via entropy
- Sample efficient
- Stable (usually)

**Weaknesses**:
- Only works with continuous actions
- Can be unstable on simple problems (may "forget" good solutions)
- More hyperparameters to tune

**When to use**: Continuous action spaces. Best choice for our shooter problem.

```python
from stable_baselines3 import SAC
model = SAC("MlpPolicy", env, learning_rate=3e-4)
```

## Algorithm Comparison

| Aspect | PPO | DQN | SAC |
|--------|-----|-----|-----|
| Action Space | Both | Discrete only | Continuous only |
| Sample Efficiency | Low | Medium | High |
| Stability | High | Medium | Medium |
| Implementation Complexity | Medium | Medium | High |
| Best For | General purpose | Simple discrete | Continuous control |

## Our Results

### 2D Discrete (PPO)
- Environment: `ShooterEnv` (distance only, 150 actions)
- Result: 82% hit rate after 100k steps
- Limitation: Auto-aim assumed, no turret control

### 3D Discrete (PPO)
- Environment: `ShooterEnv3D` (distance + bearing, 27,000 actions)
- Result: 54% hit rate after 500k steps
- Limitation: Action space too large for efficient learning

### 3D Continuous (SAC)
- Environment: `ShooterEnvContinuous` (continuous velocity, elevation, azimuth)
- Result: **97% hit rate after 20k steps**
- Best approach for this problem

## Training Dynamics

### What Good Training Looks Like

```
Step     Reward    Status
─────────────────────────
1k       -12       Random shooting, all misses
5k       -8        Still mostly missing
10k      -3        Starting to learn
15k      +32       Hitting some shots!
20k      +85       Good performance
25k      +90       Near optimal
```

### Common Problems

**Reward not improving**:
- Learning rate too high/low
- Not enough exploration
- Bug in reward function

**Reward oscillating wildly**:
- Learning rate too high
- Batch size too small
- SAC entropy coefficient issues

**Sudden collapse after good performance**:
- Common with SAC
- Use checkpoints, pick best model
- Reduce learning rate

## Hyperparameters

Key settings that affect training:

### Learning Rate
How fast the network updates. Too high = unstable, too low = slow learning.
```python
learning_rate=3e-4  # Good default
learning_rate=1e-4  # More stable, slower
```

### Batch Size
How many experiences to use per update. Larger = more stable, slower.
```python
batch_size=256   # Good for stability
batch_size=64    # Faster updates, less stable
```

### Network Architecture
Size of the neural network. Bigger = more capacity, slower, may overfit.
```python
policy_kwargs=dict(net_arch=[256, 256])  # Two hidden layers, 256 neurons each
```

### Entropy Coefficient (SAC only)
Controls exploration. Higher = more random exploration.
```python
ent_coef="auto"  # Let SAC tune it automatically (recommended)
ent_coef=0.1     # Fixed value
```

## The Learning Process Visualized

```
Episode 1 (Random):
  Shot 1: Miss (too low)     reward: -0.4
  Shot 2: Miss (too high)    reward: -0.3
  Shot 3: Miss (wrong angle) reward: -0.5
  ...
  Average: -0.4 (0% hits)

Episode 100 (Learning):
  Shot 1: Miss (close!)      reward: -0.1
  Shot 2: HIT                reward: +1.6
  Shot 3: Miss               reward: -0.2
  ...
  Average: +0.5 (30% hits)

Episode 500 (Trained):
  Shot 1: HIT (center!)      reward: +1.9
  Shot 2: HIT                reward: +1.7
  Shot 3: HIT                reward: +1.8
  ...
  Average: +1.8 (95% hits)
```

## Why RL for Ball Shooting?

You might ask: "Why not just calculate the physics?"

Good question! You could compute the optimal trajectory analytically. But RL offers advantages:

1. **Handles uncertainty**: Real robots have noise in sensors and actuators
2. **Adapts to reality**: Can fine-tune on real robot data
3. **Generalizes**: Learns patterns, not just lookup tables
4. **Flexible**: Easy to add constraints or change objectives

In practice, you might use physics for a good starting point, then RL to optimize for real-world conditions.

## Further Reading

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - Excellent free course
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Glossary

| Term | Definition |
|------|------------|
| **Agent** | The learner/decision maker |
| **Environment** | The world the agent interacts with |
| **State** | Current situation (observation) |
| **Action** | What the agent does |
| **Reward** | Feedback signal (positive = good) |
| **Episode** | One complete run (50 shots in our case) |
| **Policy** | Strategy mapping states to actions |
| **Value** | Expected future reward |
| **Q-value** | Expected future reward for state-action pair |
| **Exploration** | Trying new actions |
| **Exploitation** | Using known good actions |
