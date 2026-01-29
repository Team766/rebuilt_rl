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

## SAC Training Loop: Annotated Code Walkthrough

This section walks through a simplified SAC training loop step-by-step. Understanding this helps demystify what happens when you call `model.learn()`.

### Overview: SAC's Neural Networks

SAC uses **five** neural networks:

```
┌─────────────────────────────────────────────────────────────────┐
│                        SAC Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ACTOR (Policy Network)                                         │
│  ┌─────────────┐                                                │
│  │ state ──────┼──► [mean, std] ──► sample action               │
│  └─────────────┘                                                │
│                                                                 │
│  CRITICS (Q-Networks) - Two of them for stability               │
│  ┌─────────────┐         ┌─────────────┐                        │
│  │ state ──────┼──► Q1   │ state ──────┼──► Q2                  │
│  │ action ─────┤         │ action ─────┤                        │
│  └─────────────┘         └─────────────┘                        │
│                                                                 │
│  TARGET CRITICS - Slowly updated copies for stable learning     │
│  ┌─────────────┐         ┌─────────────┐                        │
│  │ Q1_target   │         │ Q2_target   │                        │
│  └─────────────┘         └─────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Complete Training Loop

```python
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# =============================================================================
# STEP 1: INITIALIZE NETWORKS AND REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """Stores past experiences for learning.

    Why? Learning from random past experiences (not just recent ones)
    breaks correlations and stabilizes training.
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one experience tuple."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )

# Initialize networks (simplified - real implementation uses nn.Module classes)
# Actor: outputs mean and std of action distribution
# Critics: output Q-value for state-action pair

actor = ActorNetwork(state_dim=2, action_dim=3)      # Policy
critic1 = CriticNetwork(state_dim=2, action_dim=3)   # Q-function 1
critic2 = CriticNetwork(state_dim=2, action_dim=3)   # Q-function 2

# Target networks start as copies of critics
target_critic1 = copy.deepcopy(critic1)
target_critic2 = copy.deepcopy(critic2)

# Entropy coefficient (controls exploration)
# "auto" means we learn this too!
log_alpha = torch.zeros(1, requires_grad=True)  # learnable
target_entropy = -3.0  # target entropy = -action_dim

# Optimizers
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=3e-4)
critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=3e-4)
alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

replay_buffer = ReplayBuffer(capacity=500000)

# =============================================================================
# STEP 2: COLLECT EXPERIENCE (The "Rollout" Phase)
# =============================================================================

def collect_experience(env, actor, num_steps=1):
    """
    Interact with environment and store experiences.

    This is where the agent actually takes shots!
    """
    state, info = env.reset()

    for _ in range(num_steps):
        # Actor outputs a probability distribution over actions
        # We SAMPLE from it (not take the mean) for exploration
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Actor outputs mean and log_std of Gaussian distribution
            mean, log_std = actor(state_tensor)
            std = log_std.exp()

            # Sample action from Gaussian: action = mean + std * noise
            noise = torch.randn_like(mean)
            action = mean + std * noise

            # Squash to [-1, 1] using tanh (keeps actions bounded)
            action = torch.tanh(action)

        # Execute action in environment (take the shot!)
        action_np = action.squeeze().numpy()
        next_state, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        # Store experience in replay buffer
        # This is the "memory" that allows off-policy learning
        replay_buffer.push(state, action_np, reward, next_state, done)

        state = next_state if not done else env.reset()[0]

    return replay_buffer

# =============================================================================
# STEP 3: UPDATE CRITICS (Learn to evaluate actions)
# =============================================================================

def update_critics(batch_size=256, gamma=0.99):
    """
    Train critics to accurately predict Q-values.

    Q(s,a) should equal: reward + γ * Q(next_state, next_action)

    This is the "Bellman equation" - the foundation of Q-learning.
    """
    # Sample random batch from replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    alpha = log_alpha.exp()  # Current entropy coefficient

    # --- Compute target Q-value ---
    with torch.no_grad():
        # Ask actor: "What would you do in the next state?"
        next_mean, next_log_std = actor(next_states)
        next_std = next_log_std.exp()
        noise = torch.randn_like(next_mean)
        next_actions = torch.tanh(next_mean + next_std * noise)

        # Compute log probability of next actions (for entropy bonus)
        next_log_prob = compute_log_prob(next_mean, next_log_std, next_actions)

        # Ask TARGET critics: "How good is that next action?"
        # Use MINIMUM of two critics (prevents overestimation)
        target_q1 = target_critic1(next_states, next_actions)
        target_q2 = target_critic2(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)

        # Bellman target with entropy bonus
        # "soft" Q includes entropy: we want high reward AND high randomness
        target_q = rewards + gamma * (1 - dones) * (target_q - alpha * next_log_prob)

    # --- Update Critic 1 ---
    current_q1 = critic1(states, actions)
    critic1_loss = F.mse_loss(current_q1, target_q)  # Mean squared error

    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    critic1_optimizer.step()

    # --- Update Critic 2 ---
    current_q2 = critic2(states, actions)
    critic2_loss = F.mse_loss(current_q2, target_q)

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    critic2_optimizer.step()

    return critic1_loss.item(), critic2_loss.item()

# =============================================================================
# STEP 4: UPDATE ACTOR (Learn to take better actions)
# =============================================================================

def update_actor(batch_size=256):
    """
    Train actor to output actions that critics rate highly.

    Actor's goal: maximize Q(state, actor(state)) + entropy

    The entropy term encourages exploration - don't be too confident!
    """
    states, _, _, _, _ = replay_buffer.sample(batch_size)

    alpha = log_alpha.exp()

    # Get actions from current policy
    mean, log_std = actor(states)
    std = log_std.exp()
    noise = torch.randn_like(mean)
    actions = torch.tanh(mean + std * noise)

    # Compute log probability (measures how "confident" the policy is)
    log_prob = compute_log_prob(mean, log_std, actions)

    # Ask critics: "How good are these actions?"
    q1 = critic1(states, actions)
    q2 = critic2(states, actions)
    min_q = torch.min(q1, q2)  # Conservative estimate

    # Actor loss: we MINIMIZE this, so we MAXIMIZE (Q - alpha * log_prob)
    # - Maximize Q: take actions critics like
    # - Maximize entropy (-log_prob): stay exploratory
    actor_loss = (alpha * log_prob - min_q).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item(), log_prob.mean().item()

# =============================================================================
# STEP 5: UPDATE ENTROPY COEFFICIENT (Auto-tune exploration)
# =============================================================================

def update_alpha(log_prob):
    """
    Automatically adjust exploration level.

    If entropy is too low (policy too confident), increase alpha.
    If entropy is too high (policy too random), decrease alpha.

    This is what "auto" entropy coefficient does.
    """
    # Target: maintain entropy around target_entropy
    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    return log_alpha.exp().item()

# =============================================================================
# STEP 6: UPDATE TARGET NETWORKS (Slow tracking for stability)
# =============================================================================

def update_targets(tau=0.005):
    """
    Slowly update target networks toward current networks.

    Why not just copy? Sudden changes destabilize learning.
    Slow updates (τ=0.005 means 0.5% per update) keep things stable.

    target = τ * current + (1 - τ) * target
    """
    for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# =============================================================================
# STEP 7: THE MAIN TRAINING LOOP
# =============================================================================

def train_sac(env, total_timesteps=30000, batch_size=256, learning_starts=500):
    """
    Complete SAC training loop.

    For our ball shooter:
    - Each timestep = one shot
    - ~20,000 steps to reach 97% accuracy
    """
    state, info = env.reset()
    episode_reward = 0
    episode_num = 0

    for step in range(total_timesteps):

        # --- Collect Experience ---
        # Early on: random actions (fill replay buffer)
        # Later: use learned policy
        if step < learning_starts:
            action = env.action_space.sample()  # Random
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                mean, log_std = actor(state_t)
                std = log_std.exp()
                action = torch.tanh(mean + std * torch.randn_like(mean))
                action = action.squeeze().numpy()

        # Take action (shoot!)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store experience
        replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        # --- Learn from Experience ---
        if step >= learning_starts:
            # Update critics (learn to evaluate)
            critic_loss1, critic_loss2 = update_critics(batch_size)

            # Update actor (learn to act)
            actor_loss, log_prob = update_actor(batch_size)

            # Update entropy coefficient (auto-tune exploration)
            alpha = update_alpha(log_prob)

            # Update target networks (slow tracking)
            update_targets(tau=0.005)

        # --- Episode Management ---
        if done:
            episode_num += 1
            print(f"Episode {episode_num}: reward={episode_reward:.1f}")
            episode_reward = 0
            state, info = env.reset()
        else:
            state = next_state

    return actor  # Return trained policy

# =============================================================================
# PUTTING IT ALL TOGETHER
# =============================================================================

# In practice, you just do this:
from stable_baselines3 import SAC
from src.env.shooter_env_continuous import ShooterEnvContinuous

env = ShooterEnvContinuous()
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30000)  # All the above happens inside here!

# The trained actor can now hit 97% of shots
obs, info = env.reset()
action, _ = model.predict(obs, deterministic=True)  # No exploration noise
```

### Key Insights from the Code

1. **Two Critics**: SAC uses two Q-networks and takes the minimum. This prevents overestimation - if one critic is too optimistic, the other keeps it in check.

2. **Target Networks**: Instead of using critics directly for computing targets, SAC uses slowly-updated copies. This prevents the "moving target" problem where the network chases itself.

3. **Entropy Bonus**: The `α * log_prob` term rewards uncertainty. Early in training, the policy stays exploratory. As it learns, entropy naturally decreases.

4. **Replay Buffer**: By learning from random past experiences, SAC breaks correlations between consecutive samples. This is crucial for stable learning.

5. **Tanh Squashing**: Actions are squashed to [-1, 1] using tanh. This keeps outputs bounded and differentiable.

### What Happens During Our Training

```
Step 0-500:     Random actions, filling replay buffer
                Policy: completely random
                Reward: ~-10 (all misses)

Step 500-5000:  Critics learning to predict rewards
                Actor starting to learn patterns
                Reward: -10 to -5 (still mostly missing)

Step 5000-15000: Actor finding good strategies
                 Entropy decreasing (more confident)
                 Reward: -5 to +50 (starting to hit!)

Step 15000-20000: Fine-tuning
                  High Q-values for good actions
                  Reward: +80 to +95 (90%+ hits)
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
