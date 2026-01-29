# Training Guide

This guide explains how to train the ball shooter RL model.

## Quick Start

```bash
# Activate your Python environment (must have gymnasium, stable-baselines3, torch)
source /path/to/your/venv/bin/activate

# Train with default settings (SAC, continuous actions, 30k steps)
python scripts/train.py --algorithm SAC --env-type continuous
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--algorithm` | PPO | RL algorithm: `PPO`, `DQN`, or `SAC` |
| `--env-type` | 2d | Environment type: `2d`, `3d`, or `continuous` |
| `--timesteps` | 30,000 | Total training steps |
| `--n-envs` | 4 | Number of parallel environments |
| `--eval-freq` | 1,000 | How often to evaluate (in steps) |
| `--learning-rate` | 3e-4 | Learning rate |
| `--seed` | 42 | Random seed for reproducibility |
| `--save-dir` | models | Directory to save trained models |
| `--log-dir` | logs | Directory for TensorBoard logs |
| `--air-resistance` | off | Enable air resistance in physics simulation |

## Recommended Configurations

### For Best Results (Recommended)
```bash
python scripts/train.py \
    --algorithm SAC \
    --env-type continuous \
    --timesteps 30000 \
    --eval-freq 1000
```

SAC with continuous actions achieves ~97% hit rate in about 20k steps.

### For Faster Training (Less Accurate)
```bash
python scripts/train.py \
    --algorithm PPO \
    --env-type 2d \
    --timesteps 100000
```

PPO with discrete 2D actions is simpler but limited to auto-aim scenarios.

### For Discrete Turret Control
```bash
python scripts/train.py \
    --algorithm PPO \
    --env-type 3d \
    --timesteps 500000
```

Uses discrete action space for velocity, elevation, and azimuth. Requires more training due to large action space (27,000 actions).

### With Air Resistance (More Realistic)
```bash
python scripts/train.py \
    --algorithm SAC \
    --env-type continuous \
    --timesteps 30000 \
    --air-resistance
```

Enables drag force simulation for more realistic ball trajectories. Air resistance reduces trajectory height by 1-6% depending on velocity and angle. Models trained with air resistance will be more accurate on real robots.

## Understanding Training Output

During training, you'll see periodic output like:

```
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 50       |  <- Episode length (50 shots)
|    ep_rew_mean     | 75.4     |  <- Average reward per episode
| time/              |          |
|    episodes        | 392      |  <- Total episodes completed
|    fps             | 43       |  <- Training speed
|    total_timesteps | 19600    |  <- Current training step
| train/             |          |
|    actor_loss      | -13.9    |  <- Policy network loss
|    critic_loss     | 4.43     |  <- Value network loss
|    ent_coef        | 0.198    |  <- Entropy coefficient (SAC)
---------------------------------
```

### Key Metrics to Watch

- **ep_rew_mean**: Average episode reward. Higher is better.
  - Negative (~-10): Model is missing most shots
  - 0-50: Model is learning, hitting some shots
  - 50-80: Good performance (~50-80% hit rate)
  - 80-100: Excellent performance (~90%+ hit rate)

- **Eval mean_reward**: Deterministic evaluation reward (more reliable than rollout).

## Training Outputs

After training, you'll find:

```
models/
└── SAC_CONT_20260128_152908/    # Algorithm_EnvType_Timestamp
    ├── best/
    │   └── best_model.zip       # Best model during training (USE THIS)
    ├── checkpoints/
    │   ├── shooter_5000_steps.zip
    │   ├── shooter_10000_steps.zip
    │   └── ...
    └── final_model.zip          # Final model (may not be best)

logs/
└── SAC_CONT_20260128_152908/
    ├── evaluations.npz          # Evaluation history
    └── SAC_1/
        └── events.out.tfevents...  # TensorBoard logs
```

**Important**: Always use `best/best_model.zip` for deployment. The final model may have degraded performance due to training instability.

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/
```
Then open http://localhost:6006 in your browser.

### Live Log Watching
```bash
# If running in background
tail -f /tmp/training.log
```

## GPU vs CPU Training

The training script automatically detects GPU availability:

```python
# From train.py - automatic device selection
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

GPU training is significantly faster. On a DGX Spark (GB10):
- 30k steps: ~11 minutes
- 500k steps: ~3.5 hours

## Troubleshooting

### Out of Memory (GPU)
Reduce batch size or number of environments:
```bash
python scripts/train.py --n-envs 1
```

### Training Instability (SAC)
SAC can collapse after finding good solutions. Solutions:
1. Use the best checkpoint, not final model
2. Reduce learning rate: `--learning-rate 1e-4`
3. Stop training earlier if performance degrades

### Slow Training
1. Use GPU if available
2. Increase parallel environments: `--n-envs 8`
3. Use a simpler environment: `--env-type 2d`

## Evaluation

After training, evaluate your model:

```bash
python scripts/evaluate.py models/SAC_CONT_*/best/best_model.zip \
    --env-type continuous \
    --episodes 100
```

See the evaluation output for hit rate by distance and sample shots.
