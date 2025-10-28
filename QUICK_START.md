# 🚀 Quick Start Guide

## New to the Config System?

Start here! This guide gets you up and running in 2 minutes.

## TL;DR

```bash
# Run with default settings
uv run mlx-grpo.py --config configs/default.toml

# Quick smoke test
uv run mlx-grpo.py --config configs/smoke_test.toml

# Override any setting
uv run mlx-grpo.py --config configs/default.toml --set num_generations=32
```

## Your First Run

### 1. Choose a Config

Three configs are ready to use:

| Config | Use Case | Speed | Quality |
|--------|----------|-------|---------|
| `smoke_test.toml` | Quick testing & debugging | 🚀 Fastest | Basic |
| `default.toml` | Balanced experiments | ⚡ Fast | Good |
| `production.toml` | Full training runs | 🐌 Slow | Best |

### 2. Run the Trainer

**For quick testing:**
```bash
uv run mlx-grpo.py --config configs/smoke_test.toml
```

**For balanced experiments:**
```bash
uv run mlx-grpo.py --config configs/default.toml
```

**For production:**
```bash
uv run mlx-grpo.py --config configs/production.toml
```

### 3. Tweak on the Fly

No need to edit files! Override any setting:

```bash
# Try different learning rates
uv run mlx-grpo.py --config configs/default.toml --set learning_rate=5e-7

# Use more generations
uv run mlx-grpo.py --config configs/default.toml --set num_generations=64

# Multiple overrides
uv run mlx-grpo.py --config configs/default.toml \
  --set num_generations=32 \
  --set max_new_tokens=256 \
  --set temperature=0.8
```

## Common Tasks

### Task: Try a Different Model

```bash
uv run mlx-grpo.py --config configs/default.toml \
  --set model_name="mlx-community/Qwen2.5-3B-Instruct-4bit" \
  --set output_dir="outputs/Qwen-3B-test"
```

### Task: Quick Debug Run

```bash
# Minimal generations, short sequences
uv run mlx-grpo.py --config configs/smoke_test.toml
```

### Task: Full Training Run

```bash
# All DeepSeek parameters
uv run mlx-grpo.py --config configs/production.toml
```

### Task: Create Custom Config

```bash
# Copy and edit
cp configs/default.toml configs/my_experiment.toml
# Edit configs/my_experiment.toml with your settings
uv run mlx-grpo.py --config configs/my_experiment.toml
```

### Task: Reproducible Experiments

```bash
# Set a specific seed
uv run mlx-grpo.py --config configs/default.toml --set seed=42

# Share your config file with team
git add configs/my_experiment.toml
git commit -m "Add experiment config"
```

## What Can I Configure?

### Most Common Settings

```toml
# Model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Training
learning_rate = 1e-6
num_generations = 8
max_new_tokens = 128
temperature = 0.7

# System
seed = 0
output_dir = "outputs/my_run"
```

### All Settings

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for complete list of 30+ parameters.

## Tips & Tricks

### 💡 Tip 1: Start Small
```bash
# Use smoke_test.toml first to verify everything works
uv run mlx-grpo.py --config configs/smoke_test.toml
```

### 💡 Tip 2: Use Environment Variable
```bash
# Set once, run many times
export MLX_GRPO_CONFIG=configs/my_experiment.toml
uv run mlx-grpo.py
uv run mlx-grpo.py --set seed=42
uv run mlx-grpo.py --set learning_rate=5e-7
```

### 💡 Tip 3: Parameter Sweeps
```bash
# Try multiple learning rates
for lr in 1e-6 5e-7 1e-7; do
    uv run mlx-grpo.py --config configs/default.toml \
        --set learning_rate=$lr \
        --set output_dir="outputs/lr_${lr}"
done
```

### 💡 Tip 4: Save Successful Configs
```bash
# Found good settings? Save them!
cat > configs/successful_run.toml << EOF
learning_rate = 5e-7
num_generations = 32
temperature = 0.8
seed = 42
EOF
```

## Troubleshooting

### ❌ "Config file not found"
**Solution:** Check the path. Use full path or ensure you're in the right directory.
```bash
uv run mlx-grpo.py --config /full/path/to/configs/default.toml
```

### ❌ "Unknown config key"
**Solution:** Check spelling. See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for valid keys.

### ❌ "Failed to coerce value"
**Solution:** Check the value type. Numbers shouldn't have quotes in CLI:
```bash
# ✅ Good
--set num_generations=32

# ❌ Bad (will cause issues)
--set num_generations="32"
```

## Next Steps

1. ✅ Run your first training with `configs/smoke_test.toml`
2. ✅ Try different settings with `--set` overrides
3. ✅ Create your own config file
4. ✅ Read [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for advanced usage

## Need Help?

- 📖 Full documentation: [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- 📋 All changes: [CHANGELOG.md](CHANGELOG.md)
- 🔄 Migration notes: [.config-migration-notes.md](.config-migration-notes.md)
- 📊 Implementation details: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## Example Session

```bash
# Terminal session showing typical usage

$ cd MLX-GRPO

# Quick test
$ uv run mlx-grpo.py --config configs/smoke_test.toml
[config] loading configs/smoke_test.toml
================================================================================
MLX-GRPO Training Pipeline
================================================================================
Model: Qwen/Qwen2.5-1.5B-Instruct
Output Dir: outputs/smoke-test
...

# That worked! Try with more generations
$ uv run mlx-grpo.py --config configs/smoke_test.toml --set num_generations=16
[config] loading configs/smoke_test.toml
[config] applying overrides: ['num_generations=16']
...

# Create custom config
$ cp configs/default.toml configs/my_run.toml
$ vim configs/my_run.toml  # Edit settings

# Run with custom config
$ uv run mlx-grpo.py --config configs/my_run.toml
...

# Success! 🎉
```

---

**Happy training! 🚀**

