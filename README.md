
# MLX-GRPO

**MLX-GRPO** is a training framework for large language models (LLMs) that leverages Apple’s MLX framework exclusively. Designed to run natively on Apple Silicon using the Metal backend, this project implements Group-based Relative Policy Optimization (GRPO) with a chain-of-thought prompting structure. The pipeline includes dataset preparation, reward function definitions, and GRPO training—all running in a pure MLX environment (no CUDA).

## Features
- **Pure MLX Integration:** Runs solely on Apple Silicon via MLX‑LM using the Metal backend.
- **GRPO Training Pipeline:** Implements multiple reward functions (e.g., correctness, format-check, XML count) to optimize chain-of-thought responses.
- **Universal Model Support:** Convert and use any Hugging Face model with built-in conversion utilities.
- **Dataset Preprocessing:** Uses the GSM8K dataset to test multi-step reasoning.
- **Modern Python Packaging:** Managed via `pyproject.toml` and launched using the `uv` CLI runner.
- **Inference Tools:** Test models with generation, chat, and streaming modes.
- **Easy to Run:** Start training with:
  
  ```bash
  uv run mlx-grpo.py
  ```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Doriandarko/MLX-GRPO.git
   cd MLX-GRPO
   ```

2. **Create and Activate a Virtual Environment:**
   (Ensure you have Python 3.11 or later installed.)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   This project uses a `pyproject.toml` file for dependency management. First, install the `uv` CLI runner:
   ```bash
   pip install uv
   ```
   Then, install the remaining dependencies (pure MLX path):
   ```bash
   pip install "mlx>=0.29.3" "mlx-lm>=0.28.3" "datasets>=4.2.0" "transformers>=4.56.2" "uv>=0.0.1"
   ```

## Usage

> 🚀 **New to the config system?** Start with [QUICK_START.md](QUICK_START.md) for a 2-minute guide!

### Run with a config file

To start training using the GRPO pipeline (pure MLX), run:

```bash
uv run mlx-grpo.py --config configs/default.toml
```

This command executes `mlx-grpo.py` using the `uv` runner and the dependencies in `pyproject.toml`.

Override any setting from the command line without editing TOML:

```bash
uv run mlx-grpo.py --config configs/default.toml \
  --set num_generations=64 \
  --set max_new_tokens=512 \
  --set learning_rate=5e-7
```

You can also set the config path via env var:

```bash
export MLX_GRPO_CONFIG=configs/my_run.toml
uv run mlx-grpo.py
```

If no config file is specified, the trainer will use built-in defaults from the `MLXGRPOConfig` dataclass.

### Quick Examples

**Smoke test (fast iteration):**
```bash
uv run mlx-grpo.py --config configs/smoke_test.toml
```

**Production run:**
```bash
uv run mlx-grpo.py --config configs/production.toml
```

**Custom tweaks on the fly:**
```bash
# Start with smoke test but increase generations
uv run mlx-grpo.py --config configs/smoke_test.toml --set num_generations=16

# Try a different model
uv run mlx-grpo.py --config configs/default.toml \
  --set model_name="mlx-community/Qwen2.5-3B-Instruct-4bit" \
  --set output_dir="outputs/Qwen-3B-experiment"

# Adjust learning rate
uv run mlx-grpo.py --config configs/production.toml --set learning_rate=5e-7
```

## Configuration Files

The `configs/` directory contains example TOML configuration files:

- **default.toml:** Balanced configuration good for initial testing (8 generations, 128 tokens)
- **smoke_test.toml:** Minimal settings for quick iteration (4 generations, 64 tokens)
- **production.toml:** Full DeepSeek-inspired settings (64 generations, 512 tokens)

You can create your own config files or modify existing ones to suit your needs.

📖 **See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for complete documentation on configuration options and advanced usage.**

## Model Utilities

The `utils/` directory provides powerful utilities for working with any Hugging Face model:

### 🔄 Convert Any Model to MLX

Convert any Hugging Face model to MLX format with optional quantization:

```bash
# Convert and quantize a model to 4-bit
uv run python utils/convert_model.py \
    --hf-path mistralai/Mistral-7B-Instruct-v0.3 \
    --quantize

# Use the converted model with GRPO
uv run mlx-grpo.py \
    --config configs/prod.toml \
    --set model_name="mlx_model"
```

### 🚀 Run Inference

Test your models with multiple inference modes:

```bash
# Single prompt generation
uv run python utils/inference.py \
    --model mlx_model \
    --prompt "Explain quantum computing"

# Interactive chat
uv run python utils/inference.py \
    --model mlx_model \
    --chat

# Streaming generation
uv run python utils/inference.py \
    --model mlx_model \
    --prompt "Write a story" \
    --stream
```

📖 **See [utils/README.md](utils/README.md) for complete documentation, examples, and advanced usage.**

## Project Structure

- **mlx-grpo.py:** Main training script that loads the GSM8K dataset, defines reward functions, loads the model (using MLX‑LM), and runs GRPO training.
- **configs/:** Directory containing TOML configuration files for different training scenarios.
- **utils/:** Utility scripts for model conversion and inference. See [utils/README.md](utils/README.md).
- **pyproject.toml:** Contains project metadata and dependencies.
- Additional modules and files can be added as the project evolves.

## Reproducibility

For reproducible sampling, MLX uses a global PRNG that is seeded at the start of training. The seed can be configured via `MLXGRPOConfig.seed` (default: 0). Set `mx.random.seed(config.seed)` to ensure consistent generation across runs.

## Contributing

Contributions are welcome! Please open issues or submit pull requests with improvements or bug fixes.

## License

This project is licensed under the MIT License.



