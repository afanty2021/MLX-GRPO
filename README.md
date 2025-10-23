
# MLX-GRPO

**MLX-GRPO** is a training framework for large language models (LLMs) that leverages Apple’s MLX framework exclusively. Designed to run natively on Apple Silicon using the Metal backend, this project implements Group-based Relative Policy Optimization (GRPO) with a chain-of-thought prompting structure. The pipeline includes dataset preparation, reward function definitions, and GRPO training—all running in a pure MLX environment (no CUDA).

## Features
- **Pure MLX Integration:** Runs solely on Apple Silicon via MLX‑LM using the Metal backend.
- **GRPO Training Pipeline:** Implements multiple reward functions (e.g., correctness, format-check, XML count) to optimize chain-of-thought responses.
- **Dataset Preprocessing:** Uses the GSM8K dataset to test multi-step reasoning.
- **Modern Python Packaging:** Managed via `pyproject.toml` and launched using the `uv` CLI runner.
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

To start training using the GRPO pipeline (pure MLX), simply run:

```bash
uv run mlx-grpo.py
```
This command executes `mlx-grpo.py` using the `uv` runner and the dependencies in `pyproject.toml`.

## Project Structure

- **mlx-grpo.py:** Main training script that loads the GSM8K dataset, defines reward functions, loads the model (using MLX‑LM), and runs GRPO training.
- **pyproject.toml:** Contains project metadata and dependencies.
- Additional modules and files can be added as the project evolves.

## Reproducibility

For reproducible sampling, MLX uses a global PRNG that is seeded at the start of training. The seed can be configured via `MLXGRPOConfig.seed` (default: 0). Set `mx.random.seed(config.seed)` to ensure consistent generation across runs.

## Contributing

Contributions are welcome! Please open issues or submit pull requests with improvements or bug fixes.

## License

This project is licensed under the MIT License.



