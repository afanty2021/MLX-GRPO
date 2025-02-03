
# MLX-GRPO

**MLX-GRPO** is a training framework for large language models (LLMs) that leverages Apple’s MLX framework exclusively. Designed to run natively on Apple Silicon using the Metal backend, this project implements Group-based Relative Policy Optimization (GRPO) with a chain-of-thought prompting structure. The pipeline includes dataset preparation, reward function definitions, and GRPO training—all running in a pure MLX environment (no CUDA).

## Features
- **Pure MLX Integration:** Runs solely on Apple Silicon via MLX‑LM using the Metal backend.
- **GRPO Training Pipeline:** Implements multiple reward functions (e.g., correctness, format-check, XML count) to optimize chain-of-thought responses.
- **Dataset Preprocessing:** Uses the GSM8K dataset to test multi-step reasoning.
- **Modern Python Packaging:** Managed via `pyproject.toml` and launched using the `uv` CLI runner.
- **Easy to Run:** Start training with:
  ```bash
  uv run train_grpo.py
  ```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/MLX-GRPO.git
   cd MLX-GRPO
   ```

2. **Create and Activate a Virtual Environment:**
   (Ensure you have Python 3.12 or later installed.)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   This project uses a `pyproject.toml` file for dependency management. First, install the `uv` CLI runner:
   ```bash
   pip install uv
   ```
   Then, install the remaining dependencies using your preferred tool (e.g., Poetry) or via pip:
   ```bash
   pip install mlx>=0.22 mlx-lm>=0.10 torch>=2.0 datasets>=2.10 transformers>=4.40 peft>=0.2 trl>=0.3 uv>=0.0.1
   ```

## Usage

To start training using the GRPO pipeline (pure MLX), simply run:
```bash
uv run train_grpo.py
```
This command uses the `uv` CLI runner, which reads the configuration from `pyproject.toml` and runs the `train_grpo.py` script.

## Project Structure

- **train_grpo.py:** Main training script that loads the GSM8K dataset, defines reward functions, loads the model (using MLX‑LM), and runs GRPO training.
- **pyproject.toml:** Contains project metadata and dependencies.
- Additional modules and files can be added as the project evolves.

## Contributing

Contributions are welcome! Please open issues or submit pull requests with improvements or bug fixes.

## License

This project is licensed under the MIT License.



