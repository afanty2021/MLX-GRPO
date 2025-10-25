# MLX-GRPO Utilities

Convert and use **any Hugging Face model** with the MLX-GRPO trainer, including **Vision-Language Models**!

## ⚡ Quick Start

### Text-Only Models
```bash
# 1. Convert a model
uv run python utils/convert_model.py --hf-path Qwen/Qwen2.5-1.5B-Instruct --quantize

# 2. Test with chat
uv run python utils/inference.py --model mlx_model --chat

# 3. Train with GRPO
uv run mlx-grpo.py --config configs/smoke_test.toml --set model_name="mlx_model"
```

### Vision-Language Models (NEW! 🎉)
```bash
# 1. Install mlx-vlm
uv pip install mlx-vlm

# 2. Convert a vision model (like DeepSeek-OCR!)
uv run python utils/convert_vision_model.py --hf-path deepseek-ai/DeepSeek-OCR --quantize

# 3. Test with images
uv run python utils/inference_vision.py --model mlx_vision_model --image photo.jpg --prompt "Describe this image"
```

## 📦 Installation

All dependencies are managed via the project's `pyproject.toml`. The utilities use `mlx-lm` for text models and `mlx-vlm` for vision models.

### Text-Only Models
```bash
# From the project root
uv sync

# Or if using pip:
pip install mlx-lm>=0.28.3
```

### Vision-Language Models
```bash
# Install mlx-vlm for vision support
uv pip install mlx-vlm

# Some models (like DeepSeek-OCR) need additional dependencies:
uv pip install addict matplotlib torchvision einops

# Or if using pip:
pip install mlx-vlm addict matplotlib torchvision einops
```

## 🔄 Model Conversion

The `convert_model.py` script converts any Hugging Face model to MLX format, making it compatible with the GRPO trainer.

### Basic Usage

```bash
# Convert and quantize to 4-bit (recommended for training)
uv run python utils/convert_model.py \
    --hf-path mistralai/Mistral-7B-Instruct-v0.3 \
    --quantize

# Convert without quantization (full precision)
uv run python utils/convert_model.py \
    --hf-path meta-llama/Llama-2-7b-hf
```

### Advanced Options

```bash
# Convert with 8-bit quantization
uv run python utils/convert_model.py \
    --hf-path Qwen/Qwen2.5-7B-Instruct \
    --quantize \
    --bits 8

# Convert with custom output directory
uv run python utils/convert_model.py \
    --hf-path deepseek-ai/deepseek-coder-6.7b-instruct \
    --output-dir models/deepseek-coder-mlx \
    --quantize

# Convert and upload to Hugging Face (requires HF_TOKEN)
uv run python utils/convert_model.py \
    --hf-path mistralai/Mistral-7B-v0.3 \
    --quantize \
    --upload-repo mlx-community/my-mistral-4bit
```

### Special Model Requirements

Some models require additional flags:

```bash
# Qwen models (require trust_remote_code and eos_token)
uv run python utils/convert_model.py \
    --hf-path Qwen/Qwen2.5-7B-Instruct \
    --trust-remote-code \
    --eos-token "<|endoftext|>" \
    --quantize

# Models with custom tokenizers
uv run python utils/convert_model.py \
    --hf-path 01-ai/Yi-6B-Chat \
    --trust-remote-code \
    --quantize
```

### All Conversion Options

| Flag | Description | Default |
|------|-------------|---------|
| `--hf-path` | Hugging Face model repo (required) | - |
| `-q, --quantize` | Enable quantization | False |
| `--bits` | Quantization bits (2, 4, or 8) | 4 |
| `--group-size` | Quantization group size | 64 |
| `--output-dir` | Output directory for converted model | `mlx_model` |
| `--upload-repo` | Upload to HF repo | None |
| `--trust-remote-code` | Trust remote code in model | False |
| `--eos-token` | Specify EOS token | None |
| `--dtype` | Weight dtype (float16/bfloat16/float32) | float16 |
| `--verbose` | Print detailed progress | False |

## 🚀 Running Inference

The `inference.py` script provides multiple ways to interact with your converted models.

### Single Prompt Generation

```bash
# Basic generation
uv run python utils/inference.py \
    --model mlx_model \
    --prompt "Explain quantum computing in simple terms"

# With streaming output
uv run python utils/inference.py \
    --model mlx_model \
    --prompt "Write a short story about a robot" \
    --stream

# With system prompt
uv run python utils/inference.py \
    --model mlx_model \
    --prompt "What is 2+2?" \
    --system "You are a helpful math tutor. Explain step by step."
```

### Interactive Chat Mode

```bash
# Start chat REPL
uv run python utils/inference.py \
    --model mlx_model \
    --chat

# Chat with system prompt
uv run python utils/inference.py \
    --model mlx_model \
    --chat \
    --system "You are a helpful coding assistant"

# Chat with streaming
uv run python utils/inference.py \
    --model mlx_model \
    --chat \
    --stream
```

Chat commands:
- Type your message and press Enter to chat
- Type `clear` to reset conversation history
- Type `quit`, `exit`, or `q` to exit

### Adjusting Generation Parameters

```bash
# Creative generation (high temperature)
uv run python utils/inference.py \
    --model mlx_model \
    --prompt "Write a poem" \
    --temperature 0.9 \
    --top-p 0.95 \
    --max-tokens 1024

# Deterministic generation (greedy)
uv run python utils/inference.py \
    --model mlx_model \
    --prompt "What is the capital of France?" \
    --temperature 0.0

# Reduce repetition
uv run python utils/inference.py \
    --model mlx_model \
    --prompt "Tell me about AI" \
    --repetition-penalty 1.2
```

### All Inference Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Path to MLX model or HF repo | `mlx_model` |
| `--chat` | Start interactive chat mode | False |
| `--prompt` | Prompt for single generation | - |
| `--system` | System prompt | None |
| `--max-tokens` | Max tokens to generate | 512 |
| `--temperature` | Sampling temperature (0.0 = greedy) | 0.7 |
| `--top-p` | Nucleus sampling top-p | 0.95 |
| `--repetition-penalty` | Repetition penalty | 1.0 |
| `--stream` | Stream tokens as generated | False |
| `--verbose` | Print generation statistics | False |
| `--trust-remote-code` | Trust remote code | False |

## 🎯 Using Converted Models with GRPO

After converting a model, you can use it directly with the GRPO trainer:

```bash
# Convert model first
uv run python utils/convert_model.py \
    --hf-path mistralai/Mistral-7B-Instruct-v0.3 \
    --output-dir models/mistral-7b-mlx \
    --quantize

# Use with GRPO trainer
uv run mlx-grpo.py \
    --config configs/prod.toml \
    --set model_name="models/mistral-7b-mlx"
```

Or update your config file:

```toml
# configs/my_custom.toml
model_name = "models/mistral-7b-mlx"
output_dir = "outputs/mistral-7b-grpo"
# ... other settings
```

## 📋 Supported Models

The utilities support thousands of Hugging Face models, including:

### Popular Model Families

- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.3`, `mistralai/Mixtral-8x7B-Instruct-v0.1`
- **Llama**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-3.2-3B-Instruct`
- **Qwen**: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
- **DeepSeek**: `deepseek-ai/deepseek-coder-6.7b-instruct`, `deepseek-ai/deepseek-math-7b-instruct`
- **Phi**: `microsoft/phi-2`, `microsoft/phi-3-mini`
- **Yi**: `01-ai/Yi-6B-Chat`, `01-ai/Yi-34B-Chat`
- **StableLM**: `stabilityai/stablelm-2-zephyr-1_6b`
- **InternLM**: `internlm/internlm2-7b`
- **Falcon**: `tiiuae/falcon-mamba-7b-instruct`

### Finding Models

Browse the [MLX Community](https://huggingface.co/mlx-community) on Hugging Face for pre-converted models, or convert any compatible model from Hugging Face directly.

## 💡 Examples

### Complete Workflow: Custom Model Training

```bash
# 1. Convert a model
uv run python utils/convert_model.py \
    --hf-path deepseek-ai/deepseek-math-7b-instruct \
    --output-dir models/deepseek-math-mlx \
    --quantize

# 2. Test the model with inference
uv run python utils/inference.py \
    --model models/deepseek-math-mlx \
    --prompt "Solve: If x + 5 = 12, what is x?" \
    --system "You are a math tutor. Show your reasoning step by step."

# 3. Train with GRPO
uv run mlx-grpo.py \
    --config configs/prod.toml \
    --set model_name="models/deepseek-math-mlx" \
    --set output_dir="outputs/deepseek-math-grpo" \
    --set run_name="deepseek-math-gsm8k"

# 4. Test the fine-tuned model
uv run python utils/inference.py \
    --model outputs/deepseek-math-grpo/deepseek-math-gsm8k/checkpoint-1000 \
    --chat
```

### Quick Experimentation

```bash
# Try multiple models quickly
models=("Qwen/Qwen2.5-1.5B-Instruct" "microsoft/phi-2" "stabilityai/stablelm-2-zephyr-1_6b")

for model in "${models[@]}"; do
    name=$(basename "$model")
    uv run python utils/convert_model.py \
        --hf-path "$model" \
        --output-dir "models/$name-mlx" \
        --quantize
    
    uv run python utils/inference.py \
        --model "models/$name-mlx" \
        --prompt "What is 2+2?" \
        > "results/$name-output.txt"
done
```

### Converting Large Models

For large models (>10GB), consider:

```bash
# Use 4-bit quantization for memory efficiency
uv run python utils/convert_model.py \
    --hf-path meta-llama/Llama-2-70b-hf \
    --quantize \
    --bits 4 \
    --group-size 64 \
    --output-dir models/llama-70b-4bit

# Increase wired memory limit on macOS 15+ (if needed)
sudo sysctl iogpu.wired_limit_mb=32768
```

## 🔧 Troubleshooting

### Common Issues

**Model not found:**
```bash
# Ensure you're using the correct Hugging Face path
# Check: https://huggingface.co/models
uv run python utils/convert_model.py --hf-path correct/model-path --quantize
```

**Trust remote code error:**
```bash
# Add the flag for models that require it (Qwen, some Yi models)
uv run python utils/convert_model.py \
    --hf-path Qwen/Qwen2.5-7B-Instruct \
    --trust-remote-code \
    --quantize
```

**Out of memory:**
```bash
# Use more aggressive quantization
uv run python utils/convert_model.py \
    --hf-path large/model \
    --quantize \
    --bits 2  # More aggressive quantization

# Or increase system limits (macOS 15+)
sudo sysctl iogpu.wired_limit_mb=N  # N > model size in MB
```

**Slow generation:**
```bash
# Ensure model is quantized
# Check macOS version (15+ recommended for large models)
# Increase wired memory limit if you see warnings
```

## 🖼️ Vision-Language Models

MLX now supports vision-language models through the `mlx-vlm` package! This enables models like DeepSeek-OCR, Qwen2-VL, LLaVA, and many more.

### Installing Vision Support

```bash
# Using uv (recommended for this project)
uv pip install mlx-vlm

# Or using pip
pip install mlx-vlm
```

### Converting Vision Models

Use the `convert_vision_model.py` script to convert vision-language models:

```bash
# Convert DeepSeek-OCR (the model you wanted!)
uv run python utils/convert_vision_model.py \
    --hf-path deepseek-ai/DeepSeek-OCR \
    --quantize \
    --bits 4 \
    --output-dir models/DeepSeek-OCR-mlx

# Convert Qwen2-VL
uv run python utils/convert_vision_model.py \
    --hf-path Qwen/Qwen2-VL-2B-Instruct \
    --quantize \
    --output-dir models/qwen2-vl-mlx

# Convert LLaVA
uv run python utils/convert_vision_model.py \
    --hf-path llava-hf/llava-1.5-7b-hf \
    --quantize \
    --bits 4 \
    --output-dir models/llava-mlx
```

### Running Vision Model Inference

Use the `inference_vision.py` script to test your converted vision models:

```bash
# Basic image understanding
uv run python utils/inference_vision.py \
    --model models/DeepSeek-OCR-mlx \
    --image document.png \
    --prompt "Extract all text from this image"

# OCR with custom system prompt
uv run python utils/inference_vision.py \
    --model models/DeepSeek-OCR-mlx \
    --image receipt.jpg \
    --system "You are an OCR expert. Extract text accurately." \
    --prompt "Read this receipt"

# Interactive chat with images
uv run python utils/inference_vision.py \
    --model models/qwen2-vl-mlx \
    --chat

# Then in chat mode:
# > image photo.jpg
# > What do you see in this image?
# > image another.jpg
# > How is this different from the first image?

# Multiple images at once
uv run python utils/inference_vision.py \
    --model models/llava-mlx \
    --image img1.jpg \
    --image img2.jpg \
    --prompt "Compare these two images"
```

### Supported Vision Models

The `mlx-vlm` package supports a wide range of vision-language models:

#### OCR and Document Understanding
- **DeepSeek-VL-V2 / DeepSeek-OCR** - Advanced OCR and document understanding
- **Florence2** - Microsoft's OCR model
- **Kimi-VL** - Document understanding

#### General Vision-Language Models
- **Qwen2-VL, Qwen2.5-VL, Qwen3-VL** - Excellent general-purpose VLMs
- **LLaVA, LLaVA-Next, LLaVA-Bunny** - Popular open VLMs
- **Pixtral** - Mistral's vision model
- **Phi3-Vision** - Microsoft's compact VLM
- **PaliGemma** - Google's VLM
- **SmolVLM** - Compact vision model
- **Idefics2, Idefics3** - HuggingFace's VLMs
- **Molmo** - Allen AI's VLM
- **InternVL-Chat** - Strong multilingual VLM
- **Llama4** - Meta's vision-capable model (when released)

#### Multi-Modal Models
- **Gemma3, Gemma3n** - Google's multi-modal models with audio/video support
- **Mistral3** - Mistral's multi-modal model

### Vision Model Conversion Options

The `convert_vision_model.py` script supports the same options as the text model converter:

| Flag | Description | Default |
|------|-------------|---------|
| `--hf-path` | Hugging Face model repo (required) | - |
| `-q, --quantize` | Enable quantization | False |
| `--bits` | Quantization bits (2, 4, or 8) | 4 |
| `--group-size` | Quantization group size | 64 |
| `--output-dir` | Output directory | `mlx_vision_model` |
| `--upload-repo` | Upload to HF repo | None |
| `--dtype` | Weight dtype (float16/bfloat16/float32) | float16 |
| `--verbose` | Print detailed progress | False |

### Vision Model Inference Options

The `inference_vision.py` script provides these options:

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Path to MLX vision model | `mlx_vision_model` |
| `--image` | Image file or URL (repeatable) | - |
| `--prompt` | Text prompt | - |
| `--system` | System prompt | None |
| `--chat` | Interactive chat mode | False |
| `--max-tokens` | Max tokens to generate | 512 |
| `--temperature` | Sampling temperature | 0.7 |
| `--verbose` | Print statistics | False |

### Finding Pre-Converted Vision Models

Browse the [MLX Community](https://huggingface.co/mlx-community) for pre-converted vision models. Look for models with "VL", "Vision", "OCR", or "LLaVA" in their names.

Examples:
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`
- `mlx-community/llava-1.5-7b-4bit`
- Pre-converted DeepSeek-OCR models may be available

## 📚 Additional Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-lm) - Text-only models
- [MLX-VLM GitHub](https://github.com/Blaizzy/mlx-vlm) - Vision-language models
- [MLX Community on Hugging Face](https://huggingface.co/mlx-community)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [DeepSeekMath Paper](https://arxiv.org/abs/2402.03300)
- [DeepSeek-OCR Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## 🤝 Contributing

If you find issues or have suggestions for these utilities, please open an issue or pull request in the main repository.

## 📄 License

These utilities are part of the MLX-GRPO project and are licensed under the MIT License.

