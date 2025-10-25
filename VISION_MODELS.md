# Vision Model Support in MLX-GRPO

## 🎉 TLDR: YES, MLX DOES SUPPORT VISION MODELS!

You were absolutely right to push back! MLX has **excellent vision-language model support** through the `mlx-vlm` package.

## Quick Start: DeepSeek-OCR

Here's how to use DeepSeek-OCR with MLX (the exact model you wanted):

```bash
# 1. Install mlx-vlm + dependencies (already done in your venv!)
uv pip install mlx-vlm addict matplotlib torchvision einops

# 2. Convert DeepSeek-OCR to MLX format
uv run python utils/convert_vision_model.py \
    --hf-path deepseek-ai/DeepSeek-OCR \
    --quantize \
    --bits 4 \
    --output-dir models/DeepSeek-OCR-mlx

# 3. Use it for OCR!
uv run python utils/inference_vision.py \
    --model models/DeepSeek-OCR-mlx \
    --image your_document.png \
    --prompt "Extract all text from this image"
```

## What Changed?

### The Problem
- `mlx-lm` (text-only library) doesn't support vision models
- DeepSeek-OCR is a vision-language model (`deepseek_vl_v2`)
- Got the error: `Model type deepseek_vl_v2 not supported`

### The Solution
- MLX has a **separate package** for vision models: `mlx-vlm`
- `mlx-vlm` **DOES support** DeepSeek-VL-V2/DeepSeek-OCR!
- Created new utilities to bridge the gap

## New Utilities

### 1. `convert_vision_model.py`
Converts vision-language models from Hugging Face to MLX format.

**Features:**
- Supports 30+ vision model architectures
- Quantization (2-bit, 4-bit, 8-bit)
- Same interface as `convert_model.py`

**Usage:**
```bash
uv run python utils/convert_vision_model.py \
    --hf-path deepseek-ai/DeepSeek-OCR \
    --quantize \
    --bits 4 \
    --output-dir models/DeepSeek-OCR-mlx
```

### 2. `inference_vision.py`
Run inference with converted vision models.

**Features:**
- Single image + prompt generation
- Multi-image support
- Interactive chat mode with images
- URLs and local files supported

**Usage:**
```bash
# Basic OCR
uv run python utils/inference_vision.py \
    --model models/DeepSeek-OCR-mlx \
    --image document.jpg \
    --prompt "Extract text"

# Interactive chat
uv run python utils/inference_vision.py \
    --model models/DeepSeek-OCR-mlx \
    --chat
# Then: > image photo.jpg
#       > What's in this image?

# Multiple images
uv run python utils/inference_vision.py \
    --model models/DeepSeek-OCR-mlx \
    --image img1.jpg \
    --image img2.jpg \
    --prompt "Compare these"
```

## Supported Vision Models

MLX-VLM supports an impressive list of vision models:

### OCR & Document Models
- ✅ **DeepSeek-VL-V2 / DeepSeek-OCR** (your model!)
- Florence2
- Kimi-VL

### General Vision-Language Models
- Qwen2-VL, Qwen2.5-VL, Qwen3-VL
- LLaVA, LLaVA-Next, LLaVA-Bunny
- Pixtral (Mistral's vision model)
- Phi3-Vision
- PaliGemma
- SmolVLM
- Idefics2, Idefics3
- Molmo
- InternVL-Chat
- Llama4 (when released)

### Multi-Modal (Image + Audio/Video)
- Gemma3, Gemma3n
- Mistral3

## Common Use Cases

### OCR (Your Use Case)
```bash
uv run python utils/inference_vision.py \
    --model models/DeepSeek-OCR-mlx \
    --image receipt.jpg \
    --prompt "Extract all text and itemize"
```

### Visual Question Answering
```bash
uv run python utils/inference_vision.py \
    --model models/qwen2-vl-mlx \
    --image photo.jpg \
    --prompt "What objects are in this image?"
```

### Document Understanding
```bash
uv run python utils/inference_vision.py \
    --model models/DeepSeek-OCR-mlx \
    --image chart.png \
    --prompt "Describe the trends in this chart"
```

### Image Comparison
```bash
uv run python utils/inference_vision.py \
    --model models/llava-mlx \
    --image before.jpg \
    --image after.jpg \
    --prompt "What changed between these images?"
```

## Technical Details

### Architecture Support
MLX-VLM implements the vision encoders, fusion layers, and image preprocessing that `mlx-lm` lacks:

- Vision encoders (CLIP, SigLIP, etc.)
- Image preprocessing pipelines
- Vision-language fusion layers
- Multi-modal attention mechanisms

### Model Files
The `mlx-vlm` library in `/Users/skirano/code/MLX-GRPO/.venv/lib/python3.11/site-packages/mlx_vlm/models/` includes:

```
deepseek_vl_v2/        ← DeepSeek-OCR support!
qwen2_vl/
qwen2_5_vl/
qwen3_vl/
llava/
llava_next/
pixtral/
phi3_v/
florence2/
...and 20+ more
```

### Quantization
Vision models can be quantized just like text models:
- **4-bit**: Good balance (recommended)
- **8-bit**: Better quality, more memory
- **2-bit**: Maximum compression

The vision encoder and language model are both quantized.

## Performance Tips

### Memory Usage
Vision models use more memory than text-only models:
- 2B model + vision: ~4-6GB
- 7B model + vision: ~8-12GB
- Use quantization to reduce memory

### Speed
- First inference is slower (model compilation)
- Subsequent inferences are fast
- Image preprocessing adds overhead
- Quantized models are significantly faster

## Finding Pre-Converted Models

Check HuggingFace MLX Community for pre-converted models:
- https://huggingface.co/mlx-community

Search for:
- `Qwen2-VL-*-4bit`
- `llava-*-4bit`
- Vision models with "mlx" in the name

## Troubleshooting

### Missing dependencies (addict, matplotlib, torchvision, einops)
```
Error: This modeling file requires the following packages that were not found
in your environment: addict, matplotlib, torchvision
Error: No module named 'einops'
```

✅ **Solution**: Install all required packages for DeepSeek-OCR:
```bash
uv pip install addict matplotlib torchvision einops
```

These are required for DeepSeek-OCR and some other vision models.

### Model not found
✅ Use `convert_vision_model.py` (not `convert_model.py`)
✅ Check model name: `deepseek-ai/DeepSeek-OCR` (capital letters)

### Out of memory
✅ Use 4-bit quantization: `--bits 4`
✅ Try a smaller model (2B instead of 7B)
✅ Increase wired memory: `sudo sysctl iogpu.wired_limit_mb=16384`

### Import errors
✅ Install mlx-vlm: `uv pip install mlx-vlm`
✅ Install DeepSeek dependencies: `uv pip install addict matplotlib torchvision einops`
✅ Check it's in your venv: `pip show mlx-vlm`

### Slow inference
✅ Ensure model is quantized
✅ First run is always slower (compilation)
✅ Try smaller images (resize to 512x512)

## Resources

- **MLX-VLM GitHub**: https://github.com/Blaizzy/mlx-vlm
- **DeepSeek-OCR Model**: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **MLX Community Models**: https://huggingface.co/mlx-community
- **Full Documentation**: See `utils/README.md`

## Summary

✅ **MLX DOES support vision models** through `mlx-vlm`  
✅ **DeepSeek-OCR IS supported** as `deepseek_vl_v2`  
✅ **You can convert it now** using the new utility  
✅ **30+ vision models available** including all major VLMs  

You were right to dig deeper! 🎉

