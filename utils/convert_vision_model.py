#!/usr/bin/env python3
"""
MLX Vision Model Converter

Convert vision-language models (VLMs) to MLX format using mlx-vlm.
Supports models like DeepSeek-OCR, Qwen2-VL, LLaVA, and more.

Usage:
    python convert_vision_model.py --hf-path deepseek-ai/DeepSeek-OCR --quantize
    python convert_vision_model.py --hf-path Qwen/Qwen2-VL-2B-Instruct -q --bits 4
    python convert_vision_model.py --hf-path llava-hf/llava-1.5-7b-hf --output-dir models/llava-mlx
"""

import argparse
import os
import sys

try:
    from mlx_vlm import convert
except ImportError:
    print("=" * 80)
    print("ERROR: mlx-vlm is not installed!")
    print("=" * 80)
    print("\nPlease install mlx-vlm first:")
    print("  uv pip install mlx-vlm")
    print("\nOr if using pip:")
    print("  pip install mlx-vlm")
    print("=" * 80)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Vision-Language Models to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert DeepSeek-OCR with 4-bit quantization
  python convert_vision_model.py --hf-path deepseek-ai/DeepSeek-OCR --quantize

  # Convert Qwen2-VL without quantization
  python convert_vision_model.py --hf-path Qwen/Qwen2-VL-2B-Instruct

  # Convert with 8-bit quantization
  python convert_vision_model.py --hf-path llava-hf/llava-1.5-7b-hf -q --bits 8

  # Convert with custom output directory
  python convert_vision_model.py --hf-path deepseek-ai/DeepSeek-OCR \\
      --output-dir models/deepseek-ocr-mlx --quantize

Supported Models:
  • DeepSeek-VL-V2 / DeepSeek-OCR
  • Qwen2-VL, Qwen2.5-VL, Qwen3-VL
  • LLaVA, LLaVA-Next
  • Pixtral, Molmo, Idefics2/3
  • PaliGemma, Florence2
  • Phi3-Vision, SmolVLM
  • And more!
        """
    )

    # Required arguments
    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Hugging Face model repository path (e.g., 'deepseek-ai/DeepSeek-OCR')",
    )

    # Quantization options
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Quantize the model (default: 4-bit)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 4, 8],
        help="Quantization bits (2, 4, or 8). Default: 4",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Group size for quantization. Default: 64",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mlx_vision_model",
        help="Output directory for converted model. Default: mlx_vision_model",
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="Upload converted model to this Hugging Face repo (requires HF login)",
    )

    # Advanced options
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights. Default: float16",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed conversion progress",
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("MLX Vision Model Converter")
    print("=" * 80)
    print(f"Source Model: {args.hf_path}")
    print(f"Output Directory: {args.output_dir}")
    if args.quantize:
        print(f"Quantization: {args.bits}-bit (group_size={args.group_size})")
    else:
        print("Quantization: None")
    if args.upload_repo:
        print(f"Upload to: {args.upload_repo}")
    print("=" * 80)

    # Prepare conversion kwargs
    convert_kwargs = {
        "hf_path": args.hf_path,
        "mlx_path": args.output_dir,
        "quantize": args.quantize,
    }

    # Add quantization parameters if quantizing
    if args.quantize:
        convert_kwargs["q_bits"] = args.bits
        convert_kwargs["q_group_size"] = args.group_size

    # Add upload repo if specified
    if args.upload_repo:
        convert_kwargs["upload_repo"] = args.upload_repo

    try:
        print("\nStarting conversion...")
        print("This may take several minutes depending on model size.\n")
        
        # Convert the model using mlx-vlm
        convert(**convert_kwargs)
        
        print("\n" + "=" * 80)
        print("✓ Conversion completed successfully!")
        print("=" * 80)
        print(f"Model saved to: {args.output_dir}")
        
        if args.upload_repo:
            print(f"Model uploaded to: https://huggingface.co/{args.upload_repo}")
        
        print("\nYou can now use this model with mlx-vlm:")
        print(f'  mlx_vlm.generate --model {args.output_dir} --image path/to/image.jpg --prompt "Describe this image"')
        print("\nOr in Python:")
        print(f'  from mlx_vlm import load, generate')
        print(f'  model, processor = load("{args.output_dir}")')
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ Conversion failed!")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure the model path is correct (check https://huggingface.co/)")
        print("  - Verify the model is a supported vision-language model")
        print("  - Check your internet connection")
        print("  - Ensure you have enough disk space")
        print("\n✓ Recommended vision models that work:")
        print("  • deepseek-ai/DeepSeek-OCR")
        print("  • Qwen/Qwen2-VL-2B-Instruct")
        print("  • llava-hf/llava-1.5-7b-hf")
        print("  • HuggingFaceM4/idefics2-8b")
        print("\nFor pre-converted models, browse:")
        print("  https://huggingface.co/mlx-community")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()

