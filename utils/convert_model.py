#!/usr/bin/env python3
"""
MLX Model Converter

Convert any Hugging Face model to MLX format with optional quantization.
This allows you to use any model with the MLX-GRPO trainer.

Usage:
    python convert_model.py --hf-path mistralai/Mistral-7B-Instruct-v0.3 --quantize
    python convert_model.py --hf-path meta-llama/Llama-2-7b-hf -q --bits 4
    python convert_model.py --hf-path Qwen/Qwen2.5-7B-Instruct --output-dir models/qwen-7b-mlx
"""

import argparse
import os
from mlx_lm import convert


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face models to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert and quantize a model to 4-bit
  python convert_model.py --hf-path mistralai/Mistral-7B-Instruct-v0.3 --quantize

  # Convert without quantization
  python convert_model.py --hf-path meta-llama/Llama-2-7b-hf

  # Convert with 8-bit quantization
  python convert_model.py --hf-path Qwen/Qwen2.5-7B-Instruct -q --bits 8

  # Convert and upload to Hugging Face
  python convert_model.py --hf-path mistralai/Mistral-7B-v0.3 \\
      --quantize --upload-repo mlx-community/my-mistral-4bit

  # Convert with custom output directory
  python convert_model.py --hf-path deepseek-ai/deepseek-coder-6.7b-instruct \\
      --output-dir models/deepseek-coder-mlx
        """
    )

    # Required arguments
    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Hugging Face model repository path (e.g., 'mistralai/Mistral-7B-Instruct-v0.3')",
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
        default="mlx_model",
        help="Output directory for converted model. Default: mlx_model",
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="Upload converted model to this Hugging Face repo (requires HF login)",
    )

    # Model-specific options
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code (required for some models like Qwen)",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="EOS token (e.g., '<|endoftext|>' for Qwen models)",
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
    print("MLX Model Converter")
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
        "upload_repo": args.upload_repo,
    }

    # Add quantization parameters if quantizing
    if args.quantize:
        convert_kwargs["q_bits"] = args.bits
        convert_kwargs["q_group_size"] = args.group_size

    # Add tokenizer config if needed
    tokenizer_config = {}
    if args.trust_remote_code:
        tokenizer_config["trust_remote_code"] = True
    if args.eos_token:
        tokenizer_config["eos_token"] = args.eos_token
    
    if tokenizer_config:
        convert_kwargs["tokenizer_config"] = tokenizer_config

    try:
        print("\nStarting conversion...")
        print("This may take several minutes depending on model size.\n")
        
        # Convert the model
        convert(**convert_kwargs)
        
        print("\n" + "=" * 80)
        print("✓ Conversion completed successfully!")
        print("=" * 80)
        print(f"Model saved to: {args.output_dir}")
        
        if args.upload_repo:
            print(f"Model uploaded to: https://huggingface.co/{args.upload_repo}")
        
        print("\nYou can now use this model with MLX-GRPO:")
        print(f'  --config configs/prod.toml --set model_name="{args.output_dir}"')
        print("\nOr run inference:")
        print(f'  python utils/inference.py --model {args.output_dir}')
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ Conversion failed!")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure the model path is correct")
        print("  - Some models require --trust-remote-code flag")
        print("  - Qwen models need --eos-token '<|endoftext|>'")
        print("  - Check your internet connection")
        print("  - Ensure you have enough disk space")
        raise


if __name__ == "__main__":
    main()

