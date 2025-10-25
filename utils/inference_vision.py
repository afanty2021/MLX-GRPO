#!/usr/bin/env python3
"""
MLX Vision Model Inference Script

Run inference with MLX-converted vision-language models.
Supports image understanding, OCR, visual question answering, and more.

Usage:
    # Image understanding
    python inference_vision.py --model mlx_vision_model --image photo.jpg --prompt "Describe this image"

    # OCR with DeepSeek-OCR
    python inference_vision.py --model models/DeepSeek-OCR-mlx --image document.png --prompt "Extract all text"

    # Interactive chat with images
    python inference_vision.py --model mlx_vision_model --chat

    # Multiple images
    python inference_vision.py --model mlx_vision_model --image img1.jpg --image img2.jpg --prompt "Compare these images"
"""

import argparse
import sys
from pathlib import Path

try:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    from PIL import Image
except ImportError as e:
    print("=" * 80)
    print("ERROR: Required packages not installed!")
    print("=" * 80)
    print(f"\nMissing: {e}")
    print("\nPlease install mlx-vlm:")
    print("  uv pip install mlx-vlm")
    print("\nOr if using pip:")
    print("  pip install mlx-vlm Pillow")
    print("=" * 80)
    sys.exit(1)


def load_image(image_path):
    """Load an image from file path or URL."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path  # mlx-vlm can handle URLs directly
    
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return str(path.absolute())


def run_single_generation(model, processor, config, args):
    """Generate a single response for an image and prompt."""
    
    # Load images
    images = []
    if args.image:
        for img_path in args.image:
            try:
                images.append(load_image(img_path))
                print(f"Loaded image: {img_path}")
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
    
    if not images and not args.prompt:
        print("Error: Must provide at least --image or --prompt")
        return
    
    # Prepare prompt
    prompt = args.prompt if args.prompt else "Describe this image in detail."
    
    # Apply chat template
    if args.system:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = apply_chat_template(
            processor, config, prompt, num_images=len(images)
        )
    else:
        formatted_prompt = apply_chat_template(
            processor, config, prompt, num_images=len(images)
        )
    
    print("\n" + "=" * 80)
    print("Generating response...")
    print("=" * 80)
    
    # Generate
    try:
        output = generate(
            model,
            processor,
            formatted_prompt,
            images if images else None,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
        )
        
        print("\n" + "-" * 80)
        print("Response:")
        print("-" * 80)
        print(output)
        print("-" * 80 + "\n")
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()


def run_chat(model, processor, config, args):
    """Interactive chat mode with vision support."""
    print("\n" + "=" * 80)
    print("MLX Vision Model Chat")
    print("=" * 80)
    print("Commands:")
    print("  - Type your message and press Enter")
    print("  - 'image <path>' to load an image")
    print("  - 'clear' to reset conversation")
    print("  - 'quit', 'exit', or 'q' to exit")
    print("=" * 80 + "\n")
    
    conversation = []
    current_images = []
    
    if args.system:
        conversation.append({"role": "system", "content": args.system})
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "clear":
                conversation = []
                current_images = []
                if args.system:
                    conversation.append({"role": "system", "content": args.system})
                print("Conversation cleared.")
                continue
            
            if user_input.lower().startswith("image "):
                img_path = user_input[6:].strip()
                try:
                    img = load_image(img_path)
                    current_images.append(img)
                    print(f"Loaded image: {img_path} (Total: {len(current_images)})")
                except Exception as e:
                    print(f"Error loading image: {e}")
                continue
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            # Format prompt
            formatted_prompt = apply_chat_template(
                processor, config, user_input, num_images=len(current_images)
            )
            
            # Generate response
            try:
                output = generate(
                    model,
                    processor,
                    formatted_prompt,
                    current_images if current_images else None,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    verbose=False,
                )
                
                print(f"\nAssistant: {output}\n")
                conversation.append({"role": "assistant", "content": output})
                
                # Clear images after use (unless user wants to keep discussing them)
                # current_images = []
                
            except Exception as e:
                print(f"Error generating response: {e}")
                conversation.pop()  # Remove failed user message
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with MLX vision-language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_vision_model",
        help="Path to MLX vision model or HF repo. Default: mlx_vision_model",
    )
    
    # Input options
    parser.add_argument(
        "--image",
        type=str,
        action="append",
        help="Path to image file or URL (can be repeated for multiple images)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--system",
        type=str,
        help="System prompt",
    )
    
    # Mode options
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive chat mode",
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate. Default: 512",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 = greedy). Default: 0.7",
    )
    
    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generation statistics",
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.chat and not args.prompt and not args.image:
        parser.error("Must provide --chat OR (--prompt and/or --image)")
    
    print("=" * 80)
    print("Loading model...")
    print("=" * 80)
    print(f"Model: {args.model}")
    
    try:
        # Load model and processor
        model, processor = load(args.model)
        config = load_config(args.model)
        print("✓ Model loaded successfully")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure the model path is correct")
        print("  - Check that the model was converted using convert_vision_model.py")
        print("  - Try a pre-converted model from: https://huggingface.co/mlx-community")
        sys.exit(1)
    
    # Run inference
    if args.chat:
        run_chat(model, processor, config, args)
    else:
        run_single_generation(model, processor, config, args)


if __name__ == "__main__":
    main()

