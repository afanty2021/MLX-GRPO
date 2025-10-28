#!/usr/bin/env python3
"""
MLX Model Inference Script

Run inference with MLX-converted models in multiple modes:
- Generate: Single prompt text generation
- Chat: Interactive chat REPL
- Batch: Generate responses for multiple prompts
- Stream: Stream tokens as they're generated

Usage:
    # Single generation
    python inference.py --model mlx_model --prompt "Explain quantum computing"

    # Interactive chat
    python inference.py --model mlx_model --chat

    # Streaming generation
    python inference.py --model mlx_model --prompt "Write a story" --stream

    # With system prompt
    python inference.py --model mlx_model --prompt "What is 2+2?" --system "You are a math tutor"
"""

import argparse
import sys
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors


def format_chat_prompt(messages, tokenizer):
    """Format messages using chat template if available."""
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        # Fallback to simple format
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
        return prompt


def run_chat(model, tokenizer, args):
    """Interactive chat REPL."""
    print("=" * 80)
    print("MLX Chat Interface")
    print("=" * 80)
    print("Type your messages and press Enter. Type 'quit' or 'exit' to end.")
    print("Type 'clear' to reset conversation history.")
    print("=" * 80)
    print()

    # Initialize conversation with system prompt if provided
    conversation = []
    if args.system:
        conversation.append({"role": "system", "content": args.system})
        print(f"System: {args.system}\n")

    # Setup sampler
    sampler = make_sampler(
        args.temperature, 
        top_p=args.top_p, 
        min_p=0.0, 
        min_tokens_to_keep=1
    )
    logits_processors = make_logits_processors(
        None, 
        repetition_penalty=args.repetition_penalty, 
        repetition_context_size=20
    )

    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
                
            if user_input.lower() == "clear":
                conversation = []
                if args.system:
                    conversation.append({"role": "system", "content": args.system})
                print("\n[Conversation cleared]\n")
                continue

            # Add user message
            conversation.append({"role": "user", "content": user_input})
            
            # Format prompt
            prompt = format_chat_prompt(conversation, tokenizer)
            
            # Generate response
            print("Assistant: ", end="", flush=True)
            
            if args.stream:
                response_text = ""
                for response in stream_generate(
                    model, 
                    tokenizer, 
                    prompt, 
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                ):
                    print(response.text[len(response_text):], end="", flush=True)
                    response_text = response.text
                print()
            else:
                response_text = generate(
                    model, 
                    tokenizer, 
                    prompt=prompt, 
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    verbose=False,
                )
                print(response_text)
            
            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": response_text})
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def run_generate(model, tokenizer, args):
    """Single prompt generation."""
    # Prepare prompt
    if args.system:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt}
        ]
        prompt = format_chat_prompt(messages, tokenizer)
    else:
        prompt = args.prompt

    # Setup sampler
    sampler = make_sampler(
        args.temperature, 
        top_p=args.top_p, 
        min_p=0.0, 
        min_tokens_to_keep=1
    )
    logits_processors = make_logits_processors(
        None, 
        repetition_penalty=args.repetition_penalty, 
        repetition_context_size=20
    )

    print("=" * 80)
    print("Generating response...")
    print("=" * 80)
    print()

    if args.stream:
        # Stream generation
        response_text = ""
        for response in stream_generate(
            model, 
            tokenizer, 
            prompt, 
            max_tokens=args.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            print(response.text[len(response_text):], end="", flush=True)
            response_text = response.text
        print()
    else:
        # Standard generation
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=args.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=args.verbose,
        )
        print(response)
    
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with MLX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt generation
  python inference.py --model mlx_model --prompt "Explain quantum computing"

  # Interactive chat
  python inference.py --model mlx_model --chat

  # Streaming generation
  python inference.py --model mlx_model --prompt "Write a story" --stream

  # With system prompt
  python inference.py --model mlx_model \\
      --prompt "What is 2+2?" \\
      --system "You are a helpful math tutor"

  # Adjust sampling parameters
  python inference.py --model mlx_model \\
      --prompt "Be creative" \\
      --temperature 0.9 \\
      --top-p 0.95 \\
      --max-tokens 1024
        """
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_model",
        help="Path to MLX model directory or HuggingFace repo (default: mlx_model)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code (required for some models like Qwen)",
    )

    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive chat mode",
    )
    mode_group.add_argument(
        "--prompt",
        type=str,
        help="Prompt for single generation",
    )

    # Generation arguments
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt to guide model behavior",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7). Use 0.0 for greedy decoding",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p (default: 0.95)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (default: 1.0, no penalty)",
    )

    # Output arguments
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they are generated",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generation statistics",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.chat and not args.prompt:
        parser.error("Either --chat or --prompt must be specified")

    # Load model
    print("=" * 80)
    print("Loading model...")
    print("=" * 80)
    print(f"Model: {args.model}")
    print()

    try:
        tokenizer_config = {}
        if args.trust_remote_code:
            tokenizer_config["trust_remote_code"] = True
        
        model, tokenizer = load(args.model, tokenizer_config=tokenizer_config)
        
        print("✓ Model loaded successfully")
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure the model path exists")
        print("  - Some models require --trust-remote-code flag")
        print("  - Check model format is compatible with MLX")
        sys.exit(1)

    # Run appropriate mode
    if args.chat:
        run_chat(model, tokenizer, args)
    else:
        run_generate(model, tokenizer, args)


if __name__ == "__main__":
    main()

