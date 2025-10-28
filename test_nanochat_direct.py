#!/usr/bin/env python3
"""
Test nanochat model directly with Python API using tiktoken tokenizer
"""

import pickle
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import nanochat
from mlx_lm.utils import load_model
from pathlib import Path
import json

def load_nanochat_with_tiktoken():
    """Load nanochat model with tiktoken tokenizer"""
    
    model_dir = Path("models/nanochat-mlx")
    cache_dir = Path("/Users/skirano/.cache/huggingface/hub/models--sdobson--nanochat/snapshots/5a27404479836cf3a5ae5d9c4273d43bb17dc075")
    
    # Load tiktoken tokenizer
    print("Loading tiktoken tokenizer...")
    with open(cache_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    # Load config
    print("Loading model config...")
    with open(model_dir / "config.json", "r") as f:
        config_dict = json.load(f)
    
    # Create model args
    model_args = nanochat.ModelArgs(
        hidden_size=config_dict["hidden_size"],
        num_hidden_layers=config_dict["num_hidden_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict["num_key_value_heads"],
        vocab_size=config_dict["vocab_size"],
        max_position_embeddings=config_dict["max_position_embeddings"],
        intermediate_size=config_dict["intermediate_size"],
        rope_theta=config_dict["rope_theta"],
    )
    
    # Create model
    print("Creating model...")
    model = nanochat.Model(model_args)
    
    # Load weights
    print("Loading weights...")
    weights = mx.load(str(model_dir / "model.safetensors"))
    model.load_weights(list(weights.items()), strict=False)
    
    return model, tokenizer, model_args

def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.8):
    """Generate text with the model"""
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    print(f"Prompt tokens: {tokens[:10]}... ({len(tokens)} total)")
    
    # Convert to MLX array
    x = mx.array([tokens])
    
    print(f"Generating {max_tokens} tokens...")
    
    # Generate tokens one at a time
    for i in range(max_tokens):
        # Forward pass
        logits = model(x)
        
        # Get last token logits
        next_logits = logits[0, -1, :]
        
        # Sample from logits
        if temperature > 0:
            next_logits = next_logits / temperature
            probs = mx.softmax(next_logits, axis=-1)
            next_token = mx.random.categorical(probs)
        else:
            next_token = mx.argmax(next_logits, axis=-1)
        
        # Append token
        next_token_item = int(next_token.item())
        x = mx.concatenate([x, mx.array([[next_token_item]])], axis=1)
        
        # Decode and print
        decoded = tokenizer.decode([next_token_item])
        print(decoded, end="", flush=True)
        
        # Check for EOS (try to get EOT token safely)
        try:
            if next_token_item == tokenizer.eot_token:
                break
        except (KeyError, AttributeError):
            pass  # No EOT token defined, continue generating
    
    print("\n")
    
    # Decode full output
    output_tokens = x[0].tolist()
    full_output = tokenizer.decode(output_tokens)
    return full_output

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hi, how are you?")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Loading nanochat model...")
    print("=" * 60)
    
    model, tokenizer, model_args = load_nanochat_with_tiktoken()
    
    print(f"\nModel loaded successfully!")
    print(f"Vocab size: {tokenizer.n_vocab}")
    print(f"Model layers: {model_args.num_hidden_layers}")
    
    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print("=" * 60)
    print()
    
    output = generate_text(model, tokenizer, args.prompt, args.max_tokens, args.temperature)
    
    print("\n" + "=" * 60)
    print("Full output:")
    print("=" * 60)
    print(output)

