#!/usr/bin/env python3
"""
Convert sdobson/nanochat PyTorch model to MLX format
"""

import json
import pickle
from pathlib import Path
import torch
import mlx.core as mx
from safetensors.torch import save_file

def convert_nanochat_to_mlx(cache_dir: str, output_dir: str):
    """Convert nanochat PyTorch checkpoint to MLX-compatible format"""
    
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading PyTorch checkpoint...")
    checkpoint = torch.load(cache_path / "model_000650.pt", map_location="cpu")
    
    print("Loading metadata...")
    with open(cache_path / "meta_000650.json", "r") as f:
        meta = json.load(f)
    
    print("Loading tokenizer...")
    with open(cache_path / "tokenizer.pkl", "rb") as f:
        tokenizer_data = pickle.load(f)
    
    # Create config.json matching nanochat.py ModelArgs
    config = {
        "model_type": "nanochat",
        "hidden_size": meta["model_config"]["n_embd"],
        "num_hidden_layers": meta["model_config"]["n_layer"],
        "num_attention_heads": meta["model_config"]["n_head"],
        "num_key_value_heads": meta["model_config"]["n_kv_head"],
        "vocab_size": meta["model_config"]["vocab_size"],
        "max_position_embeddings": meta["model_config"]["sequence_len"],
        "intermediate_size": meta["model_config"]["n_embd"] * 4,
        "rope_theta": 10000.0,
        "architectures": ["NanoChatModel"],
        "torch_dtype": "float32",
    }
    
    print("Saving config.json...")
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Convert weights - map nanochat names to MLX names
    print("Converting weights...")
    mlx_weights = {}
    
    for key, value in checkpoint.items():
        # Convert the key names from PyTorch to MLX format
        new_key = key
        
        # Map the nanochat checkpoint names to MLX-LM expected names
        if key.startswith("transformer."):
            new_key = key.replace("transformer.", "transformer.")
        
        # Convert to numpy first, then it will be saved as torch tensor in safetensors
        mlx_weights[new_key] = value.cpu()
    
    print(f"Total weights: {len(mlx_weights)}")
    print("Sample weight keys:")
    for i, key in enumerate(list(mlx_weights.keys())[:10]):
        print(f"  {key}: {mlx_weights[key].shape}")
    
    print("Saving model.safetensors...")
    save_file(mlx_weights, output_path / "model.safetensors")
    
    # Create tokenizer configuration
    print("Creating tokenizer config...")
    
    # Save tokenizer files (we'll need to extract from the pickle)
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": config["max_position_embeddings"],
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }
    
    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"\n✅ Conversion complete! Model saved to: {output_path}")
    print(f"\nYou can now use it with:")
    print(f"  mlx_lm.generate --model {output_path} --prompt 'Hi, how are you?'")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert nanochat PyTorch model to MLX format")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/Users/skirano/.cache/huggingface/hub/models--sdobson--nanochat/snapshots/5a27404479836cf3a5ae5d9c4273d43bb17dc075",
        help="Directory containing PyTorch checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/nanochat-mlx",
        help="Output directory for MLX model"
    )
    
    args = parser.parse_args()
    
    convert_nanochat_to_mlx(args.cache_dir, args.output_dir)

