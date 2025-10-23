import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from mlx_lm import load as mlx_load, generate as mlx_generate
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import math
from mlx.optimizers import Adam
import re
import copy

# -------------------------------------------------------------------
# Dataset Preparation and Formatting
# -------------------------------------------------------------------
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Uncomment the middle messages below for 1-shot prompting if desired.
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            # {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            # {'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #     reasoning="9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
            #     answer="7"
            # )},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })  # type: ignore
    return data  # type: ignore

dataset = get_gsm8k_questions()

# -------------------------------------------------------------------
# Reward Functions
# -------------------------------------------------------------------
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# -------------------------------------------------------------------
# Model Configuration and Loading (Pure MLX)
# -------------------------------------------------------------------
# Use a model name (here using Qwen as an example)
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "outputs/Qwen-1.5B-MLX-GRPO"
run_name = "Qwen-1.5B-MLX-GRPO-gsm8k"

# Update training args to be MLX compatible
training_args = {
    'output_dir': output_dir,
    'run_name': run_name,
    'learning_rate': 5e-6,
    'batch_size': 1,
    'gradient_accumulation_steps': 4,
    'num_epochs': 1,
    'warmup_ratio': 0.1,
    'max_grad_norm': 0.1,
    'logging_steps': 1
}

# Note: MLX-LM has built-in LoRA support via command-line tools
# For training with LoRA, you can use: python -m mlx_lm.lora --model <model> --train
# This implementation focuses on full model fine-tuning with GRPO

def load_model(model_name):
    """Load model and tokenizer using MLX-LM"""
    model, tokenizer = mlx_load(model_name)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_model_params(model):
    """Extract trainable parameters from MLX model"""
    # MLX-LM models are typically nn.Module instances or dicts
    if isinstance(model, nn.Module):
        return model.trainable_parameters()
    elif isinstance(model, dict) and 'model' in model:
        return model['model'].trainable_parameters()
    else:
        raise ValueError(f"Unexpected model type: {type(model)}")

def calculate_log_probs_single(model, tokenizer, prompt: str, completion: str) -> mx.array:
    """
    Calculate log probability of a single completion given a prompt.
    Used during generation phase to compute old_log_probs.

    Returns the sum of log probabilities for all completion tokens.
    """
    # Tokenize prompt and completion separately to know boundaries
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    # Create full sequence
    full_tokens = prompt_tokens + completion_tokens
    input_ids = mx.array(full_tokens)[None, :]  # Add batch dimension

    # Get model logits
    if isinstance(model, nn.Module):
        logits = model(input_ids)
    elif isinstance(model, dict) and 'model' in model:
        logits = model['model'](input_ids)
    else:
        raise ValueError(f"Unexpected model type: {type(model)}")

    # Compute log probabilities using log_softmax
    # Shape: [1, seq_len, vocab_size]
    log_probs_full = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Extract log probs for completion tokens
    # log_probs_full[i] predicts token at position i+1
    prompt_len = len(prompt_tokens)
    completion_len = len(completion_tokens)

    # Get log probs for positions that predict completion tokens
    completion_log_probs = []
    for i in range(completion_len):
        pos = prompt_len - 1 + i  # Position in sequence
        if pos < len(full_tokens) - 1:
            next_token_id = full_tokens[pos + 1]
            log_prob = log_probs_full[0, pos, next_token_id]
            completion_log_probs.append(log_prob)

    # Sum log probabilities
    if len(completion_log_probs) > 0:
        return mx.sum(mx.stack(completion_log_probs))
    else:
        return mx.array(0.0)

# Load model and tokenizer
model, tokenizer = load_model(model_name)

# -------------------------------------------------------------------
# Initialize and Run GRPO Training (Pure MLX)
# -------------------------------------------------------------------
@dataclass
class MLXGRPOConfig:
    """Configuration class for MLX GRPO training"""
    output_dir: str
    run_name: str
    learning_rate: float = 1e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.1
    logging_steps: int = 1
    num_generations: int = 64  # DeepSeekMath uses 64 samples per prompt
    max_prompt_length: int = 512
    max_completion_length: int = 1024  # DeepSeekMath uses 1024
    max_new_tokens: int = 512
    temperature: float = 0.7
    clip_eps: float = 0.2  # PPO clipping epsilon
    kl_coeff: float = 0.0  # KL coefficient (can set to 0.04 as in DeepSeek)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    lr_scheduler_type: str = 'cosine'
    save_steps: int = 100

class MLXGRPOTrainer:
    def __init__(self, model, tokenizer, reward_funcs, args: MLXGRPOConfig, train_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.args = args
        self.train_dataset = train_dataset

        # CRITICAL: Three models as per article
        # 1. model - current trainable policy (π_θ)
        # 2. model_old - policy that generates responses (π_θ_old), synced periodically
        # 3. ref_model - original pretrained model (π_ref), never updated
        self.model_old = copy.deepcopy(model)  # Starts same as model
        self.ref_model = copy.deepcopy(model)  # Never changes

        # Initialize optimizer with correct API
        # Get trainable parameters
        if isinstance(model, nn.Module):
            params = model.trainable_parameters()
        elif isinstance(model, dict) and 'model' in model:
            params = model['model'].trainable_parameters()
        else:
            params = {}

        self.optimizer = Adam(learning_rate=args.learning_rate)

        self.step = 0
        self.total_steps = len(train_dataset) * args.num_epochs
        self.update_every = 10  # Sync model_old every N steps
        
    def _get_scheduler(self):
        """Implements cosine learning rate schedule"""
        warmup_steps = int(self.total_steps * self.args.warmup_ratio)
        
        def lr_schedule(step):
            # Linear warmup
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
        return self.args.learning_rate * lr_schedule
    
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string"""
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted += f"System: {content}\n\n"
            elif role == 'user':
                formatted += f"User: {content}\n\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n\n"
        formatted += "Assistant: "  # Prompt for assistant response
        return formatted

    def generate_responses(self, batch):
        """
        Generate multiple responses for a prompt using model_old.
        Also computes old_log_probs for each response.

        Returns:
            responses: List of generated strings
            old_log_probs: List of log probability values (mx.array)
            formatted_prompt: The formatted prompt string
        """
        messages = batch['prompt']
        formatted_prompt = self.format_prompt(messages)

        responses = []
        old_log_probs = []

        for i in range(self.args.num_generations):
            try:
                # CRITICAL: Generate with model_old (not self.model)
                output = mlx_generate(
                    self.model_old,  # Use old model for generation!
                    self.tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=self.args.max_new_tokens,
                    temp=self.args.temperature,
                    verbose=False
                )
                responses.append(output)

                # CRITICAL: Compute old_log_probs during generation
                # This is π_θ_old(o_i|q) - probability under the old policy
                log_prob = calculate_log_probs_single(
                    self.model_old,
                    self.tokenizer,
                    formatted_prompt,
                    output
                )
                old_log_probs.append(log_prob)

            except Exception as e:
                print(f"Generation {i} failed: {e}")
                # Fallback to empty response with zero log prob
                responses.append("")
                old_log_probs.append(mx.array(0.0))

        return responses, old_log_probs, formatted_prompt
    
    def compute_rewards(self, batch, responses: List[str]) -> mx.array:
        """
        Compute rewards for all responses using reward functions.
        Returns normalized advantages based on group mean.
        """
        # Prepare completions in the format expected by reward functions
        # Each response needs to be wrapped in the format reward functions expect
        completions = [[{"content": response}] for response in responses]

        rewards = []
        for response, completion in zip(responses, completions):
            total_reward = 0.0

            # Apply each reward function
            for reward_fn in self.reward_funcs:
                try:
                    # Call reward function with appropriate arguments
                    result = reward_fn(
                        prompts=[batch['prompt']],
                        completions=[completion],
                        answer=[batch.get('answer', '')]
                    )
                    # Sum up rewards if it returns a list
                    if isinstance(result, list):
                        total_reward += sum(result)
                    else:
                        total_reward += float(result)
                except Exception as e:
                    print(f"Reward function failed: {e}")
                    pass

            rewards.append(total_reward)

        rewards = mx.array(rewards)

        # GRPO: Compute advantages relative to group mean (baseline)
        mean_reward = mx.mean(rewards)
        std_reward = mx.std(rewards)

        # Normalize: A_i = (R_i - mean(G)) / (std(G) + eps)
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        return advantages, rewards
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        mx.save(os.path.join(path, "model.safetensors"), self.model)
        
        # Save optimizer state
        mx.save(os.path.join(path, "optimizer.safetensors"), self.optimizer.state)
        
        # Save training state
        training_state = {
            "step": self.step,
            "args": self.args.__dict__
        }
        mx.save(os.path.join(path, "trainer_state.safetensors"), training_state)

    def compute_grpo_loss(self, prompt: str, responses: List[str],
                          advantages: mx.array, old_log_probs: List[mx.array]):
        """
        Compute GRPO loss following the article's implementation.

        Args:
            prompt: The formatted prompt string
            responses: List of generated completions
            advantages: Normalized advantages (A_i)
            old_log_probs: Pre-computed log probs from model_old

        Returns:
            loss: Total loss (to be minimized)
            policy_reward_mean: Mean policy reward (for logging)
            kl_div_mean: Mean KL divergence (for logging)
        """
        policy_rewards = []
        kl_divs = []

        for response, advantage, old_log_prob in zip(responses, advantages, old_log_probs):
            try:
                # Compute log_prob with current model (π_θ)
                log_prob_current = calculate_log_probs_single(
                    self.model, self.tokenizer, prompt, response
                )

                # Compute log_prob with reference model (π_ref)
                log_prob_ref = calculate_log_probs_single(
                    self.ref_model, self.tokenizer, prompt, response
                )

                # PPO-clip objective: ratio = π_θ / π_θ_old
                ratio = mx.exp(log_prob_current - old_log_prob)
                clipped_ratio = mx.clip(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps)

                # Policy reward = min(ratio * A, clip(ratio) * A)
                policy_reward = mx.minimum(ratio * advantage, clipped_ratio * advantage)
                policy_rewards.append(policy_reward)

                # KL penalty: D_KL(π_θ || π_ref) = r - log(r) - 1
                # where r = π_ref / π_θ
                log_ratio_for_kl = log_prob_ref - log_prob_current
                ratio_for_kl = mx.exp(log_ratio_for_kl)
                kl_div = ratio_for_kl - log_ratio_for_kl - 1
                kl_divs.append(kl_div)

            except Exception as e:
                print(f"Loss computation failed: {e}")
                policy_rewards.append(mx.array(0.0))
                kl_divs.append(mx.array(0.0))

        # Average over all responses
        policy_reward_mean = mx.mean(mx.stack(policy_rewards))
        kl_div_mean = mx.mean(mx.stack(kl_divs))

        # Total objective: maximize (policy_reward - β * kl_div)
        # Return negative for minimization
        loss = -mx.mean(mx.stack([pr - self.args.kl_coeff * kl
                                   for pr, kl in zip(policy_rewards, kl_divs)]))

        return loss, policy_reward_mean, kl_div_mean

    def train_step(self, batch):
        """
        Performs a single training step using GRPO (following the article).

        Steps:
        1. Generate N responses using model_old and compute old_log_probs
        2. Compute rewards and advantages (relative to group mean)
        3. Compute GRPO loss (uses model, ref_model, and old_log_probs)
        4. Update model parameters
        5. Periodically sync model_old from model
        """
        print(f"\n{'='*60}")
        print(f"Training step {self.step + 1}")
        print(f"{'='*60}")

        # 1. Generate multiple responses with model_old
        responses, old_log_probs, formatted_prompt = self.generate_responses(batch)
        print(f"Generated {len(responses)} responses using model_old")

        # 2. Compute rewards and advantages
        advantages, rewards = self.compute_rewards(batch, responses)
        print(f"Rewards - Mean: {float(mx.mean(rewards)):.3f}, Std: {float(mx.std(rewards)):.3f}")
        print(f"Advantages - Mean: {float(mx.mean(advantages)):.3f}, Std: {float(mx.std(advantages)):.3f}")

        # Display sample response
        if len(responses) > 0:
            print(f"\n--- Sample Response (Reward: {float(rewards[0]):.3f}) ---")
            print(responses[0][:200] + "..." if len(responses[0]) > 200 else responses[0])
            print(f"---")

        # 3. Define loss function for gradient computation
        def loss_fn(params):
            """Loss function that computes GRPO objective"""
            loss, policy_reward, kl_div = self.compute_grpo_loss(
                formatted_prompt,
                responses,
                advantages,
                old_log_probs
            )
            return loss

        # 4. Compute loss and gradients
        try:
            # Get current model parameters
            if isinstance(self.model, nn.Module):
                params = self.model.trainable_parameters()
            elif isinstance(self.model, dict) and 'model' in self.model:
                params = self.model['model'].trainable_parameters()
            else:
                raise ValueError(f"Unexpected model type: {type(self.model)}")

            # Compute gradients
            loss_and_grad_fn = mx.value_and_grad(loss_fn)
            loss, grads = loss_and_grad_fn(params)

            # Gradient clipping
            grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in tree_flatten(grads)[0]))
            if grad_norm > self.args.max_grad_norm:
                grads = tree_map(lambda g: g * self.args.max_grad_norm / grad_norm, grads)

            # 5. Update parameters
            self.optimizer.update(self.model, grads)
            mx.eval(self.model)  # Ensure updates are applied

            print(f"Loss: {float(loss):.4f}, Grad Norm: {float(grad_norm):.4f}")

        except Exception as e:
            print(f"Training step failed: {e}")
            import traceback
            traceback.print_exc()
            loss = mx.array(0.0)

        self.step += 1

        # 6. Sync model_old from model periodically (as per article)
        if self.step % self.update_every == 0:
            print(f"\n*** Syncing model_old weights from model (step {self.step}) ***")
            # Copy current model to model_old
            if isinstance(self.model, nn.Module):
                self.model_old = copy.deepcopy(self.model)
            elif isinstance(self.model, dict):
                self.model_old = copy.deepcopy(self.model)
            mx.eval(self.model_old)  # Ensure copy is complete

        return loss

    def train(self):
        """Enhanced training loop with proper logging and checkpointing"""
        print(f"Starting training with {self.total_steps} total steps")
        
        for epoch in range(self.args.num_epochs):
            for batch in self.train_dataset:
                self.step += 1
                
                # Training step
                loss = self.train_step(batch)
                
                # Logging
                if self.step % self.args.logging_steps == 0:
                    print(f"Epoch {epoch}, Step {self.step}/{self.total_steps}, Loss: {loss.item():.4f}")
                
                # Save checkpoint
                if self.step % self.args.save_steps == 0:
                    checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint-{self.step}")
                    self.save_checkpoint(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

def main():
    """Main training function"""
    print("="*80)
    print("MLX-GRPO Training Pipeline")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Output Dir: {output_dir}")
    print(f"Dataset: GSM8K (train split)")
    print("="*80)

    # Initialize configuration with DeepSeek-inspired parameters
    config = MLXGRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=1e-6,  # DeepSeek uses 1e-6
        num_generations=64,  # DeepSeek uses 64 samples per prompt
        max_prompt_length=512,
        max_completion_length=1024,  # DeepSeek uses 1024
        max_new_tokens=512,
        temperature=0.7,
        clip_eps=0.2,
        kl_coeff=0.0,  # Can set to 0.04 if needed
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=1
    )

    print(f"\nTraining Configuration:")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Generations per Prompt: {config.num_generations}")
    print(f"  Max New Tokens: {config.max_new_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Clip Epsilon: {config.clip_eps}")
    print(f"  KL Coefficient: {config.kl_coeff}")
    print("="*80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model(model_name)
    print(f"Model loaded successfully")
    print(f"Model type: {type(model)}")

    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = MLXGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[
            correctness_reward_func,  # Most important - put first
            xmlcount_reward_func,
            soft_format_reward_func,
            int_reward_func,
        ],
        args=config,
        train_dataset=dataset
    )
    print("Trainer initialized")

    # Start training
    print("\n" + "="*80)
    print("Starting GRPO Training")
    print("="*80)
    trainer.train()

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Model saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
