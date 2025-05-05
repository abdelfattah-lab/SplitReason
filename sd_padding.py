#!/usr/bin/env python3
import sys
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Default model names
DRAFT_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TARGET_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

"""
NOTE: VLLM's speculative decoding config requires both draft and target
models to share the same vocab size. This script pads the smaller vocab
model to the larger vocab size after verifying token alignment.
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pad a smaller model's vocabulary to match a larger model's vocabulary"
    )
    parser.add_argument(
        "--model_a", type=str,
        default=DRAFT_NAME,
        help=f"Name or path of the first model (default: {DRAFT_NAME})"
    )
    parser.add_argument(
        "--model_b", type=str,
        default=TARGET_NAME,
        help=f"Name or path of the second model (default: {TARGET_NAME})"
    )
    parser.add_argument(
        "--suffix", type=str, default="-padded",
        help="Suffix to append to the output directory (default: -padded)"
    )
    return parser.parse_args()


def load_tokenizer(name):
    print(f"Loading tokenizer for '{name}'...")
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def safety_check(tok_a, tok_b):
    vocab_a = tok_a.get_vocab()
    vocab_b = tok_b.get_vocab()
    N = min(len(vocab_a), len(vocab_b))
    print(f"Running safety check: comparing first {N} tokens...")
    id_to_token_a = {idx: tok for tok, idx in vocab_a.items()}
    id_to_token_b = {idx: tok for tok, idx in vocab_b.items()}
    for i in range(N):
        if id_to_token_a[i] != id_to_token_b[i]:
            print(
                f"❌ Token mismatch at ID {i}: "
                f"A='{id_to_token_a[i]}' vs B='{id_to_token_b[i]}'"
            )
            sys.exit(1)
    print("✅ Safety check passed: shared tokens are aligned.")


def main():
    args = parse_args()

    # Load configs to get true vocab sizes
    print(f"Loading config for '{args.model_a}'...")
    config_a = AutoConfig.from_pretrained(args.model_a, trust_remote_code=True)
    print(f"Loading config for '{args.model_b}'...")
    config_b = AutoConfig.from_pretrained(args.model_b, trust_remote_code=True)
    size_a = config_a.vocab_size
    size_b = config_b.vocab_size
    print(f"Model A vocab size (from config): {size_a}")
    print(f"Model B vocab size (from config): {size_b}")

    # Load tokenizers and verify alignment
    tokenizer_a = load_tokenizer(args.model_a)
    tokenizer_b = load_tokenizer(args.model_b)
    safety_check(tokenizer_a, tokenizer_b)

    # Determine which model needs padding
    if size_a < size_b:
        name_small, name_large = args.model_a, args.model_b
        tok_small = tokenizer_a
        size_small, size_large = size_a, size_b
    elif size_b < size_a:
        name_small, name_large = args.model_b, args.model_a
        tok_small = tokenizer_b
        size_small, size_large = size_b, size_a
    else:
        print("🎉 Both vocab sizes match exactly. No padding needed.")
        return

    pad_size = size_large - size_small
    print(f"Padding model '{name_small}' from {size_small} to {size_large} tokens (+{pad_size})...")

    # Load and patch the smaller model
    print(f"Loading config and model for '{name_small}'...")
    config_small = AutoConfig.from_pretrained(name_small, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name_small, config=config_small, trust_remote_code=True
    )

    print("Resizing token embeddings to match larger vocab size...")
    model.config.vocab_size = size_large
    model.resize_token_embeddings(size_large)

    # Add padding tokens
    print("Adding padding tokens to tokenizer...")
    pad_tokens = [f"<extra_pad_{i}>" for i in range(pad_size)]
    tok_small.add_special_tokens({
        "additional_special_tokens": pad_tokens,
        "pad_token": pad_tokens[0]
    })
    print(f"New tokenizer size: {tok_small.vocab_size}")

    # Save padded model and tokenizer
    base_name = name_small.rstrip('/').split('/')[-1]
    output_dir = f"./{base_name}{args.suffix}"
    print(f"Saving padded model and tokenizer to '{output_dir}'...")
    model.save_pretrained(output_dir)
    tok_small.save_pretrained(output_dir)

    print("✅ Padding complete! Padded model available at:", output_dir)

if __name__ == "__main__":
    main()