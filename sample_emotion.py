"""
===============================================================================
QUANTARA NANOGPT - Emotion Sampling Script
===============================================================================
Generate emotion-aware text samples from trained model.

Usage:
  python sample_emotion.py --emotion joy --prompt "Today I feel"
  python sample_emotion.py --emotion sadness --num_samples 5
  python sample_emotion.py --interactive
===============================================================================
"""

import os
import pickle
import argparse
from contextlib import nullcontext

import torch
import tiktoken

# nanoGPT
from model import GPT, GPTConfig


def load_model(checkpoint_path, device):
    """Load trained emotion model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


def get_tokenizer(dataset_dir='data/quantara_emotion'):
    """Get appropriate tokenizer"""
    meta_path = os.path.join(dataset_dir, 'meta.pkl')

    if os.path.exists(meta_path):
        # Character-level
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # GPT-2 BPE
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return encode, decode


def generate(
    model,
    encode,
    decode,
    prompt,
    emotion=None,
    max_tokens=256,
    temperature=0.8,
    top_k=200,
    device='cpu'
):
    """Generate emotion-aware text"""
    # Format with emotion tag
    if emotion:
        formatted = f"<{emotion}>{prompt}"
    else:
        formatted = prompt

    # Encode
    start_ids = encode(formatted)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Generate
    with torch.no_grad():
        ctx = torch.amp.autocast(device_type=device, dtype=torch.float16) if device == 'cuda' else nullcontext()
        with ctx:
            y = model.generate(x, max_tokens, temperature=temperature, top_k=top_k)

    # Decode and clean
    output = decode(y[0].tolist())

    if emotion and f"</{emotion}>" in output:
        output = output.split(f"</{emotion}>")[0]
        output = output.replace(f"<{emotion}>", "").strip()

    return output


def main():
    parser = argparse.ArgumentParser(description='Quantara Emotion GPT Sampling')
    parser.add_argument('--checkpoint', default='out-quantara-emotion/ckpt.pt', help='Checkpoint path')
    parser.add_argument('--emotion', default=None, help='Target emotion (joy, sadness, anger, fear, love, surprise)')
    parser.add_argument('--prompt', default='', help='Starting prompt')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--max_tokens', type=int, default=256, help='Max tokens per sample')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling')
    parser.add_argument('--device', default='auto', help='Device (cuda, mps, cpu, auto)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    args = parser.parse_args()

    # Detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"\n[Quantara] Loading model on {device}...")

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"[!] Checkpoint not found: {args.checkpoint}")
        print(f"[!] Train the model first with:")
        print(f"    python data/quantara_emotion/prepare.py")
        print(f"    python train.py config/train_quantara_emotion_fast.py")
        return

    # Load model
    model = load_model(args.checkpoint, device)
    encode, decode = get_tokenizer()

    print(f"[Quantara] Model loaded. Emotion: {args.emotion or 'auto'}")

    if args.interactive:
        print("\n" + "=" * 60)
        print("  QUANTARA EMOTION GPT - Interactive Sampling")
        print("=" * 60)
        print("  Emotions: joy, sadness, anger, fear, love, surprise")
        print("  Type '/emotion <name>' to set emotion, '/quit' to exit")
        print("=" * 60 + "\n")

        current_emotion = args.emotion

        while True:
            try:
                user_input = input(f"[{current_emotion or 'auto'}] > ").strip()
            except KeyboardInterrupt:
                break

            if user_input.lower() == '/quit':
                break
            elif user_input.startswith('/emotion '):
                current_emotion = user_input.split(' ')[1]
                print(f"  [Emotion set to: {current_emotion}]")
                continue
            elif user_input == '/emotion':
                current_emotion = None
                print("  [Emotion cleared]")
                continue

            output = generate(
                model, encode, decode,
                prompt=user_input,
                emotion=current_emotion,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            print(f"\n{output}\n")

    else:
        # Generate samples
        print("-" * 60)

        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f"\n--- Sample {i+1} ---")

            output = generate(
                model, encode, decode,
                prompt=args.prompt,
                emotion=args.emotion,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            print(output)

        print("-" * 60)


if __name__ == "__main__":
    main()
