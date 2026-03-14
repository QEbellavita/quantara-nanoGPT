"""
===============================================================================
QUANTARA NANOGPT - Emotion Sampling Script (32-Emotion Taxonomy)
===============================================================================
Generate emotion-aware text samples from trained model.
Supports 32 emotions across 9 families with hierarchical classification.

Usage:
  python sample_emotion.py --emotion joy --prompt "Today I feel"
  python sample_emotion.py --emotion anxiety --num_samples 5
  python sample_emotion.py --interactive
  python sample_emotion.py --list-emotions
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


# ─── 32-Emotion Taxonomy ────────────────────────────────────────────────────

EMOTION_FAMILIES = {
    'Joy': ['joy', 'excitement', 'enthusiasm', 'fun', 'gratitude', 'pride'],
    'Sadness': ['sadness', 'grief', 'boredom', 'nostalgia'],
    'Anger': ['anger', 'frustration', 'hate', 'contempt', 'disgust', 'jealousy'],
    'Fear': ['fear', 'anxiety', 'worry', 'overwhelmed', 'stressed'],
    'Love': ['love', 'compassion'],
    'Calm': ['calm', 'relief', 'mindfulness', 'resilience', 'hope'],
    'Self-Conscious': ['guilt', 'shame'],
    'Surprise': ['surprise'],
    'Neutral': ['neutral'],
}

ALL_EMOTIONS = [e for emotions in EMOTION_FAMILIES.values() for e in emotions]


def print_emotion_list():
    """Print all 32 emotions grouped by family."""
    print("\n  32 Emotions (9 Families):")
    print("  " + "-" * 56)
    for family, emotions in EMOTION_FAMILIES.items():
        print(f"  {family:16s}: {', '.join(emotions)}")
    print("  " + "-" * 56)
    print(f"  Total: {len(ALL_EMOTIONS)} emotions\n")


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
    parser = argparse.ArgumentParser(description='Quantara Emotion GPT Sampling (32 Emotions)')
    parser.add_argument('--checkpoint', default='out-quantara-emotion/ckpt.pt', help='Checkpoint path')
    parser.add_argument('--emotion', default=None, help='Target emotion (see --list-emotions for all 32)')
    parser.add_argument('--prompt', default='', help='Starting prompt')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--max_tokens', type=int, default=256, help='Max tokens per sample')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling')
    parser.add_argument('--device', default='auto', help='Device (cuda, mps, cpu, auto)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--list-emotions', action='store_true', help='List all 32 emotions and exit')
    args = parser.parse_args()

    # List emotions and exit
    if args.list_emotions:
        print_emotion_list()
        return

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
        print("  QUANTARA EMOTION GPT - Interactive Sampling (32 Emotions)")
        print("=" * 60)
        print_emotion_list()
        print("  Commands:")
        print("    /emotion <name>  — set target emotion")
        print("    /emotion         — clear emotion (auto)")
        print("    /family <name>   — list emotions in a family")
        print("    /list            — show all emotions")
        print("    /quit            — exit")
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
                em = user_input.split(' ', 1)[1].strip().lower()
                if em in ALL_EMOTIONS:
                    current_emotion = em
                    family = None
                    for fam, ems in EMOTION_FAMILIES.items():
                        if em in ems:
                            family = fam
                            break
                    print(f"  [Emotion set to: {current_emotion} ({family} family)]")
                else:
                    print(f"  [Unknown emotion: {em}. Use /list to see all.]")
                continue
            elif user_input == '/emotion':
                current_emotion = None
                print("  [Emotion cleared]")
                continue
            elif user_input.startswith('/family '):
                fam_name = user_input.split(' ', 1)[1].strip()
                found = False
                for family, emotions in EMOTION_FAMILIES.items():
                    if family.lower() == fam_name.lower():
                        print(f"  {family}: {', '.join(emotions)}")
                        found = True
                        break
                if not found:
                    print(f"  [Unknown family: {fam_name}. Families: {', '.join(EMOTION_FAMILIES.keys())}]")
                continue
            elif user_input == '/list':
                print_emotion_list()
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
