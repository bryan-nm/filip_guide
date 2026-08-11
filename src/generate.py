"""Prompt-guided protein generation (Milestone 4 CLI).

    python -m src.generate --prompt "FUNCTION: ..." --ckpt <classifier.pt> \
        --length 100 --n 4 --gamma 1.0 --likelihood sigmoid

Decodes from the frozen ESM-C generator (the unconditional prior p(x)) by masked
iterative unmasking, reweighting each step by the AMPLIFY-FILIP classifier's
p(prompt | x)^gamma (DEG). The generator and classifier use different tokenizers;
a TokenizerBridge maps between them. `--gamma 0` is the unconditional baseline.

Reports the classifier's log p(prompt | sequence) for the generated samples so you
can see guidance moving generation toward the prompt (guided > unconditional).
"""
from __future__ import annotations

import argparse

import torch

from config import default_cfg
from src import encoders
from src.bridge import TokenizerBridge
from src.esmc_generative import ESMCGenerator
from src.filip_predictor import build_predictor_from_checkpoint
from src.guide import DEGGuidance
from src.sample import guided_generate


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def score_prompt_logprob(predictor, seqs, device) -> torch.Tensor:
    """Classifier log p(prompt | seq) for each finished (full) sequence."""
    enc = predictor.ptok(seqs, padding=True, truncation=True, max_length=1024,
                         return_tensors="pt")
    return predictor.log_prob(enc["input_ids"].to(device), enc["attention_mask"].to(device))


def main() -> None:
    cfg = default_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ckpt", required=True, help="trained FILIP classifier checkpoint")
    ap.add_argument("--length", type=int, default=100)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=cfg.guidance.gamma)
    ap.add_argument("--likelihood", default=cfg.guidance.likelihood_mode,
                    choices=["sigmoid", "softmax_bank"])
    ap.add_argument("--prompt-bank", default=cfg.guidance.prompt_bank_path)
    ap.add_argument("--temperature", type=float, default=cfg.guidance.sample_temperature)
    ap.add_argument("--order", default="ascending", choices=["ascending", "random"])
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--also-unconditional", action="store_true",
                    help="also generate a gamma=0 baseline for comparison")
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"[gen] device={device} gamma={args.gamma} likelihood={args.likelihood} "
          f"length={args.length} n={args.n}", flush=True)

    # Generator (ESM-C) and classifier (AMPLIFY-FILIP from the checkpoint).
    generator = ESMCGenerator.load(cfg.guidance.generator_path, device)
    predictor = build_predictor_from_checkpoint(args.ckpt, cfg, device, mode=args.likelihood)
    predictor.set_target(args.prompt, *_load_text(cfg, device), cfg.data.max_text_tokens)
    if args.likelihood == "softmax_bank":
        predictor.load_bank(args.prompt_bank)

    bridge = TokenizerBridge(generator.tok, predictor.ptok).to(device)

    def run(gamma):
        guidance = DEGGuidance(generator, predictor, bridge, gamma)
        seqs = guided_generate(generator_seed=args.seed, guidance=guidance, length=args.length,
                               n=args.n, temperature=args.temperature, order=args.order,
                               greedy=args.greedy)
        lp = score_prompt_logprob(predictor, seqs, device)
        return seqs, lp

    print(f"\n=== guided (gamma={args.gamma}) ===", flush=True)
    guided_seqs, guided_lp = run(args.gamma)
    for s, l in zip(guided_seqs, guided_lp.tolist()):
        print(f"  logp(prompt|seq)={l:+.3f}  {s}", flush=True)
    print(f"  mean logp(prompt|seq) = {guided_lp.mean().item():+.3f}", flush=True)

    if args.also_unconditional:
        print("\n=== unconditional (gamma=0) ===", flush=True)
        uncond_seqs, uncond_lp = run(0.0)
        for s, l in zip(uncond_seqs, uncond_lp.tolist()):
            print(f"  logp(prompt|seq)={l:+.3f}  {s}", flush=True)
        print(f"  mean logp(prompt|seq) = {uncond_lp.mean().item():+.3f}", flush=True)
        print(f"\n[gen] guidance shift in mean logp(prompt|seq): "
              f"{(guided_lp.mean() - uncond_lp.mean()).item():+.3f}", flush=True)


def _load_text(cfg, device):
    tmodel, ttok = encoders.load_text_encoder(cfg.model.text_encoder_path, device,
                                              cfg.data.caption_field_labels)
    return tmodel, ttok


if __name__ == "__main__":
    main()
