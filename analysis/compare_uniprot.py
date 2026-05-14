"""
evaluate_vs_uniprot.py

Compares two sources of sequence pairs on how well their predicted FES matches
the ground-truth FES vectors from the validation set:

  1. Model-generated sequences  -- conditioned on val-set FES vectors
  2. Random-chop baseline       -- pairs of 20-aa windows sampled from UniProt

For each source we compute:
  MSE( predict_fes(seq_pair), ground_truth_fes )

Usage:
  python evaluate_vs_uniprot.py \
      --checkpoint esm2_fes_finetuned.pt \
      --uniprot    uniprot.fasta \
      [--n_samples  256]
      [--temperature 1.0]
      [--device     cuda]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── project imports ────────────────────────────────────────────────────────────
from helpers import (
    load_seqs_and_fes,
    seqs_to_tokens,
    tokens_to_seq,
    random_train_val_split,
    validate_sequences,
)
from encoder_decoder_model import (
    AA_TOKENS,
    SEP_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    idx_to_aa,
    aa_to_idx,
    fes_encoder_decoder_model,
)

SEQ_LEN = 20  # length of each half of the pair


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FES PREDICTION  –– replace this with your actual forward model
# ══════════════════════════════════════════════════════════════════════════════

def predict_fes(seq_pairs: np.ndarray, device: str = "cpu") -> np.ndarray:
    return validate_sequences(fes_groundtruth, sequences)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  UniProt FASTA reader & random-chop sampler
# ══════════════════════════════════════════════════════════════════════════════

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: str) -> list[str]:
    """Return list of upper-cased protein sequences from a FASTA file."""
    sequences, current = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line.upper())
        if current:
            sequences.append("".join(current))
    # Keep only sequences long enough to yield two non-overlapping windows
    min_len = SEQ_LEN * 2 + 1
    kept = [s for s in sequences if len(s) >= min_len and all(c in VALID_AA for c in s)]
    print(f"[UniProt] {len(sequences):,} sequences read → {len(kept):,} usable (len≥{min_len}, standard AA)")
    return kept


def random_chop_pairs(sequences: list[str], n: int, rng: random.Random) -> np.ndarray:
    """
    Sample n pairs of non-overlapping 20-aa windows from random UniProt sequences.
    Returns int array  (n, 2, 20)  using the aa_to_idx encoding.
    """
    pairs = np.zeros((n, 2, SEQ_LEN), dtype=np.int64)
    for i in range(n):
        seq = rng.choice(sequences)
        max_start = len(seq) - 2 * SEQ_LEN
        start1 = rng.randint(0, max_start)
        start2 = rng.randint(start1 + SEQ_LEN, len(seq) - SEQ_LEN)
        pairs[i, 0] = [aa_to_idx[c] for c in seq[start1: start1 + SEQ_LEN]]
        pairs[i, 1] = [aa_to_idx[c] for c in seq[start2: start2 + SEQ_LEN]]
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Decode model output → integer seq pairs
# ══════════════════════════════════════════════════════════════════════════════

def decode_generated(generated: torch.Tensor) -> np.ndarray:
    """
    Convert raw generated token sequences (B, T) into int arrays (B, 2, 20).
    Tokens after EOS are ignored; SEP splits the two chains.
    Chains shorter than SEQ_LEN are right-padded with alanine (index 0).
    """
    B = generated.shape[0]
    pairs = np.zeros((B, 2, SEQ_LEN), dtype=np.int64)
    for i, seq in enumerate(generated.tolist()):
        # Truncate at first EOS
        if EOS_TOKEN in seq:
            seq = seq[: seq.index(EOS_TOKEN)]
        if SEP_TOKEN in seq:
            split = seq.index(SEP_TOKEN)
            chain1 = [t for t in seq[:split]          if t < len(AA_TOKENS)]
            chain2 = [t for t in seq[split + 1:]      if t < len(AA_TOKENS)]
        else:
            chain1 = [t for t in seq[:SEQ_LEN]        if t < len(AA_TOKENS)]
            chain2 = [t for t in seq[SEQ_LEN:]        if t < len(AA_TOKENS)]

        for j, chain in enumerate([chain1, chain2]):
            trimmed = chain[:SEQ_LEN]
            pairs[i, j, : len(trimmed)] = trimmed
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    device = args.device
    rng = random.Random(args.seed)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(args.seed)

    # ── load validation FES ────────────────────────────────────────────────
    print("Loading data …")
    unilat_fes, unilat_seqs, aa_index_table, _ = load_seqs_and_fes(cluster=False)
    _, _, fes_val, seqs_val = random_train_val_split(unilat_fes, unilat_seqs)

    n = min(args.n_samples, len(fes_val))
    idx = np.random.default_rng(args.seed).choice(len(fes_val), size=n, replace=False)
    gt_fes = fes_val[idx].astype(np.float32)           # (n, 1326) ground truth
    print(f"Using {n} validation samples.  FES shape: {gt_fes.shape}")

    # ── load model ────────────────────────────────────────────────────────
    print(f"Loading model from {args.checkpoint} …")
    model = fes_encoder_decoder_model(fes_dim=gt_fes.shape[1], n_tokens=1)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # ── generate sequences conditioned on val FES ─────────────────────────
    print("Generating sequences from model …")
    batch_size = args.batch_size
    all_generated_pairs = []

    # ── sample random UniProt chops ───────────────────────────────────────
    print(f"Reading UniProt FASTA from {args.uniprot} …")
    uniprot_seqs = read_fasta(args.uniprot)
    if not uniprot_seqs:
        sys.exit("No usable sequences found in the FASTA file.")

    for start in tqdm(range(0, n, batch_size)):
        batch_fes = torch.from_numpy(gt_fes[start: start + batch_size]).to(device)
        with torch.no_grad():
            gen = model.generate(batch_fes, max_len=42, temperature=args.temperature)

        test_seq = torch.stack([gen[:, :20], gen[:, 21:41] ], dim=1)
        # Validate generated sequences
        validate_sequences(batch_fes.detach().cpu().numpy(), test_seq.detach().cpu().numpy())
        
        # uniprot baseline
        uniprot_pairs = random_chop_pairs(uniprot_seqs, n=test_seq.shape[0], rng=rng)          # (n, 2, 20)
        baseline_seq = uniprot_pairs # torch.stack([uniprot_pairs[:, :20], uniprot_pairs[:, 21:41] ], dim=1)
        validate_sequences(batch_fes.detach().cpu().numpy(), baseline_seq)

        # random baseline
        test_seq_baseline = torch.randint(0, 20, (32, 2, 20))
        validate_sequences(batch_fes.detach().cpu().numpy(), test_seq_baseline.detach().cpu().numpy())

        all_generated_pairs.append(decode_generated(gen.cpu()))

    # generated_pairs = np.concatenate(all_generated_pairs, axis=0)   # (n, 2, 20)
    # print(f"Generated pairs shape: {generated_pairs.shape}")

    # # ── predict FES for both sets ─────────────────────────────────────────
    # print("Predicting FES for generated sequences …")
    # pred_fes_generated = predict_fes(generated_pairs, device=device) # (n, 1326)

    # print("Predicting FES for UniProt random chops …")
    # pred_fes_uniprot = predict_fes(uniprot_pairs, device=device)      # (n, 1326)

    # # ── compute per-sample MSE then mean ──────────────────────────────────
    # def mse(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    #     """Return (mean MSE, std MSE) over the sample axis."""
    #     per_sample = ((pred - target) ** 2).mean(axis=1)   # (n,)
    #     return per_sample.mean(), per_sample.std()

    # mse_gen_mean,  mse_gen_std  = mse(pred_fes_generated, gt_fes)
    # mse_uni_mean,  mse_uni_std  = mse(pred_fes_uniprot,   gt_fes)

    # # ── also run validate_sequences if it gives extra signal ─────────────
    # print("\n── validate_sequences (model-generated) ──────────────────────")
    # validate_sequences(gt_fes, generated_pairs)

    # print("\n── validate_sequences (UniProt random chops) ─────────────────")
    # validate_sequences(gt_fes, uniprot_pairs)

    # # ── report ────────────────────────────────────────────────────────────
    # width = 54
    # print("\n" + "═" * width)
    # print(" FES reconstruction MSE  (lower = better)")
    # print("═" * width)
    # print(f"  {'Source':<30} {'MSE (mean ± std)':>20}")
    # print("─" * width)
    # print(f"  {'Model-generated (conditioned)':<30} {mse_gen_mean:10.4f} ± {mse_gen_std:.4f}")
    # print(f"  {'UniProt random chops':<30} {mse_uni_mean:10.4f} ± {mse_uni_std:.4f}")
    # print("─" * width)
    # delta = mse_uni_mean - mse_gen_mean
    # pct   = 100 * delta / (mse_uni_mean + 1e-12)
    # if delta > 0:
    #     print(f"  Model-generated sequences are {pct:.1f}% better (lower MSE)")
    # elif delta < 0:
    #     print(f"  Random UniProt chops are {-pct:.1f}% better (lower MSE)")
    # else:
    #     print("  Tie.")
    # print("═" * width)

    # # ── save detailed results ─────────────────────────────────────────────
    # out = {
    #     "gt_fes":               gt_fes,
    #     "generated_pairs":      generated_pairs,
    #     "uniprot_pairs":        uniprot_pairs,
    #     "pred_fes_generated":   pred_fes_generated,
    #     "pred_fes_uniprot":     pred_fes_uniprot,
    #     "mse_generated":        mse_gen_mean,
    #     "mse_uniprot":          mse_uni_mean,
    # }
    # np.savez("evaluation_results.npz", **out)
    # print("\nDetailed results saved to evaluation_results.npz")


# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",   default="esm2_fes_finetuned.pt",  help="Path to saved model weights")
    p.add_argument("--uniprot",      required=True,                     help="Path to UniProt FASTA file")
    p.add_argument("--n_samples",    type=int,   default=256,           help="Number of val samples to evaluate")
    p.add_argument("--batch_size",   type=int,   default=32,            help="Generation batch size")
    p.add_argument("--temperature",  type=float, default=1.0,           help="Sampling temperature")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())