# evaluate_conditional_generation.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

from helpers import (
    load_seqs_and_fes,
    seqs_to_tokens,
    random_train_val_split,
)

from unconditional_model import (
    decoder_model,
    idx_to_aa,
    aa_to_idx,
    SEP_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load data
# --------------------------------------------------

unilat_fes, unilat_seqs, aa_index_table, aa_index_table_reversed = load_seqs_and_fes(cluster=False)

_, seqs_train, _, seqs_val = random_train_val_split(
    unilat_fes,
    unilat_seqs
)

# tokenized validation set
target_seqs_val = seqs_to_tokens(seqs_val, aa_index_table)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = decoder_model(
    d_model=320,
    nhead=8,
    num_layers=4,
)

model.load_state_dict(
    torch.load(
        "unconditional_decoder.pt",
        map_location=device
    )
)

model.to(device)
model.eval()

# --------------------------------------------------
# Generate seq2 conditioned on seq1
# --------------------------------------------------

generated_seq2s = []
true_seq2s = []

MAX_NEW_TOKENS = 24

with torch.no_grad():

    for pair in tqdm(target_seqs_val[:1000]):

        seq1 = pair[0]
        true_seq2 = pair[1]

        # ------------------------------------------
        # Prompt:
        # BOS + seq1 + SEP
        # ------------------------------------------

        prompt = torch.tensor(
            [EOS_TOKEN] + seq1.tolist() + [SEP_TOKEN],
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        generated = prompt.clone()

        # ------------------------------------------
        # autoregressive generation
        # ------------------------------------------

        for _ in range(MAX_NEW_TOKENS):

            logits = model.decoder(generated)

            next_logits = logits[:, -1, :] / 0.6

            probs = torch.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

            if next_token.item() == EOS_TOKEN:
                break

        # ------------------------------------------
        # decode generated seq2
        # ------------------------------------------

        gen_tokens = generated[0].tolist()

        # everything after SEP
        split = gen_tokens.index(SEP_TOKEN)

        seq2_tokens = []

        for t in gen_tokens[split + 1:]:

            if t in (EOS_TOKEN, PAD_TOKEN):
                break

            if t < 20:
                seq2_tokens.append(t)

        gen_seq2 = ''.join(
            idx_to_aa[t]
            for t in seq2_tokens
        )

        true_seq2_str = ''.join(
            idx_to_aa[t]
            for t in true_seq2.tolist()
        )

        generated_seq2s.append(gen_seq2)
        true_seq2s.append(true_seq2_str)

# --------------------------------------------------
# Plot AA distribution
# --------------------------------------------------

def aa_distribution(seqs):

    aa_string = ''.join(seqs)

    counts = Counter(aa_string)

    aas = sorted(aa_to_idx.keys())

    freqs = np.array([
        counts.get(aa, 0)
        for aa in aas
    ])

    freqs = freqs / freqs.sum()

    return aas, freqs

aas, gen_freqs = aa_distribution(generated_seq2s)
_, true_freqs = aa_distribution(true_seq2s)

# --------------------------------------------------
# Plot comparison
# --------------------------------------------------

x = np.arange(len(aas))
width = 0.4

plt.figure(figsize=(10, 3))

plt.bar(
    x - width/2,
    true_freqs,
    width,
    label="Real",
    color='darkgray',
    edgecolor='gray',
    linewidth=2
)

plt.bar(
    x + width/2,
    gen_freqs,
    width,
    label="Generated",
    color='cornflowerblue',
    edgecolor='royalblue',
    linewidth=2
)

plt.xticks(x, aas)

plt.xlabel("Amino Acid")
plt.ylabel("Frequency")

# plt.title("AA Distribution: True vs Generated seq2")

plt.legend()

plt.tight_layout()

plt.savefig(
    "conditional_generation_distribution.png",
    dpi=400
)

plt.show()

# --------------------------------------------------
# Print some examples
# --------------------------------------------------

print("\nExamples:\n")

for i in range(10):

    print(f"TRUE : {true_seq2s[i]}")
    print(f"GEN  : {generated_seq2s[i]}")
    print()
