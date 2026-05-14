# eval_uncond_temperature_sweep.py

import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

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

# --------------------------------------------------
# config
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

TEMPERATURES = np.linspace(0.0, 1.0, 11)

N_SAMPLES = 1000
MAX_NEW_TOKENS = 24

# --------------------------------------------------
# load data
# --------------------------------------------------

unilat_fes, unilat_seqs, aa_index_table, aa_index_table_reversed = load_seqs_and_fes(cluster=False)

_, seqs_train, _, seqs_val = random_train_val_split(
    unilat_fes,
    unilat_seqs
)

target_seqs_val = seqs_to_tokens(
    seqs_val,
    aa_index_table
)

# --------------------------------------------------
# validation distribution
# --------------------------------------------------

true_seq2s = []

for pair in target_seqs_val[:N_SAMPLES]:

    seq2 = pair[1]

    seq2_str = ''.join(
        idx_to_aa[t]
        for t in seq2.tolist()
    )

    true_seq2s.append(seq2_str)

# --------------------------------------------------
# load model
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
# helpers
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


def distribution_similarity(freqs1, freqs2):

    jsd = jensenshannon(freqs1, freqs2)

    return 1 - jsd


# --------------------------------------------------
# true distribution
# --------------------------------------------------

aas, true_freqs = aa_distribution(true_seq2s)

# --------------------------------------------------
# temperature sweep
# --------------------------------------------------

all_gen_freqs = []
all_similarities = []

for temperature in TEMPERATURES:

    print(f"\nTemperature = {temperature:.2f}")

    generated_seq2s = []

    with torch.no_grad():

        for pair in tqdm(target_seqs_val[:N_SAMPLES]):

            seq1 = pair[0]

            prompt = torch.tensor(
                [EOS_TOKEN] + seq1.tolist() + [SEP_TOKEN],
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

            generated = prompt.clone()

            for _ in range(MAX_NEW_TOKENS):

                logits = model.decoder(generated)

                # avoid divide-by-zero
                if temperature == 0:
                    next_token = logits[:, -1, :].argmax(
                        dim=-1,
                        keepdim=True
                    )

                else:

                    next_logits = logits[:, -1, :] / temperature

                    probs = torch.softmax(
                        next_logits,
                        dim=-1
                    )

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

            # decode seq2
            gen_tokens = generated[0].tolist()

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

            generated_seq2s.append(gen_seq2)

    # AA distribution
    _, gen_freqs = aa_distribution(
        generated_seq2s
    )

    similarity = distribution_similarity(
        gen_freqs,
        true_freqs
    )

    print(f"Similarity: {similarity:.4f}")

    all_gen_freqs.append(gen_freqs)
    all_similarities.append(similarity)

# --------------------------------------------------
# BIG PLOT
# --------------------------------------------------

nrows = 3
ncols = 4

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(18, 10)
)

axes = axes.flatten()

for i, temperature in enumerate(TEMPERATURES):

    ax = axes[i]

    gen_freqs = all_gen_freqs[i]

    x = np.arange(len(aas))

    width = 0.4

    ax.bar(
        x - width/2,
        true_freqs,
        width,
        label="Real",
        color='darkgray',
        edgecolor='gray',
        linewidth=1
    )

    ax.bar(
        x + width/2,
        gen_freqs,
        width,
        label="Generated",
        color='cornflowerblue',
        edgecolor='royalblue',
        linewidth=1
    )

    ax.set_xticks(x)
    ax.set_xticklabels(aas)

    ax.set_ylim(0, max(
        true_freqs.max(),
        gen_freqs.max()
    ) * 1.15)

    similarity = all_similarities[i]

    ax.set_title(
        f"T={temperature:.1f}\n"
        f"Similarity={similarity:.3f}"
    )

# remove empty subplot
for j in range(len(TEMPERATURES), len(axes)):
    fig.delaxes(axes[j])

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=2,
    fontsize=12
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    "temperature_sweep_distributions.png",
    dpi=400
)

plt.show()

# --------------------------------------------------
# similarity curve
# --------------------------------------------------

plt.figure(figsize=(6, 4))

plt.plot(
    TEMPERATURES,
    all_similarities,
    marker='o',
    linewidth=2
)

plt.xlabel("Temperature")
plt.ylabel("1 - Jensen-Shannon Distance")
plt.title("Distribution Similarity vs Temperature")

plt.ylim(0, 1)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "temperature_sweep_similarity_curve.png",
    dpi=400
)

plt.show()
