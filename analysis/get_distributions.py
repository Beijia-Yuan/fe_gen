import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader
from helpers import load_seqs_and_fes, seqs_to_tokens, random_train_val_split
from unconditional_model import seq_dataset, decoder_model, train, collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"
unilat_fes, unilat_seqs, aa_index_table, aa_index_table_reversed = load_seqs_and_fes(cluster=False)
_, seqs_train, _, seqs_val = random_train_val_split(unilat_fes, unilat_seqs)
print(seqs_val)

# Validation set distribution
all_seqs = seqs_train.flatten()
all_aas = ''.join(all_seqs)
aa_counts = Counter(all_aas)
aas = sorted(aa_counts.keys())
counts = [aa_counts[aa] for aa in aas]
freqs = np.array(counts) / np.sum(counts)
plt.figure(figsize=(10, 3))
plt.bar(aas, freqs)
plt.xlabel('Amino Acid')
plt.ylabel('Frequency')
plt.title('Amino Acid Distribution')
plt.tight_layout()
plt.savefig('valset_aa_distribution.png', dpi=400)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import Counter
# df = pd.read_csv("./results/generated_sequences.txt", delim_whitespace=True)
# all_seqs = (
#     df["chain1"].tolist() +
#     df["chain2"].tolist()
# )
# print(f"Number of sequences: {len(all_seqs)}")
# all_aas = ''.join(all_seqs)
# aa_counts = Counter(all_aas)
# aas = sorted(aa_counts.keys())
# counts = np.array([aa_counts[aa] for aa in aas])
# freqs = counts / counts.sum()
# plt.figure(figsize=(10, 3))
# plt.bar(aas, freqs)
# plt.xlabel("Amino Acid")
# plt.ylabel("Frequency")
# plt.title("Generated Sequence Amino Acid Distribution")
# plt.tight_layout()
# plt.savefig(
#     "generated_aa_distribution.png",
#     dpi=400
# )
# plt.show()

# Uncond generated seq distribution
log_file = "train_unconditional.out"
with open(log_file, "r") as f:
    text = f.read()
epoch_to_seqs = {}
epoch_chunks = re.split(r"Epoch (\d+)/100", text)
for i in range(1, len(epoch_chunks), 2):
    epoch = int(epoch_chunks[i])
    chunk = epoch_chunks[i + 1]
    matches = re.findall(
        r"Sample \d+: seq1=([A-Z]+) \| seq2=([A-Z]+)",
        chunk
    )
    seqs = []
    for s1, s2 in matches:
        seqs.append(s1)
        seqs.append(s2)
    epoch_to_seqs[epoch] = seqs

epochs_to_plot = [15, 20, 25, 30] # [25, 50, 75, 100]
fig, axes = plt.subplots(2, 2, figsize=(10, 3))
axes = axes.flatten()
for ax, epoch in zip(axes, epochs_to_plot):
    if epoch not in epoch_to_seqs:
        ax.set_title(f"Epoch {epoch} (missing)")
        continue
    seqs = epoch_to_seqs[epoch]
    aa_string = ''.join(seqs)
    aa_counts = Counter(aa_string)
    aas = sorted(aa_counts.keys())
    counts = np.array([aa_counts[a] for a in aas])
    freqs = counts / counts.sum()
    ax.bar(aas, freqs)
    ax.set_title(f"Epoch {epoch}")
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig('generated_aa_distribution.png', dpi=400)
plt.show()

# # Cond generated seq distribution
# log_file = "train_encoder_decoder_rand_testset.out"
# with open(log_file, "r") as f:
#     text = f.read()
# epoch_to_seqs = {}
# epoch_chunks = re.split(r"Epoch (\d+)/100", text)
# for i in range(1, len(epoch_chunks), 2):
#     epoch = int(epoch_chunks[i])
#     chunk = epoch_chunks[i + 1]
#     matches = re.findall(
#         r"Sample \d+: seq1=([A-Z]+) \| seq2=([A-Z]+)",
#         chunk
#     )
#     seqs = []
#     for s1, s2 in matches:
#         seqs.append(s1)
#         seqs.append(s2)
#     epoch_to_seqs[epoch] = seqs

# epochs_to_plot = [25, 50, 75, 100]
# fig, axes = plt.subplots(2, 2, figsize=(10, 3))
# axes = axes.flatten()
# for ax, epoch in zip(axes, epochs_to_plot):
#     if epoch not in epoch_to_seqs:
#         ax.set_title(f"Epoch {epoch} (missing)")
#         continue
#     seqs = epoch_to_seqs[epoch]
#     aa_string = ''.join(seqs)
#     aa_counts = Counter(aa_string)
#     aas = sorted(aa_counts.keys())
#     counts = np.array([aa_counts[a] for a in aas])
#     freqs = counts / counts.sum()
#     ax.bar(aas, freqs)
#     ax.set_title(f"Epoch {epoch}")
#     ax.set_xlabel("Amino Acid")
#     ax.set_ylabel("Frequency")

# plt.tight_layout()
# plt.savefig('cond_genset_dist.png', dpi=400)
# plt.show()