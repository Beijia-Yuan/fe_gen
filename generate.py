import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import EsmModel, AutoTokenizer
from helpers import load_seqs_and_fes, seqs_to_tokens, tokens_to_seq, random_train_val_split, validate_sequences
from encoder_decoder_model import fes_dataset, fes_encoder_decoder_model, train, collate_fn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")
idx_to_aa = {i: aa for i, aa in enumerate(AA_TOKENS)}
aa_to_idx = {aa: i for i, aa in idx_to_aa.items()}
SEP_TOKEN = 20
EOS_TOKEN = 21
PAD_TOKEN = 22
VOCAB_SIZE = 23

# load & split data
unilat_fes, unilat_seqs, aa_index_table, aa_index_table_reversed = load_seqs_and_fes(cluster=False)
_, _, fes_val, seqs_val = random_train_val_split(unilat_fes, unilat_seqs)

# validation set
fes_vectors_val = fes_val # torch.from_numpy(fes_val).float()
target_seqs_val = seqs_to_tokens(seqs_val, aa_index_table)

# train
model = fes_encoder_decoder_model(fes_dim=1326, n_tokens=1).to(device)
model.load_state_dict(torch.load('/home/jc4587/fe_data/esm2_fes_finetuned.pt'))

# Inference
model.eval()
# select a random sample for validation
randomstate = np.random.RandomState(0)
random_indices = randomstate.choice(fes_vectors_val.shape[0], size=32, replace=False)
test_fes = torch.from_numpy(fes_vectors_val[random_indices]).float().to(device)
print(test_fes.shape)
generated = model.generate(test_fes, temperature=1.0)
test_seq = torch.stack([generated[:, :20], generated[:, 21:41] ], dim=1)
# Validate generated sequences
validate_sequences(test_fes.detach().cpu().numpy(), test_seq.detach().cpu().numpy())
# random base line
test_seq_baseline = torch.randint(0, 20, (32, 2, 20))
validate_sequences(test_fes.detach().cpu().numpy(), test_seq_baseline.detach().cpu().numpy())

# Parse output
for i, seq in enumerate(generated):
    seq = seq.tolist()
    if SEP_TOKEN in seq:
        split = seq.index(SEP_TOKEN)
        seq1 = [idx_to_aa.get(t, '?') for t in seq[:split]]
        seq2 = [idx_to_aa.get(t, '?') for t in seq[split+1:] if t not in (EOS_TOKEN, PAD_TOKEN)]
    else:
        seq1 = [idx_to_aa.get(t, '?') for t in seq[:20]]
        seq2 = [idx_to_aa.get(t, '?') for t in seq[20:] if t not in (EOS_TOKEN, PAD_TOKEN)]
    print(f"Sample {i}: seq1={''.join(seq1)} | seq2={''.join(seq2)}")