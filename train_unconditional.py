# train.py
import torch
from torch.utils.data import DataLoader
from helpers import (
    load_seqs_and_fes,
    seqs_to_tokens,
    random_train_val_split,
)
from unconditional_model import seq_dataset, decoder_model, train, collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

# FIX: capture unilat_fes instead of discarding it with _; it is needed
# by random_train_val_split below
unilat_fes, unilat_seqs, aa_index_table, aa_index_table_reversed = load_seqs_and_fes(cluster=False)

# FIX: capture all four return values; discard fes splits (unconditional model)
_, seqs_train, _, seqs_val = random_train_val_split(unilat_fes, unilat_seqs)

# --- train set ---
target_seqs_train = seqs_to_tokens(seqs_train, aa_index_table)
# FIX: seq_dataset no longer takes fes_vectors as first arg
dataset_train = seq_dataset(target_seqs_train)
dataloader_train = DataLoader(
    dataset_train,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

# --- validation set ---
target_seqs_val = seqs_to_tokens(seqs_val, aa_index_table)
dataset_val = seq_dataset(target_seqs_val)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

# --- train ---
model = decoder_model(d_model=320, nhead=8, num_layers=4)
# FIX: removed undefined `fes_val` argument; train() no longer takes fes
model = train(
    model,
    dataloader_train,
    dataloader_val,
    n_epochs=100,
    lr=1e-4,
    device=device,
)

# import torch
# import pickle
# import numpy as np
# from tqdm import tqdm
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import EsmModel, AutoTokenizer
# from helpers import load_seqs_and_fes, seqs_to_tokens, tokens_to_seq, random_train_val_split, validate_sequences
# from unconditional_model import seq_dataset, decoder_model, train, collate_fn
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # load & split data
# _, unilat_seqs, aa_index_table, aa_index_table_reversed = load_seqs_and_fes()
# _, seqs_train, _, seqs_val = random_train_val_split(unilat_fes, unilat_seqs)

# # train set
# target_seqs_train = seqs_to_tokens(seqs_train, aa_index_table)
# dataset_train = seq_dataset(target_seqs_train)
# dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

# # validation set
# target_seqs_val = seqs_to_tokens(seqs_val, aa_index_table)
# dataset_val = seq_dataset(target_seqs_val)
# dataloader_val = DataLoader(
#     dataset_val, 
#     batch_size=16, 
#     shuffle=True, 
#     collate_fn=collate_fn, 
#     num_workers=4, 
#     pin_memory=True
# )

# # train
# model = decoder_model(n_tokens=1)
# model = train(
#     model, 
#     dataloader_train, 
#     dataloader_val, 
#     fes_val, 
#     n_epochs=100, 
#     lr=1e-4, 
#     device=device
# )