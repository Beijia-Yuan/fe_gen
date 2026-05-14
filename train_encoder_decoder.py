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

# load & split data
unilat_fes, unilat_seqs, aa_index_table, aa_index_table_reversed = load_seqs_and_fes()
fes_train, seqs_train, fes_val, seqs_val = random_train_val_split(unilat_fes, unilat_seqs)

# train set
fes_vectors_train = torch.from_numpy(fes_train).float()
target_seqs_train = seqs_to_tokens(seqs_train, aa_index_table)
dataset_train = fes_dataset(fes_vectors_train, target_seqs_train)
dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

# validation set
fes_vectors_val = torch.from_numpy(fes_val).float()
target_seqs_val = seqs_to_tokens(seqs_val, aa_index_table)
dataset_val = fes_dataset(fes_vectors_val, target_seqs_val)
dataloader_val = DataLoader(
    dataset_val, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=collate_fn, 
    num_workers=4, 
    pin_memory=True
)

# train
model = fes_encoder_decoder_model(fes_dim=1326, n_tokens=1)
model = train(
    model, 
    dataloader_train, 
    dataloader_val, 
    fes_val, 
    n_epochs=100, 
    lr=1e-4, 
    device=device
)