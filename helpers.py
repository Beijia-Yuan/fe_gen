import torch
import pickle
import subprocess
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import EsmModel, AutoTokenizer
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def maxmin_subsample(X, n_subsamples):
    # Start with a random point or the first point
    selected_indices = [0]
    remaining_indices = list(range(1, len(X)))
    # Track the distance from each point to its nearest selected neighbor
    # (Initially infinity)
    min_distances = cdist(X[selected_indices], X[remaining_indices]).min(axis=0)
    for _ in tqdm(range(1, n_subsamples)):
        # Pick the point that is furthest from its nearest neighbor
        farthest_point_idx = np.argmax(min_distances)
        new_idx = remaining_indices.pop(farthest_point_idx)
        selected_indices.append(new_idx)
        if not remaining_indices:
            break   
        # Update min_distances: only need to check distance to the NEWLY added point
        new_distances = cdist(X[new_idx:new_idx+1], X[remaining_indices]).ravel()
        min_distances = np.delete(min_distances, farthest_point_idx)
        min_distances = np.minimum(min_distances, new_distances)
    return selected_indices

# Laods & returns unilat_fes and unilat_seqs
def load_seqs_and_fes(cluster=True):
    if cluster: print('Loading & clustering data')
    
    unilat_fes = np.load('/scratch/gpfs/ZHONGE/jc4587/Old/unilat_fes_new.npy')
    unilat_seqs = np.load('/scratch/gpfs/ZHONGE/jc4587/Old/unilat_seqs_new.npy')

    if cluster:
        fes_scaled = StandardScaler().fit_transform(unilat_fes)
        pca = PCA(n_components=50)
        fes_reduced = pca.fit_transform(fes_scaled)
        idxs = maxmin_subsample(fes_reduced, 20_000)

        unilat_fes = unilat_fes[idxs]
        unilat_seqs = unilat_seqs[idxs]

    with open('aa_index_table.pkl', 'rb') as file:
        aa_index_table = pickle.load(file)
    # dictionary where keys are numbers and values are letters
    aa_index_table_reversed = {value: key for key, value in aa_index_table.items()}
    return unilat_fes, unilat_seqs, aa_index_table, aa_index_table_reversed

# Convert (N, 2) to (N, 2, 20)
def seqs_to_tokens(seqs, aa_index_table):
  tokens = torch.zeros((seqs.shape[0], 2, 20), dtype=torch.int)
  for n in range(seqs.shape[0]):
    for i in range(20):
      tokens[n, 0, i] = aa_index_table[seqs[n, 0][i]]
      tokens[n, 1, i] = aa_index_table[seqs[n, 1][i]]
  return tokens

# Convert (N, 2) to (N, 2, 20)
def tokens_to_seq(tokens, aa_index_table_reversed):
    seqs = np.empty((tokens.shape[0], 2, 20), dtype='U1')
    for n in range(tokens.shape[0]):
        for i in range(20):
            seqs[n, 0, i] = aa_index_table_reversed[tokens[n, 0][i].item()]
            seqs[n, 1, i] = aa_index_table_reversed[tokens[n, 1][i].item()]
    return seqs

def random_train_val_split(unilat_fes, unilat_seqs, seed=0):
    idx = np.random.default_rng(seed).permutation(len(unilat_fes))
    split = int(len(unilat_fes) * (1 - 0.2))
    train_idx, val_idx = idx[:split], idx[split:]
    return unilat_fes[train_idx], unilat_seqs[train_idx], unilat_fes[val_idx], unilat_seqs[val_idx]

# Sloppy temp way to check predicted fes of generated sequences
def validate_sequences(fes_groundtruth, sequences):
    np.save("test_seqs.npy", sequences)
    _ = subprocess.run(
        ["python", "generate_fe_landscapes.py", "test_seqs.npy", "test_fes.npy"],
        capture_output=True,
        text=True
    )
    fes_predicted = np.load("test_fes.npy").mean(axis=0)
    print(f'Avg. free energy difference: {np.mean((fes_predicted - fes_groundtruth)**2)}')
