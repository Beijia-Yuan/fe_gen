import numpy as np
from tqdm import tqdm
import sys

def create_grids(max_bound, min_bound, l):
    z = max_bound.size
    assert z == min_bound.size
    d = max_bound - min_bound
    ranges = [np.arange(min(m+l/2, (m+M)/2), M, l) for m, M in zip(min_bound, max_bound)]
    mesh = np.meshgrid(*ranges, indexing="ij")
    grid = np.stack(mesh, axis=-1).reshape(-1, z)
    return grid

def nearest_l2(arr1, arr2):
    diff = arr1[:, None, :] - arr2[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    idx = np.argmin(dist2, axis=1)
    return idx


def grid_density(points, l):
    mab = np.max(points, axis = 0)
    mib = np.min(points, axis = 0)
    grids = create_grids(mab, mib, l) 
    counts = np.zeros(len(grids))
    partition = []
    for i in range(0, len(points), 1000): 
        idxs = nearest_l2(points[i:i+1000], grids)
        np.add.at(counts, idxs, 1)
        partition.append(idxs)
    partition = np.concatenate(partition)
    p = counts/len(points)
    mask = (p>= 5e-5)
    assert np.round(p.sum(), 10) == 1
    return grids, p, partition, mask

if __name__ == "__main__":
    i = sys.argv[1]
    train_idx = np.load("/home/by7175/scratch/proteome/eosdata/random_data/train_idx.npy")
    ztilde = np.load("/home/by7175/scratch/transformer/HPferound_r12/zdim10/Qs/aligned/ztilde_new.npy").mean(0)

    train_ztilde = ztilde[train_idx]
    grids, p, partitions, mask = grid_density(train_ztilde, l=1)

    print(grids.shape, p.shape, partitions.shape, mask.shape)

    idxs = np.arange(len(grids))[mask]
    masked_grids = grids[mask]
    masked_idxs = idxs
    valid_masked_indices = {}
    
    for j, part in enumerate(masked_idxs):
        if part not in valid_masked_indices:
            q = np.flatnonzero(partitions == part)
            valid_masked_indices[part] = q
    
    pairs = []
    n = len(masked_grids)
    
    for j in tqdm(range(n)):
        possibles = valid_masked_indices[masked_idxs[j]]
        for k in range(j, n):
            possibles2 = valid_masked_indices[masked_idxs[k]]
            chosen = np.random.choice(possibles)
            chosen2 = np.random.choice(possibles2)
            pairs.append((chosen, chosen2))
            chosen = np.random.choice(possibles)
            chosen2 = np.random.choice(possibles2)
            pairs.append((chosen, chosen2))

    pairs = np.array(pairs)
    print(pairs.shape)
    np.save(f"/home/by7175/bscratch/unilat_random_pairs_idx{i}.npy", pairs)
