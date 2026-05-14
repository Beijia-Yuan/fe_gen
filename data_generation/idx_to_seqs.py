import numpy as np

for i in range(3, 26):
    idx = np.load(f"/home/by7175/bscratch/pairs/unilat_random_pairs_idx{i}.npy")
    HPstr = np.load("/home/by7175/scratch/proteome/meta/HPseq_str.npy")
    
    unilat_seqs = HPstr[idx]
    print(unilat_seqs.shape)
    np.save(f"/home/by7175/bscratch/mar28/unilat_random_pairs_seq_{i}.npy", unilat_seqs)


