import numpy as np

f = np.load("/home/by7175/scratch/proteome/eosdata/random_data/pressures_new/pressures_r12_2.npy")
pairs = np.unique(f[:,3:5].astype(int), axis = 0)
print(pairs[0])
print(pairs.shape)
strs = np.load("/home/by7175/scratch/proteome/meta/HPseq_str.npy")
unilat_seqs = strs[pairs - 1]
print(unilat_seqs.shape)
np.save("/home/by7175/bscratch/unilat_seqs.npy", unilat_seqs)
np.save("/home/by7175/bscratch/unilat_idxs.npy", pairs - 1)
