# import numpy as np

# seqs = np.array([[np.arange(20), np.arange(20)]])
# print(seqs.shape)
# np.save("test_seqs.npy", seqs)

# # after running
# # python generate_fe_landscapes.py test_seqs.npy test_fes.npy

# fes = np.load("test_fes.npy")
# print(fes.shape)

import numpy as np
import subprocess

seqs = np.array([[np.arange(20), np.arange(20)]])
print(seqs.shape)
np.save("test_seqs.npy", seqs)

_ = subprocess.run(
    ["python", "generate_fe_landscapes.py", "test_seqs.npy", "test_fes.npy"],
    capture_output=True,
    text=True
)

fes = np.load("test_fes.npy")
print(fes.shape)
