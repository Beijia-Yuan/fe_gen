import numpy as np
import torch
from tqdm import tqdm
import os
import sys
import importlib
import glob

def prepare_dataset(pairs, concs, all_seqs):
    seqs = all_seqs[pairs] #(x*y, 2, 40)
    data = np.concatenate([
        np.repeat(seqs[:,None,:,:], concs.shape[0], axis = 1), 
        np.repeat(concs[None,:,:,None], seqs.shape[0], axis = 0)
        ], axis = -1)
    data_torch = torch.tensor(data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3]), dtype = torch.float32)
    return data_torch

def get_pth_files(path):
    # Get only files ending with .pth (case-insensitive), sorted for stability
    files = [f for f in glob.glob(os.path.join(path, "*")) if os.path.isfile(f)]
    pths = [f for f in files if f.lower().endswith(".pth")]
    return sorted(pths)

def get_models(model_name, params_path, zdim, device):
    sys.path.append("/home/by7175/Server_idr/transformer/models")
    try:
        fe_fn = importlib.import_module(model_name)
    except ImportError as e:
        sys.exit(f"Error importing module '{model_name}': {e}")
    pths = get_pth_files(params_path)
    if len(pths) != 5:
        raise RuntimeError(f"Expected 5 checkpoints, found {len(pths)} in {params_path}")
    models = []
    for i in range(5):
        model = fe_fn.integrated_fe_predictor(zdim).to(device)
        ckpt = torch.load(pths[i], map_location=device, weights_only=True)
        state = ckpt.get("state_dict", ckpt)  # handle both styles
        model.load_state_dict(state)
        models.append(model)
    return models

if __name__ == "__main__":
    idx = sys.argv[1]
    pairs = np.load(f"/home/by7175/bscratch/pairs/unilat_random_pairs_idx{idx}.npy")
    all_seqs = np.load("/home/by7175/scratch/proteome/meta/HPseq_onehot.npy")
    module = "fe_predictor_tf_linc_meanz"
    params_path = "/home/by7175/scratch/transformer/HPferound_unilat_r12/zdim10/bestm/"
    zdim = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = get_models(module, params_path, zdim, device)

    X, Y = np.meshgrid(np.arange(0, 2.51, 0.05), np.arange(0, 2.51, 0.05), indexing = "ij") 
    concs = np.stack([X.ravel(), Y.ravel()], axis = 1)
    mask = np.round(concs.sum(axis=1), 10)<=2.5
    concs = concs[mask] # (N, 2)

    data_torch = prepare_dataset(pairs, concs, all_seqs)
    fes = []
    batch_pair = 8
    batch_size = batch_pair*len(concs)  # adjust based on your GPU/CPU memory
    n_samples = data_torch.shape[0]
    
    model_fes = np.ones((len(models), len(pairs), len(concs)))*(-1)
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, n_samples, batch_size)):
                j = start // len(concs)
                end = min(start + batch_size, n_samples)
                assert end % len(concs) == 0
                k = end // len(concs)
                batch = data_torch[start:end].to(device)
                fe_batch = model(batch).detach().cpu().numpy()
                model_fes[i, j:k] = (fe_batch).reshape(k-j, concs.shape[0])
    np.save(f"/home/by7175/bscratch/mar28/unilat_fes_new120k_{idx}.npy", model_fes)
    
