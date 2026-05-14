#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional
import scipy.constants as sc

kb = sc.k
T = 300
convert = 0.1*kb*T*1e27/101325

class force_field_encoder(nn.Module):
    def __init__(
        self,
        zdim: int,
        d_model: int = 16,
        num_heads: int = 2,
        ff_dim: int = 48,
        num_layers: int = 3,
        max_len: int = 50,
        dropout: float = 0.1,
        vocab_size: int = 20,   # assuming 20 unique tokens
    ):
        super().__init__()

        # 1. token embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # 2. positional encoding (fixed, added to token embeddings)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe.unsqueeze(0))  # shape [1, L, D]

        # 3. Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # 4. projection head
        self.head = nn.Linear(d_model, zdim)

    @staticmethod
    def _inverse_seq(seqs: torch.Tensor) -> torch.Tensor:
        """Reverse sequences along length dimension (dim=1)."""
        if seqs is None:
            return None
        return torch.flip(seqs, dims=[1])

    def _encode(self, token_batch: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """Token -> embedding space → mean-pooled latent."""
        x = self.tok_emb(token_batch)                               # [B, L, D]
        x = x + self.pos_encoding[:, : x.size(1)]                   # broadcast PE

        x = self.encoder(x, src_key_padding_mask=padding_mask)      # [B, L, D]

        # mean-pool while ignoring padding
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1, keepdim=True)      # [B, 1]
            x = (x.masked_fill(padding_mask.unsqueeze(-1), 0).sum(dim=1) /
                 lengths.clamp(min=1))
        else:
            x = x.mean(dim=1)

        return self.head(x)                                         # [B, zdim]

    def forward(self, token_batch: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """
        token_batch : LongTensor [B, L]    — padded with an index (e.g. 0)
        padding_mask: BoolTensor [B, L]    — True where padded
        """
        out_fwd = self._encode(token_batch, padding_mask)
        out_rev = self._encode(self._inverse_seq(token_batch), self._inverse_seq(padding_mask))
        return 0.5 * (out_fwd + out_rev)

class latent_force_conc_representation(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, latents, concs):
        """
        latent1: N*zdim
        latents: N*s*zdim, concs: N*s*1
        """
        zc_pairs = torch.cat((latents, concs),axis = 2) 
        return zc_pairs #N*s*(zdim+1)

class mean_field_decoder(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.zdim = zdim
        self.net = nn.Sequential(
                nn.Linear(zdim, 128),
                nn.SiLU(),
                nn.Linear(128,64),
                nn.SiLU(),
                nn.Linear(64, zdim * (zdim + 1) // 2),
                )

        iu, ju = torch.triu_indices(zdim, zdim)
        self.register_buffer("iu", iu, persistent=False)
        self.register_buffer("ju", ju, persistent=False)

    def vector_to_sym(self, v):
        B = v.shape[0]
        z = self.zdim
        G = v.new_zeros((B, z, z))
        G[:, self.iu, self.ju] = v
        # Mirror upper triangle; subtract diagonal once (it was doubled)
        G = G + G.transpose(-1, -2) - torch.diag_embed(G.diagonal(dim1=-2, dim2=-1))
        return G

    def forward(self, latents, concs):
        ctot = concs.sum(dim = 1) #(N,1)
        x = (concs * latents).sum(dim = 1) # x: (B, z)
        params = self.net(x)                      # (B, z*(z+1)/2)
        G = self.vector_to_sym(params)            # (B, z, z)
        fex = torch.einsum('bi,bij,bj->b', x, G, x)  # (B,)
        fid = (concs*(torch.log(concs+1e-20)-1)).sum(dim=1)
        #fid = ctot*torch.log(ctot/1e-4)
        return convert*(fid+fex.reshape(-1,1))


class integrated_fe_predictor(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.model1 = force_field_encoder(zdim)
        self.model3 = mean_field_decoder(zdim)

    def forward(self, x):

        b, n, lp1 = x.shape
        l = lp1-1

        tokens = x[:, :, :-1].long()          # ① cast to int (torch.int64 / torch.long)
        tokens = tokens.contiguous().view(b * n, l)
        latents = self.model1(tokens).view(b, n, -1)
        concs = x[:,:,-1:]
        
        out_tf = self.model3(latents, concs)

        return out_tf


