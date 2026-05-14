# encoder_decoder_model.py
import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import EsmModel, AutoTokenizer
from helpers import load_seqs_and_fes, seqs_to_tokens, tokens_to_seq, validate_sequences

AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")
idx_to_aa = {i: aa for i, aa in enumerate(AA_TOKENS)}
aa_to_idx = {aa: i for i, aa in idx_to_aa.items()}
SEP_TOKEN = 20
EOS_TOKEN = 21
PAD_TOKEN = 22
VOCAB_SIZE = 23

# Data set that returns fes, sequence pairs
class fes_dataset(Dataset):
    def __init__(self, fes_vectors, target_sequences):
        self.fes = torch.tensor(fes_vectors, dtype=torch.float32)
        self.sequences = torch.tensor(target_sequences, dtype=torch.long)

    def __len__(self):
        return len(self.fes)

    def __getitem__(self, idx):
        fes = self.fes[idx]
        seqs = self.sequences[idx]
        sep = torch.tensor([SEP_TOKEN])
        eos = torch.tensor([EOS_TOKEN])
        target = torch.cat([seqs[0], sep, seqs[1], eos])
        return fes, target

def collate_fn(batch):
    fes, sequences = zip(*batch)
    fes = torch.stack(fes)
    sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=PAD_TOKEN
    )
    return fes, sequences

class FESEncoder(nn.Module):
    def __init__(self, fes_dim=1326, d_model=320, n_tokens=16):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(fes_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, d_model * n_tokens)
        )

    def forward(self, fes):
        B = fes.shape[0]
        return self.proj(fes).view(B, self.n_tokens, self.d_model)

class SequenceDecoder(nn.Module):
    def __init__(self, d_model=320, nhead=8, num_layers=4, vocab_size=VOCAB_SIZE, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_embed = nn.Embedding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tgt_tokens, memory, tgt_key_padding_mask=None):
        B, T = tgt_tokens.shape
        positions = torch.arange(T, device=tgt_tokens.device).unsqueeze(0)
        x = self.embed(tgt_tokens) + self.pos_embed(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_tokens.device)
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.lm_head(out)

class fes_encoder_decoder_model(nn.Module):
    def __init__(self, fes_dim=1326, n_tokens=1):
        super().__init__()
        self.esm2 = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        d_model = self.esm2.config.hidden_size
        self.fes_encoder = FESEncoder(fes_dim=fes_dim, d_model=d_model, n_tokens=n_tokens)
        self.decoder = SequenceDecoder(d_model=d_model, nhead=8, num_layers=4)

    def forward(self, fes, labels=None):
        B = fes.shape[0]
        encoded_fes = self.fes_encoder(fes)
        loss = None
        logits = None
        if labels is not None:
            decoder_input = labels.clone()
            decoder_input[decoder_input == -100] = PAD_TOKEN
            decoder_input = torch.cat([
                torch.full((B, 1), EOS_TOKEN, dtype=torch.long, device=fes.device),
                decoder_input[:, :-1]
            ], dim=1)
            pad_mask = (decoder_input == PAD_TOKEN)
            logits = self.decoder(decoder_input, encoded_fes, tgt_key_padding_mask=pad_mask)
            loss_labels = labels.clone()
            loss_labels[loss_labels == PAD_TOKEN] = -100
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, VOCAB_SIZE),
                loss_labels.view(-1)
            )
        return loss, logits

    @torch.no_grad()
    def generate(self, fes, max_len=42, temperature=1.0):
        B = fes.shape[0]
        device = fes.device
        memory = self.fes_encoder(fes)

        generated = torch.full((B, 1), EOS_TOKEN, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.decoder(generated, memory)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1) # remove this line if doing argmax
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1) next_logits.argmax(dim=-1, keepdim=True)  next_logits.argmax(dim=-1, keepdim=True) 
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == EOS_TOKEN).all():
                break

        return generated[:, 1:]

def train(model, dataloader_train, dataloader_val, unilat_fes, n_epochs=20, lr=1e-3, device='cuda'):
    model.to(device)
    best_val_loss = 1e9

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(range(n_epochs)):
        model.train()
        total_loss = 0
        for fes, labels in tqdm(dataloader_train):
            fes, labels = fes.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'): #torch.cuda.amp.autocast():
                loss, _ = model(fes, labels=labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        # validation set
        model.eval()
        total_loss_val = 0
        with torch.no_grad():
            for fes, labels in tqdm(dataloader_val):
                fes, labels = fes.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    loss, _ = model(fes, labels=labels)
                total_loss_val += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {total_loss/len(dataloader_train):.4f} | Val Loss: {total_loss_val/len(dataloader_val):.4f}")
        epoch_val_loss = total_loss_val/len(dataloader_val)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'esm2_fes_finetuned.pt')

        # Inference
        model.eval()
        # select a random sample for validation
        randomstate = np.random.RandomState(0)
        random_indices = randomstate.choice(unilat_fes.shape[0], size=32, replace=False)
        test_fes = torch.from_numpy(unilat_fes[random_indices]).float().to(device)
        # old: test_fes = torch.from_numpy(unilat_fes[:32]).float().to(device)
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

    return model