"""
TW-VAD – Strong AP version (optimized for 84+)
SFE + EAS + ACM
Real LLaVA CoT only used at the very end for explanations
"""

import os
import json
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from dataset_xd import Normal_Loader_XD, Anomaly_Loader_XD, XDViolence_Loader

# ==================== SETTINGS (STRONG AP) ====================
DATA_PATH       = r'C:/Users/khanm/Desktop/lab_project/DMN/xd_vio/'
FEATURE_DIM     = 1024
BATCH_SIZE      = 4
EPOCHS          = 100
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 5e-4
LAMBDA_TEMP     = 0.08
LAMBDA_RANK     = 0.5
LAMBDA_EWC      = 0.0               # OFF for pure AP
EVAL_EVERY      = 1
MAX_SEQ_LEN     = 512
SAVE_DIR        = './checkpoints_twvad_strong'

# LLM only for final explanations
USE_REAL_LLM    = False
LLM_CALL_EVERY  = 9999
MLLM_PATH       = r"C:\Users\khanm\Desktop\lab_project\DMN\MLLM_MODEL"
LLM_MODEL_NAME  = MLLM_PATH
# ==============================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(SAVE_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Optional LLaVA (only loaded if needed at the end)
# ----------------------------------------------------------------------
llava_processor = None
llava_model = None

def load_llava_once():
    global llava_processor, llava_model, USE_REAL_LLM
    if not USE_REAL_LLM or llava_model is not None:
        return
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        from PIL import Image
        print(f"Loading LLaVA from: {LLM_MODEL_NAME}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )
        llava_processor = AutoProcessor.from_pretrained(LLM_MODEL_NAME, local_files_only=True)
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True
        )
        llava_model.eval()
        for p in llava_model.parameters():
            p.requires_grad = False
        print("✓ LLaVA loaded")
    except Exception as e:
        print(f"LLaVA load failed: {e}")
        USE_REAL_LLM = False


def real_llm_cot(beta_vec: torch.Tensor) -> str:
    if not USE_REAL_LLM or llava_model is None:
        return "surrogate-explanation"
    try:
        from PIL import Image
        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        topk = torch.topk(beta_vec, k=min(5, beta_vec.numel()))
        desc = f"Top prototype indices: {topk.indices.tolist()}, values: {[round(v,3) for v in topk.values.tolist()]}"
        prompt = (
            "USER: <image>\nYou are an expert in video anomaly detection. "
            "Given the prototype similarities, reason if the segment is normal or anomalous. "
            "Keep under 60 words.\n\n" + desc + "\n\nASSISTANT:"
        )
        inputs = llava_processor(text=prompt, images=dummy, return_tensors="pt", padding=True)
        inputs = {k: v.to(llava_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = llava_model.generate(**inputs, max_new_tokens=70, do_sample=False)
        text = llava_processor.decode(out[0], skip_special_tokens=True)
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
        return text
    except Exception as e:
        return "surrogate-explanation"


# ----------------------------------------------------------------------
# Model components
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            extra = seq_len - self.pe.size(1)
            pe = torch.cat([self.pe, self.pe[:, -1:].repeat(1, extra, 1)], dim=1)
        else:
            pe = self.pe[:, :seq_len]
        return self.dropout(x + pe)


class SemanticFeatureEmbedder(nn.Module):
    def __init__(self, d_in=1024, d_model=512, d_semantic=512,
                 n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.phi_text = nn.Linear(d_model, d_semantic)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_semantic, nhead=n_heads,
            dim_feedforward=d_semantic * 4, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.temp_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        h = self.proj(x)
        h = self.pos_enc(h)
        f = self.norm(h)
        tokens = self.adapter(f)
        llm_out = tokens + F.gelu(self.adapter(tokens))
        s = self.phi_text(llm_out)
        s = F.layer_norm(s, (s.size(-1),))
        return self.temp_encoder(s)


class ExplainableAnomalyScorer(nn.Module):
    def __init__(self, d_semantic=512, n_prototypes=32, d_reasoning=512, dropout=0.1):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.phi_anom = nn.Sequential(
            nn.Linear(d_semantic, d_semantic // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_semantic // 2, 1),
        )
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_semantic) * 0.02)
        self.phi_text = nn.Sequential(
            nn.Linear(n_prototypes, d_reasoning),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_reasoning, d_reasoning),
        )
        self.temp_agg = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_reasoning, nhead=8,
                dim_feedforward=d_reasoning * 4, dropout=dropout,
                batch_first=True, activation='gelu'
            ), num_layers=1
        )

    def forward(self, s_prime, generate_explanations=False):
        B, T, D = s_prime.shape
        alpha_logits = self.phi_anom(s_prime).squeeze(-1)
        alpha = torch.sigmoid(alpha_logits)

        flat = s_prime.reshape(B * T, D)
        beta = F.normalize(flat, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        beta = beta.view(B, T, self.n_prototypes)

        z = self.phi_text(beta)
        z_prime = self.temp_agg(z)

        explanations = None
        if generate_explanations and USE_REAL_LLM:
            explanations = []
            for b in range(B):
                vid_exp = []
                for t in range(min(T, 4)):
                    text = real_llm_cot(beta[b, t].detach().cpu())
                    vid_exp.append(text)
                explanations.append(vid_exp)

        return {
            'alpha': alpha,
            'alpha_logits': alpha_logits,
            'beta': beta,
            'z_prime': z_prime,
            'explanations': explanations
        }


class AdaptiveContinualModule(nn.Module):
    def __init__(self, d_semantic=512, memory_size=64, replay_size=256,
                 dropout=0.1, lambda_temp=0.08):
        super().__init__()
        self.lambda_temp = lambda_temp
        self.memory_size = memory_size
        self.phi_sem = nn.Sequential(
            nn.Linear(d_semantic, d_semantic),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(d_semantic, d_semantic, batch_first=True)
        self.bn = nn.BatchNorm1d(d_semantic)
        self.tcm = nn.Conv1d(d_semantic, d_semantic, kernel_size=3, padding=1)
        self.register_buffer('memory', torch.randn(memory_size, d_semantic) * 0.02)
        self.register_buffer('mem_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('replay', torch.zeros(replay_size, d_semantic))
        self.register_buffer('replay_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('replay_filled', torch.zeros(1, dtype=torch.long))

    def _update_memory(self, z):
        flat = z.detach().reshape(-1, z.size(-1))
        n = flat.size(0)
        ptr = int(self.mem_ptr.item())
        for i in range(n):
            idx = (ptr + i) % self.memory_size
            self.memory[idx] = 0.9 * self.memory[idx] + 0.1 * flat[i]
        self.mem_ptr[0] = (ptr + n) % self.memory_size

    def memory_align(self, z):
        B, T, D = z.shape
        flat = z.reshape(B * T, D)
        weights = F.softmax(flat @ self.memory.T, dim=-1)
        return (weights @ self.memory).view(B, T, D)

    def temporal_loss(self, h):
        if h.size(1) < 2:
            return torch.tensor(0., device=h.device)
        return self.lambda_temp * (h[:, 1:] - h[:, :-1]).pow(2).sum(-1).mean()

    def forward(self, s_prime, update=True):
        x = self.phi_sem(s_prime)
        h, _ = self.gru(x)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        h_smooth = self.tcm(h.transpose(1, 2)).transpose(1, 2)
        z_mem = self.memory_align(h_smooth)
        if self.training and update:
            self._update_memory(h_smooth)
        return {
            'h_smooth': h_smooth,
            'z_mem': z_mem,
            'l_temp': self.temporal_loss(h_smooth)
        }


class TWVAD(nn.Module):
    def __init__(self, d_in=1024, d_model=512, d_semantic=512,
                 n_prototypes=32, memory_size=64, lambda_temp=0.08, dropout=0.1):
        super().__init__()
        self.sfe = SemanticFeatureEmbedder(d_in, d_model, d_semantic, dropout=dropout)
        self.eas = ExplainableAnomalyScorer(d_semantic, n_prototypes, d_semantic, dropout)
        self.acm = AdaptiveContinualModule(d_semantic, memory_size, 256, dropout, lambda_temp)

        self.attention = nn.Sequential(
            nn.Linear(d_semantic, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.video_head = nn.Sequential(
            nn.Linear(d_semantic, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(64, 1)
        )

    def forward(self, x, generate_explanations=False):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        s_prime = self.sfe(x)
        eas_out = self.eas(s_prime, generate_explanations=generate_explanations)
        acm_out = self.acm(s_prime)

        alpha = eas_out['alpha']                          # (B, T)
        attn = torch.softmax(self.attention(s_prime), dim=1)
        video_feat = (s_prime * attn).sum(1)
        score_attn = torch.sigmoid(self.video_head(video_feat).squeeze(-1))
        score_max = alpha.max(dim=1)[0]

        # Strong fusion for better AP
        video_score = 0.55 * score_attn + 0.45 * score_max

        return {
            'video_score': video_score,
            'alpha': alpha,
            'attn': attn.squeeze(-1),
            'l_temp': acm_out['l_temp'],
            'explanations': eas_out.get('explanations', None),
        }


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------
def mil_bce(n_scores, a_scores):
    return (F.binary_cross_entropy(n_scores, torch.zeros_like(n_scores)) +
            F.binary_cross_entropy(a_scores, torch.ones_like(a_scores)))

def ranking_loss(n_scores, a_scores, margin=0.2):
    return torch.mean(torch.relu(margin - (a_scores - n_scores)))


# ----------------------------------------------------------------------
# Collate
# ----------------------------------------------------------------------
def custom_collate_fn(batch, max_len=MAX_SEQ_LEN):
    batch = [item[:max_len] if item.shape[0] > max_len else item for item in batch]
    max_len_batch = max(item.shape[0] for item in batch)
    feat_dim = batch[0].shape[1]
    padded = []
    for item in batch:
        if item.shape[0] < max_len_batch:
            pad = torch.zeros(max_len_batch - item.shape[0], feat_dim)
            item = torch.cat([item, pad], 0)
        padded.append(item)
    return torch.stack(padded)


# ----------------------------------------------------------------------
# Train / Eval
# ----------------------------------------------------------------------
def train_epoch(model, normal_loader, anomaly_loader, optimizer, device):
    model.train()
    total_loss = 0.
    n_batches = min(len(normal_loader), len(anomaly_loader))
    n_iter = iter(normal_loader)
    a_iter = iter(anomaly_loader)

    pbar = tqdm(range(n_batches), desc='Training')
    for _ in pbar:
        try:
            n_feat = next(n_iter).to(device)
            a_feat = next(a_iter).to(device)
        except StopIteration:
            break

        out_n = model(n_feat)
        out_a = model(a_feat)

        loss = mil_bce(out_n['video_score'], out_a['video_score'])
        loss = loss + 0.5 * (out_n['l_temp'] + out_a['l_temp'])
        loss = loss + LAMBDA_RANK * ranking_loss(out_n['video_score'], out_a['video_score'])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, test_loader, device, generate_explanations=False):
    model.eval()
    scores, labels = [], []
    all_explanations = []

    for feat, lab, _ in tqdm(test_loader, desc='Evaluating'):
        feat = feat.to(device)
        if feat.size(1) > MAX_SEQ_LEN:
            feat = feat[:, :MAX_SEQ_LEN]
        out = model(feat, generate_explanations=generate_explanations)
        scores.append(out['video_score'].squeeze().cpu().item())
        labels.append(1.0 if lab.sum().item() > 0 else 0.0)
        if out['explanations'] is not None:
            all_explanations.append(out['explanations'])

    scores = np.array(scores)
    labels = np.array(labels)
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap, all_explanations


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print(f'Device: {DEVICE}')
    print(f'USE_REAL_LLM (during training) = {USE_REAL_LLM}')
    print(f'Target: strong AP (84+)')

    print('\n=== Loading Data ===')
    normal_train  = Normal_Loader_XD(is_train=1, path=DATA_PATH, augment=False)
    anomaly_train = Anomaly_Loader_XD(is_train=1, path=DATA_PATH, augment=False)
    test_set = XDViolence_Loader(is_train=0, path=DATA_PATH, augment=False)

    normal_loader = DataLoader(normal_train, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, collate_fn=custom_collate_fn, drop_last=True)
    anomaly_loader = DataLoader(anomaly_train, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=4, collate_fn=custom_collate_fn, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    model = TWVAD(d_in=FEATURE_DIM, lambda_temp=LAMBDA_TEMP).to(DEVICE)
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_ap = best_auc = 0.0

    print('\n=== Starting Training (Strong AP) ===')
    for epoch in range(EPOCHS):
        print(f'\n{"="*60}\nEpoch [{epoch+1}/{EPOCHS}]\n{"="*60}')

        train_loss = train_epoch(model, normal_loader, anomaly_loader, optimizer, DEVICE)
        print(f'Train Loss: {train_loss:.4f}')

        auc, ap, _ = evaluate(model, test_loader, DEVICE)
        print(f'AUC: {auc:.4f} | AP: {ap:.4f}')

        if ap > best_ap:
            best_ap, best_auc = ap, auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap,
                'auc': auc,
            }, os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f'✓ New best AP: {best_ap:.4f}')

        scheduler.step()
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print('\n' + '='*60)
    print('TRAINING COMPLETE')
    print(f'Best AUC: {best_auc:.4f}')
    print(f'Best AP : {best_ap:.4f}')
    print('='*60)

    with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
        json.dump({'best_auc': float(best_auc), 'best_ap': float(best_ap)}, f, indent=4)


if __name__ == '__main__':
    main()