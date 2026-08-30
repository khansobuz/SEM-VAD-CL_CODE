
import os
import json
import random
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
class Config:
    TRAIN_CAPTION_DIR = Path(r"C:\Users\khanm\Desktop\lab_project\DMN\LLaVA1.5_CAPTIONS_PER_VIDEO\train")
    TEST_CAPTION_DIR  = Path(r"C:\Users\khanm\Desktop\lab_project\DMN\LLaVA1.5_CAPTIONS_PER_VIDEO\test")
    SENT_TRANS_PATH   = Path(r"C:\Users\khanm\Desktop\lab_project\SEM-VAD-CL\minilm_model")

    CHECKPOINT_DIR = Path("checkpoints_twvad_caption")
    RESULTS_DIR    = Path("results_twvad_caption")

    EMBEDDING_DIM  = 384
    HIDDEN_DIM     = 512
    NUM_PROTOTYPES = 32
    MEMORY_SIZE    = 256

    BATCH_SIZE     = 16
    EPOCHS         = 150
    LEARNING_RATE  = 3e-4
    WEIGHT_DECAY   = 5e-5
    TEMPERATURE    = 0.07
    LAMBDA_TEMP    = 0.05

    QUICK_TEST         = False
    NUM_TRAIN_SAMPLES  = 1690 if QUICK_TEST else None
    NUM_TEST_SAMPLES   = 210 if QUICK_TEST else None

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        assert self.TRAIN_CAPTION_DIR.exists(), f"Missing: {self.TRAIN_CAPTION_DIR}"
        assert self.TEST_CAPTION_DIR.exists(), f"Missing: {self.TEST_CAPTION_DIR}"
        assert self.SENT_TRANS_PATH.exists(), f"Missing: {self.SENT_TRANS_PATH}"

config = Config()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

# ===================== TEXT ENCODER =====================
print(f"Loading MiniLM from: {config.SENT_TRANS_PATH}")
tokenizer = AutoTokenizer.from_pretrained(str(config.SENT_TRANS_PATH))
text_encoder = AutoModel.from_pretrained(str(config.SENT_TRANS_PATH)).to(config.DEVICE)
text_encoder.eval()
for p in text_encoder.parameters():
    p.requires_grad = False
print("✓ Text encoder loaded")

def encode_texts(texts: List[str]) -> torch.Tensor:
    if not texts:
        return torch.zeros(1, config.EMBEDDING_DIM, device=config.DEVICE)
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(config.DEVICE)
    with torch.no_grad():
        out = text_encoder(**inputs)
        emb = out.last_hidden_state.mean(dim=1)
        emb = F.normalize(emb, p=2, dim=1)
    return emb

# ===================== 1. SFE =====================
class SemanticFeatureEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )
        self.proj = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(config.HIDDEN_DIM, config.EMBEDDING_DIM),
            nn.LayerNorm(config.EMBEDDING_DIM)
        )

    def forward(self, x):  # (B, T, 384)
        h, _ = self.lstm(x)
        return self.proj(h)

# ===================== 2. ACM =====================
class AdaptiveContinualModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(config.EMBEDDING_DIM, config.EMBEDDING_DIM, batch_first=True)
        self.conv = nn.Conv1d(config.EMBEDDING_DIM, config.EMBEDDING_DIM, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(config.EMBEDDING_DIM)

        self.register_buffer("normal_memory", torch.zeros(config.MEMORY_SIZE, config.EMBEDDING_DIM))
        self.register_buffer("anomaly_memory", torch.zeros(config.MEMORY_SIZE, config.EMBEDDING_DIM))
        self.register_buffer("n_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("a_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("n_filled", torch.zeros(1, dtype=torch.long))
        self.register_buffer("a_filled", torch.zeros(1, dtype=torch.long))

    def _update(self, emb, score):
        if score < 0.3:
            idx = int(self.n_ptr.item()) % self.config.MEMORY_SIZE
            self.normal_memory[idx] = 0.9 * self.normal_memory[idx] + 0.1 * emb.detach()
            self.n_ptr[0] = (idx + 1) % self.config.MEMORY_SIZE
            self.n_filled[0] = min(self.config.MEMORY_SIZE, int(self.n_filled.item()) + 1)
        elif score > 0.7:
            idx = int(self.a_ptr.item()) % self.config.MEMORY_SIZE
            self.anomaly_memory[idx] = 0.9 * self.anomaly_memory[idx] + 0.1 * emb.detach()
            self.a_ptr[0] = (idx + 1) % self.config.MEMORY_SIZE
            self.a_filled[0] = min(self.config.MEMORY_SIZE, int(self.a_filled.item()) + 1)

    def temporal_loss(self, h):
        if h.size(1) < 2:
            return torch.tensor(0., device=h.device)
        return config.LAMBDA_TEMP * (h[:, 1:] - h[:, :-1]).pow(2).sum(-1).mean()

    def forward(self, x, scores=None, update_memory=False):
        h, _ = self.gru(x)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h + x)
        l_temp = self.temporal_loss(h)

        if update_memory and scores is not None and self.training:
            B, T, D = h.shape
            flat_h = h.reshape(B * T, D)
            flat_s = scores.reshape(B * T)
            for i in range(flat_h.size(0)):
                self._update(flat_h[i], flat_s[i].item())
        return h, l_temp

# ===================== 3. EAS =====================
class ExplainableAnomalyScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.normal_prototypes = nn.Parameter(
            torch.randn(config.NUM_PROTOTYPES, config.EMBEDDING_DIM) * 0.02
        )
        self.anomaly_prototypes = nn.Parameter(
            torch.randn(config.NUM_PROTOTYPES, config.EMBEDDING_DIM) * 0.02
        )
        self.scorer = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM * 3, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, embeddings, captions):
        B, T, D = embeddings.shape
        flat = embeddings.reshape(B * T, D)

        emb_n = F.normalize(flat, dim=-1)
        n_proto = F.normalize(self.normal_prototypes, dim=-1)
        a_proto = F.normalize(self.anomaly_prototypes, dim=-1)

        n_sim = emb_n @ n_proto.T
        a_sim = emb_n @ a_proto.T
        n_max, n_idx = n_sim.max(dim=1)
        a_max, a_idx = a_sim.max(dim=1)

        best_n = self.normal_prototypes[n_idx]
        best_a = self.anomaly_prototypes[a_idx]

        reason_in = torch.cat([flat, best_n, best_a], dim=1)
        logits = self.scorer(reason_in).squeeze(-1)
        scores = torch.sigmoid(logits).view(B, T)

        explanations = []
        for b in range(B):
            video_exp = []
            caps = captions[b] if b < len(captions) else []
            for t in range(min(T, len(caps), 8)):
                cap = caps[t][:90] + ("..." if len(caps[t]) > 90 else "")
                ns = n_max.view(B, T)[b, t].item()
                ans = a_max.view(B, T)[b, t].item()
                sc = scores[b, t].item()
                tag = "ANOMALY" if sc > 0.7 else ("UNCERTAIN" if sc > 0.4 else "NORMAL")
                video_exp.append(
                    f"Seg{t}: \"{cap}\" | N={ns:.3f} A={ans:.3f} Score={sc:.3f} → {tag}"
                )
            explanations.append(video_exp)

        return scores, explanations, n_max.view(B, T), a_max.view(B, T)

# ===================== Full Model =====================
class TWVADCaption(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sfe = SemanticFeatureEmbedder(config)
        self.acm = AdaptiveContinualModule(config)
        self.eas = ExplainableAnomalyScorer(config)
        self.proj_head = nn.Linear(config.EMBEDDING_DIM, 128)

    def forward(self, embeddings, captions, update_memory=False):
        sfe_out = self.sfe(embeddings)
        scores_tmp, _, _, _ = self.eas(sfe_out, captions)
        acm_out, l_temp = self.acm(sfe_out, scores=scores_tmp, update_memory=update_memory)
        scores, explanations, n_sim, a_sim = self.eas(acm_out, captions)
        proj = self.proj_head(acm_out)
        return scores, explanations, proj, l_temp

# ===================== Losses (improved) =====================
def mil_loss(scores, labels):
    """Top-k MIL – stronger and more stable"""
    losses = []
    for s, y in zip(scores, labels):
        k = min(3, s.numel())
        topk = torch.topk(s, k=k).values.mean()
        if y == 1:
            losses.append(F.relu(1.0 - topk) + 0.1 * (1.0 - topk) ** 2)
        else:
            losses.append(topk.pow(2))
    return torch.stack(losses).mean()

def contrastive_loss(proj, temperature=0.07):
    pooled = proj.mean(dim=1)
    pooled = F.normalize(pooled, dim=1)
    logits = pooled @ pooled.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def focal_loss(scores, labels, gamma=2.0, alpha=0.75):
    B, T = scores.shape
    # soft target for anomaly (0.9) reduces overconfidence
    y = torch.where(
        labels == 1,
        torch.full_like(labels, 0.9, dtype=torch.float),
        torch.zeros_like(labels, dtype=torch.float)
    )
    y = y.unsqueeze(1).expand(B, T)
    bce = F.binary_cross_entropy(scores, y, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()

# ===================== Data =====================
def load_caption_data(caption_dir: Path, max_samples=None):
    print(f"Loading from: {caption_dir}")
    files = sorted(caption_dir.glob("*.json"))
    if max_samples:
        files = files[:max_samples]

    data = []
    for f in tqdm(files, desc=caption_dir.name):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                obj = json.load(fp)
            captions = [c["caption"] for c in obj["captions"]]
            if len(captions) < 4:
                continue
            name = f.stem
            label = 0 if ("Normal" in name or "normal" in name) else 1
            data.append({"video_name": name, "captions": captions, "label": label})
        except Exception as e:
            print(f"Skip {f.name}: {e}")
    n0 = sum(d["label"] == 0 for d in data)
    print(f"  Loaded {len(data)} | Normal={n0} | Anomaly={len(data) - n0}")
    return data

def make_balanced_epoch(train_data):
    """Balance normal / anomaly each epoch"""
    normal = [d for d in train_data if d["label"] == 0]
    anomaly = [d for d in train_data if d["label"] == 1]
    if len(anomaly) == 0 or len(normal) == 0:
        return train_data
    if len(anomaly) < len(normal):
        anomaly = (anomaly * (len(normal) // len(anomaly) + 1))[:len(normal)]
    else:
        normal = (normal * (len(anomaly) // len(normal) + 1))[:len(anomaly)]
    mixed = normal + anomaly
    random.shuffle(mixed)
    return mixed

# ===================== Train / Eval =====================
def train_model(model, train_data, test_data, config):
    opt = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    sch = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)

    best_auc, best_ap = 0.0, 0.0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_data = make_balanced_epoch(train_data)
        total_loss, n_batches = 0.0, 0

        pbar = tqdm(range(0, len(epoch_data), config.BATCH_SIZE),
                    desc=f"Epoch {epoch}/{config.EPOCHS}")
        for st in pbar:
            batch = epoch_data[st:st + config.BATCH_SIZE]
            try:
                embs, caps, labels = [], [], []
                for it in batch:
                    e = encode_texts(it["captions"])
                    embs.append(e)
                    caps.append(it["captions"])
                    labels.append(it["label"])

                max_t = max(e.size(0) for e in embs)
                x = torch.stack([F.pad(e, (0, 0, 0, max_t - e.size(0))) for e in embs])
                y = torch.tensor(labels, device=config.DEVICE)

                scores, explanations, proj, l_temp = model(x, caps, update_memory=True)

                loss = (
                    0.70 * mil_loss(scores, y) +
                    0.15 * contrastive_loss(proj, config.TEMPERATURE) +
                    0.10 * focal_loss(scores, y) +
                    0.05 * l_temp
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            except Exception as e:
                print(f"Batch error: {e}")
                continue

        sch.step()

        if epoch % 2 == 0 or epoch == config.EPOCHS:
            auc, ap = evaluate(model, test_data, config, epoch)
            avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch} | Loss {avg_loss:.4f} | AUC {auc:.4f} | AP {ap:.4f} | BestAUC {best_auc:.4f}")

            if auc > best_auc:
                best_auc, best_ap = auc, ap
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "auc": auc,
                    "ap": ap
                }, config.CHECKPOINT_DIR / "best_model.pth")
                print(f"✓ New best AUC: {auc:.4f}")

    return best_auc, best_ap

@torch.no_grad()
def evaluate(model, test_data, config, epoch=0):
    model.eval()
    preds, labels, results = [], [], []

    for it in tqdm(test_data, desc="Eval", leave=False):
        try:
            emb = encode_texts(it["captions"]).unsqueeze(0)
            scores, explanations, _, _ = model(emb, [it["captions"]], update_memory=False)

            # robust video score: 0.6 * max + 0.4 * top-5 mean
            flat = scores.flatten()
            k = min(5, flat.numel())
            topk_mean = torch.topk(flat, k=k).values.mean().item()
            video_score = 0.6 * flat.max().item() + 0.4 * topk_mean

            preds.append(video_score)
            labels.append(it["label"])
            results.append({
                "video_name": it["video_name"],
                "label": "Anomaly" if it["label"] == 1 else "Normal",
                "score": video_score,
                "explanations": explanations[0][:5] if explanations else []
            })
        except Exception:
            continue

    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
    else:
        auc, ap = 0.0, 0.0

    with open(config.RESULTS_DIR / f"results_epoch_{epoch}.json", "w", encoding="utf-8") as f:
        json.dump({"auc": auc, "ap": ap, "results": results}, f, indent=2, ensure_ascii=False)

    return auc, ap

# ===================== Main =====================
def main():
    print("=" * 70)
    print("TW-VAD Caption (Improved) | Target AUC >= 0.88")
    print("=" * 70)

    train_data = load_caption_data(config.TRAIN_CAPTION_DIR, config.NUM_TRAIN_SAMPLES)
    test_data = load_caption_data(config.TEST_CAPTION_DIR, config.NUM_TEST_SAMPLES)

    if not train_data or not test_data:
        print("No data loaded. Check paths.")
        return

    model = TWVADCaption(config).to(config.DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_auc, best_ap = train_model(model, train_data, test_data, config)

    print("=" * 70)
    print(f"Done | Best AUC: {best_auc:.4f} | Best AP: {best_ap:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
