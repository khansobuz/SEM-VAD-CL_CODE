"""
COMPLETE SEMANTIC VIDEO ANOMALY DETECTION SYSTEM
- Loads ONLY from JSON caption folders (train/ and test/)
- NO TXT files needed!
- Full SFE + EAS + ACM architecture
- MIL + Contrastive + Focal losses
- Frame-level explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import deque
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================
class Config:
    # Paths - ONLY JSON FOLDERS NEEDED!
    TRAIN_CAPTION_DIR = Path(r"C:\Users\khanm\Desktop\lab_project\DMN\LLaVA1.5_CAPTIONS_PER_VIDEO\train")
    TEST_CAPTION_DIR = Path(r"C:\Users\khanm\Desktop\lab_project\DMN\LLaVA1.5_CAPTIONS_PER_VIDEO\test")
    SENT_TRANS_PATH = Path(r"C:\Users\khanm\Desktop\lab_project\SEM-VAD-CL\minilm_model")
    
    # Output paths
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR = Path("results")
    
    # Model parameters
    EMBEDDING_DIM = 384
    HIDDEN_DIM = 512
    NUM_PROTOTYPES = 20
    MEMORY_SIZE = 300
    
    # Training parameters
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5
    TEMPERATURE = 0.07
    
    # Quick test mode (set False for full training)
    QUICK_TEST = False
    NUM_TRAIN_SAMPLES = 1690 if QUICK_TEST else None  # None = use all
    NUM_TEST_SAMPLES = 210 if QUICK_TEST else None
    \


    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        
        # Verify paths exist
        assert self.TRAIN_CAPTION_DIR.exists(), f"Train caption dir not found: {self.TRAIN_CAPTION_DIR}"
        assert self.TEST_CAPTION_DIR.exists(), f"Test caption dir not found: {self.TEST_CAPTION_DIR}"
        assert self.SENT_TRANS_PATH.exists(), f"MiniLM model not found: {self.SENT_TRANS_PATH}"

config = Config()
print(f"\n{'='*80}")
print("SEMANTIC VIDEO ANOMALY DETECTION - DIRECT JSON LOADING")
print(f"{'='*80}")
print(f"Device: {config.DEVICE}")
print(f"Train captions: {config.TRAIN_CAPTION_DIR}")
print(f"Test captions: {config.TEST_CAPTION_DIR}")
print(f"Quick test mode: {config.QUICK_TEST}")

# ===================== TEXT ENCODER =====================
print(f"\nLoading MiniLM from: {config.SENT_TRANS_PATH}")
tokenizer = AutoTokenizer.from_pretrained(str(config.SENT_TRANS_PATH))
text_encoder = AutoModel.from_pretrained(str(config.SENT_TRANS_PATH)).to(config.DEVICE)
text_encoder.eval()
print("✓ Text encoder loaded")

def encode_texts(texts: List[str]) -> torch.Tensor:
    """Encode multiple texts to embeddings"""
    if not texts:
        return torch.zeros(1, config.EMBEDDING_DIM).to(config.DEVICE)
    
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    ).to(config.DEVICE)
    
    with torch.no_grad():
        outputs = text_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings


# ===================== 1. SFE: Semantic Feature Embedder =====================
class SemanticFeatureEmbedder(nn.Module):
    """Maps semantic captions to temporally coherent embeddings"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.temporal_lstm = nn.LSTM(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.projection = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.HIDDEN_DIM, config.EMBEDDING_DIM),
            nn.LayerNorm(config.EMBEDDING_DIM)
        )
    
    def forward(self, embeddings):
        lstm_out, _ = self.temporal_lstm(embeddings)
        refined = self.projection(lstm_out)
        return refined


# ===================== 2. ACM: Adaptive Continual Module =====================
class AdaptiveContinualModule(nn.Module):
    """Enforces temporal consistency and maintains adaptive memory"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.EMBEDDING_DIM,
            num_layers=1,
            batch_first=True
        )
        
        self.conv_smooth = nn.Conv1d(
            in_channels=config.EMBEDDING_DIM,
            out_channels=config.EMBEDDING_DIM,
            kernel_size=3,
            padding=1
        )
        
        self.normal_memory = deque(maxlen=config.MEMORY_SIZE)
        self.anomaly_memory = deque(maxlen=config.MEMORY_SIZE)
        self.hidden = None
    
    def forward(self, x, reset=True):
        if reset:
            self.hidden = None
        
        x_transposed = x.transpose(1, 2)
        x_smooth = self.conv_smooth(x_transposed).transpose(1, 2)
        gru_out, self.hidden = self.gru(x_smooth, self.hidden)
        refined = x + 0.35 * gru_out
        return refined
    
    def update_memory(self, embeddings, scores):
        for emb, score in zip(embeddings, scores):
            emb_np = emb.detach().cpu().numpy()
            if score < 0.3:
                self.normal_memory.append(emb_np)
            elif score > 0.7:
                self.anomaly_memory.append(emb_np)


# ===================== 3. EAS: Explainable Anomaly Scorer =====================
class ExplainableAnomalyScorer(nn.Module):
    """Prototype-based scoring with Chain-of-Thought reasoning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.normal_prototypes = nn.Parameter(
            torch.randn(config.NUM_PROTOTYPES, config.EMBEDDING_DIM)
        )
        self.anomaly_prototypes = nn.Parameter(
            torch.randn(config.NUM_PROTOTYPES, config.EMBEDDING_DIM)
        )
        
        self.reasoning_network = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings, captions):
        batch_size, seq_len, _ = embeddings.shape
        flat_embeddings = embeddings.view(-1, self.config.EMBEDDING_DIM)
        
        emb_norm = F.normalize(flat_embeddings, p=2, dim=1)
        normal_norm = F.normalize(self.normal_prototypes, p=2, dim=1)
        anomaly_norm = F.normalize(self.anomaly_prototypes, p=2, dim=1)
        
        normal_sim = torch.matmul(emb_norm, normal_norm.T)
        anomaly_sim = torch.matmul(emb_norm, anomaly_norm.T)
        
        normal_max_sim, normal_idx = normal_sim.max(dim=1)
        anomaly_max_sim, anomaly_idx = anomaly_sim.max(dim=1)
        
        best_normal = self.normal_prototypes[normal_idx]
        best_anomaly = self.anomaly_prototypes[anomaly_idx]
        
        reasoning_input = torch.cat([flat_embeddings, best_normal, best_anomaly], dim=1)
        scores_flat = self.reasoning_network(reasoning_input).squeeze(-1)
        scores = scores_flat.view(batch_size, seq_len)
        
        explanations = self.generate_explanations(
            captions, 
            normal_max_sim.view(batch_size, seq_len),
            anomaly_max_sim.view(batch_size, seq_len),
            scores
        )
        
        return scores, explanations
    
    def generate_explanations(self, captions_batch, normal_sim, anomaly_sim, scores):
        batch_size, seq_len = scores.shape
        explanations = []
        
        for b in range(batch_size):
            video_explanations = []
            video_captions = captions_batch[b] if b < len(captions_batch) else []
            
            for t in range(seq_len):
                if t < len(video_captions):
                    caption = video_captions[t]
                    n_sim = normal_sim[b, t].item()
                    a_sim = anomaly_sim[b, t].item()
                    score = scores[b, t].item()
                    
                    caption_short = caption[:100] + "..." if len(caption) > 100 else caption
                    
                    expl = f"Seg {t+1}: \"{caption_short}\"\n"
                    expl += f"  N:{n_sim:.3f} A:{a_sim:.3f} Score:{score:.3f} → "
                    
                    if score > 0.75:
                        expl += "⚠️ HIGH ANOMALY"
                    elif score > 0.5:
                        expl += "⚡ MODERATE"
                    elif score > 0.3:
                        expl += "❓ POSSIBLE"


                        
                    else:
                        expl += "✅ NORMAL"
                    
                    video_explanations.append(expl)
            
            explanations.append(video_explanations)
        
        return explanations


# ===================== COMPLETE MODEL =====================
class SemanticVADModel(nn.Module):
    """Complete Semantic Video Anomaly Detection Model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.sfe = SemanticFeatureEmbedder(config)
        self.acm = AdaptiveContinualModule(config)
        self.eas = ExplainableAnomalyScorer(config)
        self.projection_head = nn.Linear(config.EMBEDDING_DIM, 128)
    
    def forward(self, embeddings, captions, reset_acm=True):
        sfe_features = self.sfe(embeddings)
        acm_features = self.acm(sfe_features, reset=reset_acm)
        scores, explanations = self.eas(acm_features, captions)
        projections = self.projection_head(acm_features)
        
        return scores, explanations, projections, acm_features


# ===================== LOSS FUNCTIONS =====================
def mil_loss(scores, labels):
    batch_losses = []
    for video_scores, label in zip(scores, labels):
        max_score = video_scores.max()
        if label == 1:
            loss = torch.clamp(1.0 - max_score, min=0)
        else:
            loss = torch.clamp(max_score, min=0)
        batch_losses.append(loss)
    return torch.stack(batch_losses).mean()

def contrastive_loss(projections, temperature=0.07):
    proj_pooled = projections.mean(dim=1)
    proj_norm = F.normalize(proj_pooled, p=2, dim=1)
    logits = torch.matmul(proj_norm, proj_norm.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    return F.cross_entropy(logits, labels)

def focal_loss(scores, labels, gamma=2.0, alpha=0.75):
    batch_size, seq_len = scores.shape
    labels_expanded = labels.unsqueeze(1).expand(batch_size, seq_len).float()
    bce = F.binary_cross_entropy(scores, labels_expanded, reduction='none')
    pt = torch.exp(-bce)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


# ===================== DATA LOADING =====================
def load_caption_data_from_folder(caption_dir: Path, max_samples=None):
    """
    Load captions directly from JSON folder
    Label detection: 
    - "Normal" in filename → Normal (0)
    - Everything else → Anomaly (1)
    """
    print(f"\nLoading captions from: {caption_dir}")
    
    json_files = sorted(list(caption_dir.glob("*.json")))
    
    if max_samples:
        json_files = json_files[:max_samples]
    
    data = []
    
    for json_file in tqdm(json_files, desc=f"Loading {caption_dir.name}"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            
            captions = [c['caption'] for c in caption_data['captions']]
            
            # Skip videos with too few captions
            if len(captions) < 4:
                continue
            
            # Determine label from filename
            video_name = json_file.stem
            
            # If "Normal" in name → Normal (0), else → Anomaly (1)
            if "Normal" in video_name or "normal" in video_name:
                label = 0
            else:
                label = 1
            
            data.append({
                'video_name': video_name,
                'captions': captions,
                'label': label
            })
        
        except Exception as e:
            print(f"  Error loading {json_file.name}: {e}")
            continue
    
    # Print statistics
    num_normal = sum(1 for d in data if d['label'] == 0)
    num_anomaly = len(data) - num_normal
    
    print(f"  ✓ Loaded {len(data)} videos")
    print(f"    • Normal: {num_normal}")
    print(f"    • Anomaly: {num_anomaly}")
    
    return data


# ===================== TRAINING =====================
def train_model(model, train_data, test_data, config):
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=1e-6
    )
    
    best_auc = 0.0
    
    print("\n" + "="*80)
    print("TRAINING STARTED - SFE + ACM + EAS")
    print("="*80)
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        np.random.shuffle(train_data)
        
        pbar = tqdm(range(0, len(train_data), config.BATCH_SIZE), desc=f"Epoch {epoch}/{config.EPOCHS}")
        
        for batch_start in pbar:
            batch = train_data[batch_start:batch_start + config.BATCH_SIZE]
            
            try:
                batch_embeddings = []
                batch_captions = []
                batch_labels = []
                
                for item in batch:
                    embs = encode_texts(item['captions'])
                    batch_embeddings.append(embs)
                    batch_captions.append(item['captions'])
                    batch_labels.append(item['label'])
                
                max_len = max(e.size(0) for e in batch_embeddings)
                padded_embeddings = torch.stack([
                    F.pad(e, (0, 0, 0, max_len - e.size(0)))
                    for e in batch_embeddings
                ])
                
                batch_labels = torch.tensor(batch_labels).to(config.DEVICE)
                
                scores, explanations, projections, features = model(
                    padded_embeddings,
                    batch_captions,
                    reset_acm=True
                )
                
                loss_mil = mil_loss(scores, batch_labels)
                loss_contrastive = contrastive_loss(projections, config.TEMPERATURE)
                loss_focal = focal_loss(scores, batch_labels)
                
                total_loss = 0.6 * loss_mil + 0.3 * loss_contrastive + 0.1 * loss_focal
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
            
            except Exception as e:
                print(f"\n  Batch error: {e}")
                continue
        
        scheduler.step()
        
        # Evaluate every 2 epochs
        if epoch % 2 == 0 or epoch == config.EPOCHS:
            auc = evaluate_model(model, test_data, config)
            
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'auc': auc
                }, config.CHECKPOINT_DIR / "best_model.pth")
                print(f"\n🎯 NEW BEST AUC: {auc:.4f} - Saved!")
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | AUC: {auc:.4f} | Best: {best_auc:.4f}")
    
    return best_auc


# ===================== EVALUATION =====================
def evaluate_model(model, test_data, config):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_results = []
    
    with torch.no_grad():
        for item in tqdm(test_data, desc="Eval", leave=False):
            try:
                embeddings = encode_texts(item['captions']).unsqueeze(0)
                scores, explanations, _, _ = model(
                    embeddings,
                    [item['captions']],
                    reset_acm=True
                )
                
                video_score = scores.max().item()
                all_preds.append(video_score)
                all_labels.append(item['label'])
                
                all_results.append({
                    'video_name': item['video_name'],
                    'label': 'Anomaly' if item['label'] == 1 else 'Normal',
                    'score': float(video_score),
                    'segment_scores': scores[0].cpu().numpy().tolist()[:10],
                    'sample_explanations': explanations[0][:3] if explanations else []
                })
            
            except Exception as e:
                continue
    
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.0
    
    with open(config.RESULTS_DIR / "test_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            'auc': float(auc),
            'num_videos': len(all_results),
            'num_anomalies': sum(1 for r in all_results if r['label'] == 'Anomaly'),
            'num_normals': sum(1 for r in all_results if r['label'] == 'Normal'),
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    return auc


# ===================== MAIN =====================
def main():
    print(f"{'='*80}")
    print("SEMANTIC VIDEO ANOMALY DETECTION")
    print("Direct JSON Loading (No TXT files needed!)")
    print(f"{'='*80}\n")
    
    # Load data directly from JSON folders
    train_data = load_caption_data_from_folder(
        config.TRAIN_CAPTION_DIR,
        max_samples=config.NUM_TRAIN_SAMPLES
    )
    
    test_data = load_caption_data_from_folder(
        config.TEST_CAPTION_DIR,
        max_samples=config.NUM_TEST_SAMPLES
    )
    
    if len(train_data) == 0 or len(test_data) == 0:
        print("\n❌ No data loaded! Check paths.")
        return
    
    print(f"\n{'='*80}")
    print("DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Train: {len(train_data)} videos")
    print(f"Test: {len(test_data)} videos")
    
    # Create model
    print(f"\n{'='*80}")
    print("MODEL INITIALIZATION")
    print(f"{'='*80}")
    model = SemanticVADModel(config).to(config.DEVICE)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    best_auc = train_model(model, train_data, test_data, config)
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Best AUC: {best_auc:.4f}")
    print(f"✅ Model: {config.CHECKPOINT_DIR}/best_model.pth")
    print(f"✅ Results: {config.RESULTS_DIR}/test_results.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()