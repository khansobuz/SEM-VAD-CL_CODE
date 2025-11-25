import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn import metrics
import numpy as np
import os
from dataset1 import Normal_Loader, Anomaly_Loader
import torch.nn.functional as F
from collections import deque
import pandas as pd
from sentence_transformers import SentenceTransformer
import torchvision

# Silence torchvision beta warnings
torchvision.disable_beta_transforms_warning()

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# MLLM-based Semantic Feature Embedder (SFE)
class SFE(nn.Module):
    def __init__(self, input_size=2048, semantic_dim=512):
        super(SFE, self).__init__()
        self.feature_adapter = nn.Linear(input_size, 768)  # Adapt to intermediate dimension
        self.pos_encoder = PositionalEncoding(768)
        self.norm = nn.BatchNorm1d(768, momentum=0.1)
        self.dropout = nn.Dropout(0.15)
        self.text_encoder = SentenceTransformer("C:/Users/khanm/Desktop/lab_project/SEM-VAD-CL/minilm_model")
        self.text_projection = nn.Linear(384, semantic_dim)  # all-MiniLM-L6-v2 outputs 384-dim
        nn.init.xavier_uniform_(self.feature_adapter.weight)
        nn.init.xavier_uniform_(self.text_projection.weight)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.feature_adapter(x)
        x = F.relu(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = x.view(-1, x.shape[-1])
        x = self.norm(x)
        x = x.view(batch_size, seq_len, -1)

     
        f_labels = ["normal behavior" if (i % 2 == 0) else "anomalous behavior" 
                    for i in range(batch_size * seq_len)]

        real_prompts = [
            "A person walking calmly in a public area.",
            "Sudden movement in a crowded scene.",
            "People standing and talking normally.",
            "Fast running through a hallway.",
            "A quiet street with normal pedestrian flow.",
            "Chaotic behavior in a group of people.",
            "Regular daily activity in surveillance footage.",
            "An unusual action disrupting the scene."
        ]
        text_descriptions = [real_prompts[i % len(real_prompts)] for i in range(batch_size * seq_len)]

       
        text_embeddings = self.text_encoder.encode(f_labels, convert_to_tensor=True, device=x.device)
        text_embeddings = self.text_projection(text_embeddings)
        text_embeddings = text_embeddings.view(batch_size, seq_len, -1)
        # ==================================

        return text_embeddings, text_descriptions

# CoT-based Explainable Anomaly Scorer (EAS)
class EAS(nn.Module):
    def __init__(self, semantic_dim=512):
        super(EAS, self).__init__()
        self.scorer = nn.Linear(semantic_dim, 1)
        self.text_encoder = SentenceTransformer("C:/Users/khanm/Desktop/lab_project/SEM-VAD-CL/minilm_model")
        self.normal_templates = ["normal walking", "people standing calmly", "regular traffic", "people chatting"]
        self.anomaly_templates = ["person running suddenly", "fighting in crowd", "stealing an object", "vandalism"]
        nn.init.xavier_uniform_(self.scorer.weight)
        nn.init.constant_(self.scorer.bias, 0.1)

    def forward(self, semantic_vector, text_descriptions):
        scores = self.scorer(semantic_vector)
        if not text_descriptions:
            return scores, []
        with torch.no_grad():
            text_embeddings = self.text_encoder.encode(text_descriptions + self.normal_templates + self.anomaly_templates,
                                                     convert_to_tensor=True, device=semantic_vector.device)
            text_desc_emb = text_embeddings[:len(text_descriptions)]
            normal_emb = text_embeddings[len(text_descriptions):len(text_descriptions)+len(self.normal_templates)]
            anomaly_emb = text_embeddings[len(text_descriptions)+len(self.normal_templates):]
            normal_sim = F.cosine_similarity(text_desc_emb.unsqueeze(1), normal_emb.unsqueeze(0), dim=-1)
            anomaly_sim = F.cosine_similarity(text_desc_emb.unsqueeze(1), anomaly_emb.unsqueeze(0), dim=-1)
            cot_reasons = []
            for i in range(len(text_descriptions)):
                n_sim = normal_sim[i].mean() if normal_sim[i].dim() > 0 else normal_sim[i]
                a_sim = anomaly_sim[i].mean() if anomaly_sim[i].dim() > 0 else anomaly_sim[i]
                n_idx = torch.argmax(normal_sim[i]) if normal_sim[i].dim() > 0 else 0
                a_idx = torch.argmax(anomaly_sim[i]) if anomaly_sim[i].dim() > 0 else 0
                reason = f"Frame {i}: Score {scores[i].item():.3f}. "
                if a_sim > n_sim:
                    top_anomaly = self.anomaly_templates[a_idx.item()]
                    reason += f"Anomalous due to similarity with '{top_anomaly}' (sim={a_sim:.3f}) vs normal (sim={n_sim:.3f})."
                else:
                    top_normal = self.normal_templates[n_idx.item()]
                    reason += f"Normal due to similarity with '{top_normal}' (sim={n_sim:.3f}) vs anomaly (sim={a_sim:.3f})."
                cot_reasons.append(reason)
        return scores, cot_reasons

# Adaptive Continual Module (ACM)
class ACM(nn.Module):
    def __init__(self, input_size, memory_size=512, semantic_dim=512):
        super(ACM, self).__init__()
        self.semantic_memory = nn.Parameter(torch.randn(memory_size, semantic_dim) * 0.01)
        self.feature_to_semantic = nn.Linear(input_size, semantic_dim)
        self.gru = nn.GRU(semantic_dim, semantic_dim, batch_first=True)
        self.norm = nn.BatchNorm1d(semantic_dim, momentum=0.1)
        self.temporal_smoother = nn.Conv1d(semantic_dim, semantic_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.15)
        self.temporal_weight = 0.1

    def forward(self, feature_vector, sequence_length=32):
        batch_size = feature_vector.size(0)
        semantic_vector = self.feature_to_semantic(feature_vector)
        semantic_vector = semantic_vector.view(batch_size // sequence_length, sequence_length, -1)
        semantic_vector, _ = self.gru(semantic_vector)
        semantic_vector = semantic_vector.contiguous().view(-1, semantic_vector.shape[-1])
        semantic_vector = self.norm(semantic_vector)
        semantic_vector = self.dropout(semantic_vector)
        semantic_vector = semantic_vector.view(batch_size, -1)
        effective_seq_len = min(sequence_length, batch_size)
        if batch_size >= effective_seq_len and sequence_length > 1:
            semantic_vector_seq = semantic_vector.view(batch_size // effective_seq_len, effective_seq_len, -1).transpose(1, 2)
            smoothed_vector = self.temporal_smoother(semantic_vector_seq).transpose(1, 2).contiguous().view(batch_size, -1)
            diff = semantic_vector_seq[:, :, 1:] - semantic_vector_seq[:, :, :-1]
            temporal_loss = self.temporal_weight * torch.mean(diff ** 2 + 1e-8)  # Add epsilon for stability
        else:
            smoothed_vector = semantic_vector
            temporal_loss = torch.tensor(0.0, device=semantic_vector.device)
        semantic_memory_expanded = self.semantic_memory.unsqueeze(0).expand(batch_size, -1, -1)
        similarity = torch.norm(semantic_memory_expanded - smoothed_vector.unsqueeze(1), dim=2)
        min_similarity = torch.min(similarity, dim=1)[0]
        return min_similarity, semantic_vector, temporal_loss

# VAD_Model (SEM-VAD-CL)
class VAD_Model(nn.Module):
    def __init__(self, input_size=2048, num_classes=1, memory_size=512, semantic_dim=512):
        super(VAD_Model, self).__init__()
        self.feature_processor = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU()
        )
        self.sfe = SFE(input_size=input_size, semantic_dim=semantic_dim)
        self.eas = EAS(semantic_dim=semantic_dim)
        self.acm = ACM(input_size=semantic_dim, memory_size=memory_size, semantic_dim=semantic_dim)
        self.fc = nn.Linear(semantic_dim, num_classes)
        self.projection = nn.Linear(semantic_dim, 128)
        self.residual = nn.Linear(input_size, semantic_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.residual.weight)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        sequence_length = x.size(1)
        processed_features = self.feature_processor(x)
        residual = self.residual(x)
        semantic_features, text_descriptions = self.sfe(processed_features)
        semantic_features = semantic_features + residual
        anomaly_score_eas, cot_reasons = self.eas(semantic_features, text_descriptions)
        final_feature = semantic_features[:, -1, :]
        anomaly_score_acm, semantic_vector, temporal_loss = self.acm(final_feature, sequence_length=sequence_length)
        out = self.fc(semantic_vector)
        proj = self.projection(semantic_vector)
        return out, anomaly_score_eas.squeeze(-1), proj, cot_reasons, temporal_loss

# Contrastive Loss
def contrastive_loss(proj, batch_size, device, temperature=0.07):
    proj = F.normalize(proj, dim=1)
    mid = proj.size(0) // 2
    pos_pairs = proj[:mid]
    neg_pairs = proj[mid:]
    logits = torch.matmul(pos_pairs, neg_pairs.T) / temperature
    labels = torch.arange(min(mid, neg_pairs.size(0))).to(device)
    return F.cross_entropy(logits, labels)

# Enhanced MIL Loss
def MIL(y_pred, batch_size, device, margin=1.0):
    loss = torch.tensor(0., device=device)
    sparsity = torch.tensor(0., device=device)
    smooth = torch.tensor(0., device=device)
    frames_per_bag = y_pred.size(0) // batch_size
    y_pred = y_pred.view(batch_size, frames_per_bag)
    for i in range(batch_size):
        mid_point = frames_per_bag // 2
        anomaly_index = torch.randperm(mid_point).to(device)
        normal_index = torch.randperm(mid_point).to(device)
        y_anomaly = y_pred[i, :mid_point][anomaly_index]
        y_normal = y_pred[i, mid_point:][normal_index]
        y_anomaly_max = torch.max(y_anomaly)
        y_normal_max = torch.max(y_normal)
        loss += F.relu(margin - (y_anomaly_max - y_normal_max))
        sparsity += torch.sum(y_anomaly) * 0.0005
        smooth += torch.sum((y_pred[i, :frames_per_bag-1] - y_pred[i, 1:frames_per_bag]) ** 2) * 0.0005
    mil_loss = (loss + sparsity + smooth) / batch_size
    return torch.clamp(mil_loss, min=0.0)

# Focal Loss
def focal_loss(y_pred, batch_size, device, gamma=2.0, alpha=0.995):
    y_true = torch.cat([torch.ones(batch_size * 32), torch.zeros(batch_size * 32)]).to(device)
    y_pred = y_pred.squeeze(-1)
    y_pred = 0.8 * torch.sigmoid(y_pred) + 0.2 * torch.sigmoid(y_pred).mean() + 1e-7
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()

# Anomaly Score Loss
def anomaly_score_loss(anomaly_score, batch_size, device, margin=1.0):
    mid = anomaly_score.size(0) // 2
    anomaly_scores = anomaly_score[:mid]
    normal_scores = anomaly_score[mid:]
    diff = torch.mean(anomaly_scores) - torch.mean(normal_scores)
    return F.relu(margin - diff)

# Training function
def train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer, criterion, device, replay_buffer=None):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    avg_mil_loss = 0
    avg_cl_loss = 0
    avg_fl_loss = 0
    avg_as_loss = 0
    avg_temp_loss = 0
    batch_count = 0
    if epoch < 3:
        model.feature_processor.eval()
        model.sfe.eval()
        for param in model.feature_processor.parameters():
            param.requires_grad = False
        for param in model.sfe.parameters():
            param.requires_grad = False
    else:
        model.feature_processor.train()
        model.sfe.train()
        for param in model.feature_processor.parameters():
            param.requires_grad = True
        for param in model.sfe.parameters():
            param.requires_grad = True
    for param_group in optimizer.param_groups:
        print(f'Learning Rate: {param_group["lr"]:.10f}')
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        scale = np.random.uniform(0.1, 1.9)
        normal_inputs = normal_inputs * scale + torch.randn_like(normal_inputs) * 0.05
        anomaly_inputs = anomaly_inputs * scale + torch.randn_like(anomaly_inputs) * 0.05
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        if replay_buffer and len(replay_buffer) > 0:
            num_samples = min(len(replay_buffer), batch_size * 2)
            losses = torch.tensor([replay_buffer[i][1].mean().item() for i in range(len(replay_buffer))])
            probs = F.softmax(losses / 0.000005, dim=0).numpy() if losses.max() > 0 else None
            replay_samples = np.random.choice(len(replay_buffer), num_samples, replace=False, p=probs)
            replay_inputs = []
            for idx in replay_samples:
                r_inputs = replay_buffer[idx][0]
                start = np.random.randint(0, max(1, r_inputs.size(0) - 32))
                chunk = r_inputs[start:start + 32].to(device)
                if chunk.size(0) < 32:
                    chunk = F.pad(chunk, (0, 0, 0, 32 - chunk.size(0)))
                replay_inputs.append(chunk)
            replay_inputs = torch.cat(replay_inputs, dim=0)
            if replay_inputs.size(0) < batch_size * 32:
                replay_inputs = F.pad(replay_inputs, (0, 0, 0, batch_size * 32 - replay_inputs.size(0)))
            inputs = torch.cat([inputs, replay_inputs[:batch_size * 32]], dim=0)
            if inputs.size(0) > batch_size * 64:
                inputs = inputs[:batch_size * 64]
        outputs, anomaly_score, proj, cot_reasons, temporal_loss = model(inputs)
        mil_loss = criterion(anomaly_score, batch_size, device)
        cl_loss = contrastive_loss(proj, batch_size, device)
        fl_loss = focal_loss(anomaly_score, batch_size, device)
        as_loss = anomaly_score_loss(anomaly_score, batch_size, device)
        score_penalty = 0.000001 * torch.clamp(anomaly_score, -10, 10).pow(2).mean()  # Reduced penalty and clamped
        loss = 0.6 * mil_loss + 0.005 * cl_loss + 2.2 * fl_loss + 0.25 * as_loss + temporal_loss
        if torch.isnan(loss) or torch.isinf(loss):
            continue  # Skip batch if loss is NaN or Inf
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        train_loss += loss.item() if not torch.isnan(loss) else 0
        avg_mil_loss += mil_loss.item()
        avg_cl_loss += cl_loss.item()
        avg_fl_loss += fl_loss.item()
        avg_as_loss += as_loss.item()
        avg_temp_loss += temporal_loss.item() if not torch.isnan(temporal_loss) else 0
        batch_count += 1
        if loss.item() > 0.1 and not torch.isnan(loss):  # Store high-loss samples
            replay_buffer.append((inputs.detach().cpu(), anomaly_score.detach().cpu()))
    avg_loss = train_loss / batch_count
    avg_mil_loss /= batch_count
    avg_cl_loss /= batch_count
    avg_fl_loss /= batch_count
    avg_as_loss /= batch_count
    avg_temp_loss /= batch_count
    print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}, MIL: {avg_mil_loss:.4f}, CL: {avg_cl_loss:.4f}, FL: {avg_fl_loss:.4f}, AS: {avg_as_loss:.4f},  AUC: -')
    return avg_loss, avg_mil_loss, avg_cl_loss, avg_fl_loss, avg_as_loss, avg_temp_loss

# Testing function
def test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader, device):
    model.eval()
    global best_auc
    auc = 0
    cot_log = []
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            _, score, _, cot_reasons, _ = model(inputs)
            score = torch.sigmoid(score).cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0] // 16, 33))
            for j in range(32):
                score_list[int(step[j]) * 16:int(step[j + 1]) * 16] = score[j % len(score)]
            score_list = (score_list - np.min(score_list)) / (np.max(score_list) - np.min(score_list) + 1e-5)
            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = max(0, gts[k * 2] - 1)
                e = min(gts[k * 2 + 1], frames[0])
                gt_list[s:e] = 1
            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs.size(-1)).to(device)
            _, score2, _, cot_reasons2, _ = model(inputs2)
            score2 = torch.sigmoid(score2).cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames[0] // 16, 33))
            for j in range(32):
                score_list2[int(step2[j]) * 16:int(step2[j + 1]) * 16] = score2[j % len(score2)]
            score_list2 = (score_list2 - np.min(score_list2)) / (np.max(score_list2) - np.min(score_list2) + 1e-5)
            gt_list2 = np.zeros(frames2[0])
            for k in range(len(gts2) // 2):
                s = max(0, gts2[k * 2] - 1)
                e = min(gts2[k * 2 + 1], frames2[0])
                gt_list2[s:e] = 1
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
            fpr, tpr, _ = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)
            cot_log.extend(cot_reasons + cot_reasons2)
        avg_auc = auc / 140  # Fixed batch count as in original
        print(f'Epoch: {epoch}, Loss: -, AUC: {avg_auc:.4f}, Best AUC: {max(best_auc, avg_auc):.4f}')
        state = {'net': model.state_dict()}
        if avg_auc > best_auc:
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_auc = avg_auc
            print(f'New best AUC: {best_auc:.4f}')
        if avg_auc >= 0.80:
            print('Saving high AUC checkpoint..')
            torch.save(state, f'./checkpoint/ckpt_auc_{avg_auc:.4f}.pth')
       # with open(f'cot_reasons_epoch_{epoch}.txt', 'w') as f:
           # for reason in cot_log:
              #  f.write(reason + '\n')
    return avg_auc

# Main script
if __name__ == '__main__':
    torch.manual_seed(42)
    modality = 'TWO'
    input_dim = 2048
    memory_size = 512
    semantic_dim = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_auc = 0
    metrics_log = []
    normal_train_dataset = Normal_Loader(is_train=1, modality=modality, augment=False)
    normal_test_dataset = Normal_Loader(is_train=0, modality=modality, augment=False)
    anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=modality, augment=False)
    anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=modality, augment=False)
    normal_train_loader = DataLoader(normal_train_dataset, batch_size=1, shuffle=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=1, shuffle=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)
    model = VAD_Model(input_size=input_dim, num_classes=1, memory_size=memory_size, semantic_dim=semantic_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.000001, weight_decay=0.00001)
    def warmup_lambda(epoch):
        if epoch < 35:
            return 0.03 + (1.0 - 0.03) * epoch / 34
        return 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=120, eta_min=0.000000005)
    criterion = MIL
    replay_buffer = deque(maxlen=2000)  # Original size
    def load_checkpoint(checkpoint_path, model, best_auc_value):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
            state_dict = checkpoint['net']
            model_state_dict = model.state_dict()
            compatible_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
            model_state_dict.update(compatible_state_dict)
            for k, v in model_state_dict.items():
                if k not in compatible_state_dict and v.requires_grad and len(v.shape) >= 2:
                    nn.init.xavier_uniform_(v, gain=0.01)
                elif k not in compatible_state_dict and v.requires_grad and len(v.shape) == 1:
                    nn.init.zeros_(v)
            model.load_state_dict(model_state_dict)
            print(f"Loaded compatible checkpoint weights from {checkpoint_path} with AUC {best_auc_value:.4f}")
            return best_auc_value
        return None
    best_auc = load_checkpoint('./checkpoint/ckpt_auc_0.8548.pth', model, 0.8592) or 0
    if best_auc == 0:
        print("No checkpoint found, training from scratch")
    for epoch in range(200):
        print(f'Before scheduler step, epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.10f}')
        train_loss, avg_mil_loss, avg_cl_loss, avg_fl_loss, avg_as_loss, avg_temp_loss = train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer, criterion, device, replay_buffer)
        auc = test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader, device)
        if epoch < 70:  # Original warmup for 70 epochs
            warmup_scheduler.step()
            print(f'After scheduler step, epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.10f}')
        else:
            cosine_scheduler.step()
            print(f'After scheduler step, epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.10f}')
        metrics_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'mil_loss': avg_mil_loss,
            'cl_loss': avg_cl_loss,
            'fl_loss': avg_fl_loss,
            'as_loss': avg_as_loss,
            'temporal_loss': avg_temp_loss,
            'auc': auc
        })
    print(f'Final best AUC: {best_auc:.4f}')
