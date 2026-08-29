import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
import pandas as pd
import random
from scipy.ndimage import rotate

class XDViolence_Loader(Dataset):
    """
    Enhanced Dataset loader for XD-Violence dataset with improved augmentations
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path='./xd_vio/', modality='RGB', augment=False):
        super(XDViolence_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        self.augment = augment
        
        # Load ground truth JSON
        json_path = os.path.join(path, 'List/xd-violence_ground_truth.json')
        print(f"Loading ground truth from: {json_path}")
        with open(json_path, 'r') as f:
            self.ground_truth = json.load(f)
        
        print(f"Ground truth loaded: {len(self.ground_truth)} videos")
        
        # Load train/test split
        if self.is_train == 1:
            csv_file = os.path.join(path, 'List/xd-violence.training.csv')
        else:
            csv_file = os.path.join(path, 'List/xd-violence.testing.csv')
        
        print(f"Loading split from: {csv_file}")
        df = pd.read_csv(csv_file)
        self.video_list = df['video-id'].tolist()
        print(f"Loaded {len(self.video_list)} videos from split")
        
        # Feature directory
        self.feature_dir = os.path.join(path, 'Violence_five_crop_i3d_v1')

    def __len__(self):
        return len(self.video_list)

    def apply_augmentations(self, features):
        """Enhanced augmentation pipeline"""
        if not self.augment:
            return features
        
        # 1. Temporal dropout - randomly drop some temporal segments
        if random.random() < 0.3:
            features = self.temporal_dropout(features, drop_prob=0.1)
        
        # 2. Feature dropout - randomly zero out some features
        if random.random() < 0.5:
            features = self.feature_dropout(features, drop_prob=0.1)
        
        # 3. Gaussian noise injection
        if random.random() < 0.4:
            features = self.add_gaussian_noise(features, std=0.01)
        
        # 4. Temporal shift augmentation
        if random.random() < 0.3:
            features = self.temporal_shift(features)
        
        # 5. Feature normalization with random scaling
        if random.random() < 0.3:
            features = self.random_scale(features)
        
        return features

    def temporal_dropout(self, features, drop_prob=0.1):
        """Randomly drop temporal segments"""
        mask = torch.rand(features.shape[0]) > drop_prob
        if mask.sum() > 0:  # Ensure we don't drop everything
            features = features[mask]
        return features

    def feature_dropout(self, features, drop_prob=0.1):
        """Apply dropout to feature dimensions"""
        mask = torch.rand(features.shape[1]) > drop_prob
        features = features * mask.float()
        return features

    def add_gaussian_noise(self, features, std=0.01):
        """Add Gaussian noise to features"""
        noise = torch.randn_like(features) * std
        return features + noise

    def temporal_shift(self, features, max_shift=3):
        """Shift temporal features randomly"""
        shift = random.randint(-max_shift, max_shift)
        if shift > 0:
            features = torch.cat([features[shift:], features[:shift]], dim=0)
        elif shift < 0:
            features = torch.cat([features[shift:], features[:shift]], dim=0)
        return features

    def random_scale(self, features, scale_range=(0.9, 1.1)):
        """Randomly scale features"""
        scale = random.uniform(*scale_range)
        return features * scale

    def __getitem__(self, idx):
        video_id = self.video_list[idx]
        
        # Try multiple file naming conventions for XD-Violence
        feature_file = os.path.join(self.feature_dir, video_id + '_i3d.npy')
        
        if not os.path.exists(feature_file):
            feature_file = os.path.join(self.feature_dir, video_id + '.npy')
        
        if not os.path.exists(feature_file):
            if '_i3d' not in video_id:
                base_name = video_id.split('__')[0] if '__' in video_id else video_id
                feature_file = os.path.join(self.feature_dir, base_name + '_i3d.npy')
        
        if not os.path.exists(feature_file):
            import glob
            pattern = os.path.join(self.feature_dir, f"*{video_id}*.npy")
            matches = glob.glob(pattern)
            if matches:
                feature_file = matches[0]
            else:
                print(f"Warning: Feature file not found for {video_id}")
                dummy_features = np.random.randn(32, 5, 1024).astype(np.float32)
                if self.is_train == 1:
                    return torch.from_numpy(dummy_features.reshape(32, -1)).float()
                else:
                    return torch.from_numpy(dummy_features.reshape(32, -1)).float(), torch.zeros(32), 32
        
        features = np.load(feature_file)
        
        # XD-Violence features: [T, 5, 1024] -> [T, 5120]
        if len(features.shape) == 3:
            # Option 1: Concatenate crops (your current approach)
            # features = features.reshape(features.shape[0], -1)
            
            # Option 2: Average crops (often better for robustness)
            features = features.mean(axis=1)  # [T, 1024]
            
        # Convert to torch tensor first
        features = torch.from_numpy(features).float()
        
        # Apply augmentations (only in training)
        if self.augment and self.is_train == 1:
            features = self.apply_augmentations(features)
        
        # L2 normalization for better feature representation
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        if self.is_train == 1:
            return features
        else:
            gt_info = self.ground_truth.get(video_id, {})
            num_frames = gt_info.get('num_frames', features.shape[0])
            labels = gt_info.get('labels', [0] * features.shape[0])
            labels = torch.tensor(labels, dtype=torch.float32)
            
            return features, labels, num_frames


class Normal_Loader_XD(Dataset):
    """Enhanced Normal videos loader for XD-Violence"""
    def __init__(self, is_train=1, path='./xd_vio/', modality='RGB', augment=False):
        super(Normal_Loader_XD, self).__init__()
        self.is_train = is_train
        self.path = path
        self.modality = modality
        self.augment = augment
        
        json_path = os.path.join(path, 'List/xd-violence_ground_truth.json')
        with open(json_path, 'r') as f:
            ground_truth = json.load(f)
        
        if is_train == 1:
            csv_file = os.path.join(path, 'List/xd-violence.training.csv')
        else:
            csv_file = os.path.join(path, 'List/xd-violence.testing.csv')
        
        df = pd.read_csv(csv_file)
        all_videos = df['video-id'].tolist()
        
        self.data_list = []
        for video_id in all_videos:
            if '_label_A' in video_id:
                self.data_list.append(video_id)
            elif video_id in ground_truth:
                labels = ground_truth[video_id]['labels']
                if sum(labels) == 0:
                    self.data_list.append(video_id)
        
        print(f"Normal {'train' if is_train else 'test'} videos: {len(self.data_list)}")
        self.feature_dir = os.path.join(path, 'Violence_five_crop_i3d_v1')
    
    def __len__(self):
        return len(self.data_list)
    
    def apply_augmentations(self, features):
        """Same augmentations as main loader"""
        if not self.augment:
            return features
        
        if random.random() < 0.3:
            mask = torch.rand(features.shape[0]) > 0.1
            if mask.sum() > 0:
                features = features[mask]
        
        if random.random() < 0.5:
            mask = torch.rand(features.shape[1]) > 0.1
            features = features * mask.float()
        
        if random.random() < 0.4:
            noise = torch.randn_like(features) * 0.01
            features = features + noise
        
        return features
    
    def __getitem__(self, idx):
        video_id = self.data_list[idx]
        
        feature_file = os.path.join(self.feature_dir, video_id + '_i3d.npy')
        
        if not os.path.exists(feature_file):
            feature_file = os.path.join(self.feature_dir, video_id + '.npy')
        
        if not os.path.exists(feature_file):
            import glob
            pattern = os.path.join(self.feature_dir, f"*{video_id}*.npy")
            matches = glob.glob(pattern)
            if matches:
                feature_file = matches[0]
            else:
                print(f"Warning: Normal feature file not found for {video_id}")
                dummy_features = np.random.randn(32, 5, 1024).astype(np.float32)
                if self.is_train == 1:
                    return torch.from_numpy(dummy_features.mean(axis=1)).float()
                else:
                    return torch.from_numpy(dummy_features.mean(axis=1)).float(), torch.zeros(32), 32
        
        features = np.load(feature_file)
        
        # Average crops instead of concatenate
        if len(features.shape) == 3:
            features = features.mean(axis=1)
        
        features = torch.from_numpy(features).float()
        
        if self.augment and self.is_train == 1:
            features = self.apply_augmentations(features)
        
        # L2 normalization
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        if self.is_train == 1:
            return features
        else:
            json_path = os.path.join(self.path, 'List/xd-violence_ground_truth.json')
            with open(json_path, 'r') as f:
                ground_truth = json.load(f)
            
            gt_info = ground_truth.get(video_id, {})
            num_frames = gt_info.get('num_frames', features.shape[0])
            labels = gt_info.get('labels', [0] * features.shape[0])
            labels = torch.tensor(labels, dtype=torch.float32)
            
            return features, labels, num_frames


class Anomaly_Loader_XD(Dataset):
    """Enhanced Anomaly videos loader for XD-Violence"""
    def __init__(self, is_train=1, path='./xd_vio/', modality='RGB', augment=False):
        super(Anomaly_Loader_XD, self).__init__()
        self.is_train = is_train
        self.path = path
        self.modality = modality
        self.augment = augment
        
        json_path = os.path.join(path, 'List/xd-violence_ground_truth.json')
        with open(json_path, 'r') as f:
            ground_truth = json.load(f)
        
        if is_train == 1:
            csv_file = os.path.join(path, 'List/xd-violence.training.csv')
        else:
            csv_file = os.path.join(path, 'List/xd-violence.testing.csv')
        
        df = pd.read_csv(csv_file)
        all_videos = df['video-id'].tolist()
        
        self.data_list = []
        for video_id in all_videos:
            if '_label_A' not in video_id:
                self.data_list.append(video_id)
            elif video_id in ground_truth:
                labels = ground_truth[video_id]['labels']
                if sum(labels) > 0:
                    self.data_list.append(video_id)
        
        print(f"Anomaly {'train' if is_train else 'test'} videos: {len(self.data_list)}")
        self.feature_dir = os.path.join(path, 'Violence_five_crop_i3d_v1')
    
    def __len__(self):
        return len(self.data_list)
    
    def apply_augmentations(self, features):
        """Same augmentations as main loader"""
        if not self.augment:
            return features
        
        if random.random() < 0.3:
            mask = torch.rand(features.shape[0]) > 0.1
            if mask.sum() > 0:
                features = features[mask]
        
        if random.random() < 0.5:
            mask = torch.rand(features.shape[1]) > 0.1
            features = features * mask.float()
        
        if random.random() < 0.4:
            noise = torch.randn_like(features) * 0.01
            features = features + noise
        
        if random.random() < 0.3:
            max_shift = 3
            shift = random.randint(-max_shift, max_shift)
            if shift != 0:
                features = torch.roll(features, shifts=shift, dims=0)
        
        return features
    
    def __getitem__(self, idx):
        video_id = self.data_list[idx]
        
        feature_file = os.path.join(self.feature_dir, video_id + '_i3d.npy')
        
        if not os.path.exists(feature_file):
            feature_file = os.path.join(self.feature_dir, video_id + '.npy')
        
        if not os.path.exists(feature_file):
            import glob
            pattern = os.path.join(self.feature_dir, f"*{video_id}*.npy")
            matches = glob.glob(pattern)
            if matches:
                feature_file = matches[0]
            else:
                print(f"Warning: Anomaly feature file not found for {video_id}")
                dummy_features = np.random.randn(32, 5, 1024).astype(np.float32)
                if self.is_train == 1:
                    return torch.from_numpy(dummy_features.mean(axis=1)).float()
                else:
                    return torch.from_numpy(dummy_features.mean(axis=1)).float(), torch.zeros(32), 32
        
        features = np.load(feature_file)
        
        # Average crops instead of concatenate
        if len(features.shape) == 3:
            features = features.mean(axis=1)
        
        features = torch.from_numpy(features).float()
        
        if self.augment and self.is_train == 1:
            features = self.apply_augmentations(features)
        
        # L2 normalization
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        if self.is_train == 1:
            return features
        else:
            json_path = os.path.join(self.path, 'List/xd-violence_ground_truth.json')
            with open(json_path, 'r') as f:
                ground_truth = json.load(f)
            
            gt_info = ground_truth.get(video_id, {})
            num_frames = gt_info.get('num_frames', features.shape[0])
            labels = gt_info.get('labels', [0] * features.shape[0])
            labels = torch.tensor(labels, dtype=torch.float32)
            
            return features, labels, num_frames