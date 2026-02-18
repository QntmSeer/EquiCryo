"""
Dataset classes for 3D protein structure diffusion model.

Supports:
1. Single structure (with noise augmentation) — original mode
2. Conformational ensemble — research-grade mode (Upgrade 1)
"""

import torch
from torch.utils.data import Dataset
import os


class ProteinStructureDataset(Dataset):
    """Original dataset: single structure with noise augmentation."""
    
    def __init__(self, data_path, augment_noise=0.0):
        try:
            self.data = torch.load(data_path, weights_only=False)
        except TypeError:
            self.data = torch.load(data_path)

        self.coords = self.data['coords']  # (N, 3)
        self.augment_noise = augment_noise
        self.epoch_len = 1000

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        x = self.coords.clone()
        if self.augment_noise > 0:
            x = x + torch.randn_like(x) * self.augment_noise
        return x


class ConformationalEnsembleDataset(Dataset):
    """
    Ensemble dataset: samples from a distribution of conformers.
    
    This is the key upgrade — instead of memorizing one structure,
    the diffusion model learns p(x_3D) over a manifold of conformations.
    """
    
    def __init__(self, data_path, augment_noise=0.005):
        """
        Args:
            data_path: path to ensemble .pt file
            augment_noise: additional noise on top of ensemble variation
        """
        try:
            self.data = torch.load(data_path, weights_only=False)
        except TypeError:
            self.data = torch.load(data_path)
        
        self.ensemble = self.data['ensemble']  # (M, N, 3)
        self.reference = self.data.get('reference', None)
        self.n_conformers = self.ensemble.shape[0]
        self.augment_noise = augment_noise
        
        print(f"Loaded ensemble: {self.n_conformers} conformers, "
              f"{self.ensemble.shape[1]} atoms each")

    def __len__(self):
        return self.n_conformers

    def __getitem__(self, idx):
        x = self.ensemble[idx].clone()
        if self.augment_noise > 0:
            x = x + torch.randn_like(x) * self.augment_noise
        return x


class MultiProteinDataset(Dataset):
    """
    Research Upgrade: Dataset that combines multiple protein ensembles.
    Handles varying atom counts via padding.
    """
    def __init__(self, data_paths, augment_noise=0.005):
        self.datasets = [ConformationalEnsembleDataset(p, augment_noise) for p in data_paths]
        self.lengths = [len(d) for d in self.datasets]
        self.total_len = sum(self.lengths)
        self.max_atoms = max([d.ensemble.shape[1] for d in self.datasets])
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # find which dataset
        curr = 0
        for d in self.datasets:
            if idx < curr + len(d):
                x = d[idx - curr]
                n_atoms = x.shape[0]
                # Pad to max_atoms
                # (N, 3) -> (max_N, 3)
                padded_x = torch.zeros((self.max_atoms, 3))
                padded_x[:n_atoms] = x
                mask = torch.zeros(self.max_atoms, dtype=torch.bool)
                mask[:n_atoms] = True
                return padded_x, mask
            curr += len(d)

def collate_fn(batch):
    """Custom collate for padded proteins."""
    coords = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    return coords, masks
