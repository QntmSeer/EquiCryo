
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_rg(x):
    """
    Compute Radius of Gyration (Rg).
    Rg = sqrt(mean( ||x_i - center||^2 ))
    """
    if x.ndim == 2:
        x = x.unsqueeze(0)
    center = torch.mean(x, dim=1, keepdim=True)
    sq_dist = torch.sum((x - center)**2, dim=2)
    rg = torch.sqrt(torch.mean(sq_dist))
    return rg.item()

def check_data_scale():
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    # 1. Check Training Data (Lysozyme Ensemble)
    train_path = os.path.join(base, "data", "processed", "1hel_ensemble.pt")
    if os.path.exists(train_path):
        print(f"Checking Training Ensemble: {train_path}")
        try:
            data = torch.load(train_path, weights_only=False)
        except:
            data = torch.load(train_path)
            
        ensemble = data['ensemble'] # (M, N, 3)
        print(f"  Shape: {ensemble.shape}")
        
        # Stats across all atoms/conformers
        all_coords = ensemble.reshape(-1, 3)
        mean = all_coords.mean(dim=0)
        std = all_coords.std()
        max_val = all_coords.abs().max()
        
        print(f"  Global Mean: {mean}")
        print(f"  Global Std:  {std:.4f}")
        print(f"  Global Max:  {max_val:.4f}")
        
        # Average Rg
        rgs = [compute_rg(x) for x in ensemble]
        avg_rg = np.mean(rgs)
        print(f"  Average Rg:  {avg_rg:.4f}")
    else:
        print(f"Training data not found at {train_path}")

    # 2. Check Inference Target (Ground Truth used in reconstruction.py)
    gt_path = os.path.join(base, "data", "processed", "1hel_ca.pt")
    if os.path.exists(gt_path):
        print(f"\nChecking Inference Target: {gt_path}")
        try:
            data = torch.load(gt_path, weights_only=False)
        except:
            data = torch.load(gt_path)
            
        coords = data['coords'] # (N, 3)
        print(f"  Shape: {coords.shape}")
        
        mean = coords.mean(dim=0)
        std = coords.std()
        max_val = coords.abs().max()
        rg = compute_rg(coords)
        
        print(f"  Mean: {mean}")
        print(f"  Std:  {std:.4f}")
        print(f"  Max:  {max_val:.4f}")
        print(f"  Rg:   {rg:.4f}")
    else:
        print(f"Inference target not found at {gt_path}")

if __name__ == "__main__":
    check_data_scale()
