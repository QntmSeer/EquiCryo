"""
Bayesian Reconstruction with Joint Pose + Structure Inference
==============================================================

Upgrade 2: Instead of assuming known rotation R, we jointly optimize
both the 3D structure x and the viewing angle R.

Two algorithms:
1. DPS with known pose (original, for ablation)
2. DPS with alternating pose optimization (research-grade)

Upgrade 3: Quantitative evaluation suite
- RMSD vs noise level
- RMSD vs number of projections
- Ablation: with prior vs without prior
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import PointDiffusionTransformer, DiffusionModel
from projection.projector import CryoProjector


# ============================================================
# Utility: Differentiable Rotation from Axis-Angle
# ============================================================

def axis_angle_to_rotation_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix (Rodrigues' formula).
    Fully differentiable for gradient-based pose optimization.
    
    Args:
        axis_angle: (B, 3) tensor
    Returns:
        rotation_matrix: (B, 3, 3) tensor
    """
    theta = torch.norm(axis_angle, dim=-1, keepdim=True)  # (B, 1)
    theta = theta.clamp(min=1e-8)
    k = axis_angle / theta  # unit axis (B, 3)
    
    K = torch.zeros(axis_angle.shape[0], 3, 3, device=axis_angle.device)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    
    # Rodrigues: R = I + sin(theta)*K + (1-cos(theta))*K^2
    I = torch.eye(3, device=axis_angle.device).unsqueeze(0)
    theta = theta.unsqueeze(-1)  # (B, 1, 1)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.bmm(K, K)
    
    return R


# ============================================================
# Core Reconstruction: DPS (Diffusion Posterior Sampling)
# ============================================================

def reconstruct_dps(model, projector, y_meas, rot_matrix, device,
                    step_size=1.0, n_atoms=76, known_pose=True,
                    pose_lr=0.01, pose_steps_per_t=3, coordinate_scale=1.0):
    """
    Reconstruct 3D structure from 2D projections using DPS.
    
    Args:
        model: trained DiffusionModel
        projector: CryoProjector
        y_meas: (K, H, W) measured projections (K views)
        rot_matrix: (K, 3, 3) true rotations (used if known_pose=True, otherwise initial guess)
        device: torch device
        step_size: guidance step size (scaled by 1/||grad||)
        n_atoms: number of atoms
        known_pose: if True, use ground truth rotation; if False, optimize jointly
        pose_lr: learning rate for pose optimization
        pose_steps_per_t: number of pose optimization steps per diffusion step
        coordinate_scale: Scale factor to map latent coords to physical coords (x_phys = x_latent * scale)
    """
    K = y_meas.shape[0]  # number of projections

    # Initialize structure from noise
    x = torch.randn(1, n_atoms, 3, device=device)
    
    # Initialize pose parameters (axis-angle)
    if not known_pose:
        # Random initialization for pose
        pose_params = torch.randn(K, 3, device=device) * 0.1
        pose_params.requires_grad_(True)
        pose_optimizer = optim.Adam([pose_params], lr=pose_lr)
    
    losses = []
    
    for i in tqdm(reversed(range(0, model.timesteps)), total=model.timesteps, 
                  desc="DPS Reconstruction"):
        t_tensor = torch.full((1,), i, device=device, dtype=torch.long)
        
        # --- Pose optimization (Upgrade 2) ---
        if not known_pose and i < model.timesteps * 0.8:
            # Only optimize pose after initial diffusion steps (let structure stabilize first)
            for _ in range(pose_steps_per_t):
                pose_optimizer.zero_grad()
                R_est = axis_angle_to_rotation_matrix(pose_params)
                
                x_det = x.detach() * coordinate_scale # Scale for physical projection
                y_hat_all = []
                for k in range(K):
                    y_hat_k = projector.project(x_det, R_est[k:k+1])
                    y_hat_all.append(y_hat_k)
                y_hat = torch.cat(y_hat_all, dim=0)
                
                pose_loss = F.mse_loss(y_hat, y_meas)
                pose_loss.backward()
                pose_optimizer.step()
        
        # --- Structure update via DPS ---
        x = x.detach().requires_grad_(True)
        
        epsilon_theta = model.model(x, t_tensor)
        
        alpha_bar_t = model.alphas_cumprod[i]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
        
        x_0_hat = (x - sqrt_one_minus_alpha_bar * epsilon_theta) / sqrt_alpha_bar
        
        # Compute measurement loss across all projections
        if known_pose:
            R_current = rot_matrix
        else:
            R_current = axis_angle_to_rotation_matrix(pose_params.detach())
        
        total_measure_loss = torch.tensor(0.0, device=device)
        for k in range(K):
            # Scale x_0_hat to physical space for projection
            y_hat_k = projector.project(x_0_hat * coordinate_scale, R_current[k:k+1])
            total_measure_loss = total_measure_loss + F.mse_loss(y_hat_k, y_meas[k:k+1])
        
        # Gradient w.r.t. x (Latent Space)
        grad = torch.autograd.grad(total_measure_loss, x)[0]
        
        # Normalize gradient (DPS-style)
        grad_norm = torch.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
        
        # DDPM reverse step
        beta_t = model.betas[i]
        alpha_t = 1 - beta_t
        sigma_t = torch.sqrt(beta_t)
        
        coeff1 = 1 / torch.sqrt(alpha_t)
        coeff2 = beta_t / sqrt_one_minus_alpha_bar
        
        mean = coeff1 * (x.detach() - coeff2 * epsilon_theta.detach())
        mean = mean - step_size * grad
        
        if i > 0:
            noise = torch.randn_like(x).detach()
            x = mean + sigma_t * noise
        else:
            x = mean
        
        losses.append(total_measure_loss.item())
    
    R_final = R_current if known_pose else axis_angle_to_rotation_matrix(pose_params.detach())
    return x.detach() * coordinate_scale, R_final.detach(), losses


# ============================================================
# Reconstruction WITHOUT prior (baseline for ablation)
# ============================================================

def reconstruct_no_prior(projector, y_meas, rot_matrix, device,
                         n_atoms=76, n_iterations=2000, lr=0.01):
    """
    Reconstruct 3D structure from 2D projections WITHOUT a learned prior.
    """
    K = y_meas.shape[0]
    
    x = torch.randn(1, n_atoms, 3, device=device) * 0.1
    x.requires_grad_(True)
    optimizer = optim.Adam([x], lr=lr)
    
    losses = []
    for it in range(n_iterations):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        for k in range(K):
            y_hat = projector.project(x, rot_matrix[k:k+1])
            total_loss = total_loss + F.mse_loss(y_hat, y_meas[k:k+1])
        
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
    
    return x.detach(), losses


# ============================================================
# Evaluation Metrics
# ============================================================

def compute_rmsd(x_pred, x_gt):
    """Compute RMSD between predicted and ground truth structures."""
    diff = x_pred - x_gt
    return torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1))).item()


# ============================================================
# Main Experiment Runner
# ============================================================

# ============================================================
# Collapse Diagnostics (Scientific Integrity)
# ============================================================

def compute_radius_of_gyration(x):
    """
    Compute Radius of Gyration (Rg).
    Rg = sqrt(mean( ||x_i - center||^2 ))
    """
    center = torch.mean(x, dim=1, keepdim=True)
    sq_dist = torch.sum((x - center)**2, dim=2)
    rg = torch.sqrt(torch.mean(sq_dist))
    return rg.item()

def compute_pairwise_distance_distribution(x, n_bins=50, max_dist=2.0):
    """
    Compute histogram of pairwise distances.
    Returns: (density, bin_centers)
    """
    B, N, D = x.shape
    diff = x.unsqueeze(2) - x.unsqueeze(1) # (B, N, N, 3)
    dist = torch.norm(diff, dim=-1) # (B, N, N)
    
    # Get upper triangle to avoid double counting and self-dist
    mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
    dists_flat = dist[:, mask].flatten()
    
    hist = torch.histc(dists_flat, bins=n_bins, min=0, max=max_dist)
    density = hist / hist.sum()
    bin_edges = torch.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return density.cpu().numpy(), bin_centers.cpu().numpy()

# ============================================================
# Main Experiment Runner
# ============================================================

def run_full_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on {device}")
    
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    # Load trained model (Upgraded to Equivariant architecture)
    net = PointDiffusionTransformer(hidden_dim=128, num_layers=4)
    model = DiffusionModel(net, timesteps=1000).to(device)
    
    # Try multi-protein checkpoint
    ckpt_path = os.path.join(base, "experiments", "checkpoints", "ddpm_multi_protein.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(base, "experiments", "checkpoints", "ddpm_ensemble.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found. Train ensemble model first.")
        return
    
    print(f"Loading prior from {ckpt_path}")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    except (TypeError, RuntimeError):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Load ground truth - Lysozyme
    data_path = os.path.join(base, "data", "processed", "1hel_ca.pt")
    try:
        data = torch.load(data_path, weights_only=False)
    except TypeError:
        data = torch.load(data_path)
    x_gt = data['coords'].to(device).unsqueeze(0)
    n_atoms = x_gt.shape[1]
    
    rg_gt = compute_radius_of_gyration(x_gt)
    print(f"Ground Truth Rg: {rg_gt:.4f}")
    
    # Projector
    sigma_fixed = 0.1
    projector = CryoProjector(output_size=(64, 64), sigma_noise=sigma_fixed, defocus=2.0).to(device)
    os.makedirs(os.path.join(base, "experiments", "results"), exist_ok=True)
    
    print(f"Evaluating Reconstruction on {data['name']} ({n_atoms} atoms)...")
    
    # ========================================
    # Experiment 1: Baseline Degeneracy Check (Rg vs N Projections)
    # ========================================
    print("\n=== Experiment 1: Collapse Diagnostics (Rg vs # Projections) ===")
    n_proj_list = [1, 2, 5, 20]
    
    rg_prior_list = []
    rg_noprior_list = []
    
    # For PDD plot (use 1 projection case)
    pdd_gt, pdd_bins = compute_pairwise_distance_distribution(x_gt)
    pdd_prior = None
    pdd_noprior = None
    
    for n_proj in n_proj_list:
        print(f"\nNumber of projections: {n_proj}")
        rots = projector.random_rotation_matrix(n_proj, device=device)
        with torch.no_grad():
            x_expanded = x_gt.expand(n_proj, -1, -1) # Fix: Correct expansion
            y = projector.project(x_expanded, rots)
            y = y + torch.randn_like(y) * sigma_fixed
        
        # With prior
        x_r, _, _ = reconstruct_dps(model, projector, y, rots, device,
                                     step_size=0.01, n_atoms=n_atoms, known_pose=True)
        rg_p = compute_radius_of_gyration(x_r)
        rg_prior_list.append(rg_p)
        
        # Without prior
        x_np, _ = reconstruct_no_prior(projector, y, rots, device, n_atoms=n_atoms, n_iterations=1000)
        rg_np = compute_radius_of_gyration(x_np)
        rg_noprior_list.append(rg_np)
        
        print(f"  Rg (GT): {rg_gt:.3f}")
        print(f"  Rg (Prior): {rg_p:.3f}")
        print(f"  Rg (No Prior): {rg_np:.3f}")
        
        if n_proj == 1:
            pdd_prior, _ = compute_pairwise_distance_distribution(x_r)
            pdd_noprior, _ = compute_pairwise_distance_distribution(x_np)

    # ========================================
    # Generate Scientific Figures
    # ========================================
    print("\n=== Generating Figures ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Figure 1: Radius of Gyration Comparison
    x_idx = np.arange(len(n_proj_list))
    width = 0.35
    
    axes[0].bar(x_idx - width/2, rg_prior_list, width, label='Prior', color='#2196F3', alpha=0.8)
    axes[0].bar(x_idx + width/2, rg_noprior_list, width, label='No Prior', color='#F44336', alpha=0.8)
    axes[0].axhline(y=rg_gt, color='k', linestyle='--', linewidth=2, label='Ground Truth')
    
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(n_proj_list)
    axes[0].set_xlabel('Number of Projections', fontsize=12)
    axes[0].set_ylabel('Radius of Gyration (Rg)', fontsize=12)
    axes[0].set_title('Structural Collapse Detection (Rg)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Figure 2: Pairwise Distance Distribution (PDD)
    axes[1].plot(pdd_bins, pdd_gt, 'k-', linewidth=3, label='Ground Truth')
    axes[1].plot(pdd_bins, pdd_prior, '-', color='#2196F3', linewidth=2, label='Prior (1 View)')
    axes[1].plot(pdd_bins, pdd_noprior, '--', color='#F44336', linewidth=2, label='No Prior (1 View)')
    
    axes[1].set_xlabel('Pairwise Distance (Normalized)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Internal Geometry Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(base, "experiments", "results", "collapse_diagnostics.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved diagnostics figure to {fig_path}")

if __name__ == "__main__":
    run_full_evaluation()
