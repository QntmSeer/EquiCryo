
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import PointDiffusionTransformer, DiffusionModel
from projection.projector import CryoProjector
from inference.reconstruction import reconstruct_dps, compute_radius_of_gyration

def kabsch_rmsd(P, Q):
    """
    Compute RMSD between two sets of points P and Q using Kabsch algorithm.
    P, Q: (N, 3) tensors.
    """
    # Center the points
    P_centered = P - P.mean(dim=0, keepdim=True)
    Q_centered = Q - Q.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    H = torch.matmul(P_centered.T, Q_centered)

    # SVD
    U, S, V = torch.svd(H)

    # Compute rotation
    d = torch.sign(torch.det(torch.matmul(V, U.T)))
    diag = torch.ones(3, device=P.device)
    diag[2] = d
    R = torch.matmul(torch.matmul(V, torch.diag(diag)), U.T)

    # Rotate P
    P_rotated = torch.matmul(P_centered, R.T)

    # Compute RMSD
    diff = P_rotated - Q_centered
    rmsd = torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1)))
    return rmsd.item()

def generate_ablation_table():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    # Load Model & Data
    net = PointDiffusionTransformer(hidden_dim=128, num_layers=4)
    model = DiffusionModel(net, timesteps=1000).to(device)
    try:
        model.load_state_dict(torch.load(os.path.join(base, "experiments", "checkpoints", "ddpm_multi_protein.pth"), map_location=device, weights_only=False))
    except:
        model.load_state_dict(torch.load(os.path.join(base, "experiments", "checkpoints", "ddpm_multi_protein.pth"), map_location=device))
    model.eval()
    
    data = torch.load(os.path.join(base, "data", "processed", "1hel_ca.pt"), weights_only=False)
    x_gt = data['coords'].to(device).unsqueeze(0)
    n_atoms = x_gt.shape[1]
    
    # Projector
    projector = CryoProjector(output_size=(64, 64), sigma_noise=0.05).to(device)
    rot = projector.random_rotation_matrix(1, device=device)
    with torch.no_grad():
        y = projector.project(x_gt, rot) + torch.randn(1, 64, 64, device=device) * 0.05

    alphas = [0.01, 0.1, 1.0]
    coord_scale = 1.59
    
    print("\n| Alpha | Rg | Aligned RMSD |")
    print("|---|---|---|")
    
    for alpha in alphas:
        x_rec, _, _ = reconstruct_dps(model, projector, y, rot, device, 
                                      step_size=alpha, n_atoms=n_atoms, known_pose=True,
                                      coordinate_scale=coord_scale)
        
        rg = compute_radius_of_gyration(x_rec)
        rmsd = kabsch_rmsd(x_rec.squeeze(), x_gt.squeeze())
        
        print(f"| {alpha} | {rg:.3f} | {rmsd:.3f} |")

if __name__ == "__main__":
    generate_ablation_table()
