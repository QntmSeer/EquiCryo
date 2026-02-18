import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import PointDiffusionTransformer, DiffusionModel
from projection.projector import CryoProjector

def verify_equivariance():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    net = PointDiffusionTransformer(hidden_dim=64, num_layers=2).to(device)
    net.eval()
    
    # Random point cloud
    B, N = 1, 10
    x = torch.randn(B, N, 3, device=device)
    t = torch.zeros(B, device=device).long()
    
    # Get initial score
    with torch.no_grad():
        score1 = net(x, t)
    
    # Apply random rotation R
    projector = CryoProjector()
    R = projector.random_rotation_matrix(B, device=device)
    
    x_rot = torch.bmm(R, x.transpose(1, 2)).transpose(1, 2)
    
    # Get score of rotated input
    with torch.no_grad():
        score_rot = net(x_rot, t)
    
    # Rotate the original score
    score1_rot = torch.bmm(R, score1.transpose(1, 2)).transpose(1, 2)
    
    # Check if Score(R*x) == R*Score(x)
    diff = torch.norm(score_rot - score1_rot)
    print(f"Equivariance Error (Norm of diff): {diff.item():.6f}")
    
    if diff.item() < 1e-4:
        print("SUCCESS: Model is SE(3)-equivariant!")
    else:
        print("FAILURE: Model is NOT equivariant.")

if __name__ == '__main__':
    verify_equivariance()
