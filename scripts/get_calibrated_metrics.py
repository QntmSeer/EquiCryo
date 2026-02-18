
import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import PointDiffusionTransformer, DiffusionModel
from projection.projector import CryoProjector
from inference.reconstruction import reconstruct_dps, compute_radius_of_gyration, compute_rmsd

def get_calibrated_metrics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    # Load Model
    net = PointDiffusionTransformer(hidden_dim=128, num_layers=4)
    model = DiffusionModel(net, timesteps=1000).to(device)
    ckpt_path = os.path.join(base, "experiments", "checkpoints", "ddpm_multi_protein.pth")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    except:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Load GT
    data_path = os.path.join(base, "data", "processed", "1hel_ca.pt")
    try:
        data = torch.load(data_path, weights_only=False)
    except:
        data = torch.load(data_path)
    x_gt = data['coords'].to(device).unsqueeze(0)
    n_atoms = x_gt.shape[1]
    
    # Setup Projector
    sigma = 0.05
    projector = CryoProjector(output_size=(64, 64), sigma_noise=sigma).to(device)
    rot = projector.random_rotation_matrix(1, device=device)
    with torch.no_grad():
        y = projector.project(x_gt, rot) + torch.randn(1, 64, 64, device=device) * sigma
        
    print(f"Running calibrated reconstruction (scale=1.59)...")
    x_rec, _, losses = reconstruct_dps(model, projector, y, rot, device, 
                                       step_size=0.1, # Using a conservative alpha for best quality
                                       n_atoms=n_atoms, known_pose=True,
                                       coordinate_scale=1.59)
    
    rg = compute_radius_of_gyration(x_rec)
    rmsd = compute_rmsd(x_rec, x_gt)
    final_loss = losses[-1] if losses else 0.0
    
    print(f"METRICS:")
    print(f"Rg: {rg:.4f}")
    print(f"RMSD: {rmsd:.4f}")
    print(f"Loss: {final_loss:.6f}")

if __name__ == "__main__":
    get_calibrated_metrics()
