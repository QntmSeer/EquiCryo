import sys
import os
import torch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projection.projector import CryoProjector

def generate_projections():
    # Load data
    data_path = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior\data\processed\ubiquitin_ca.pt"
    try:
        data = torch.load(data_path, weights_only=False)
    except TypeError:
        data = torch.load(data_path)
    
    coords = data['coords'] # (N, 3)
    
    # Init Projector
    # Ubiquitin is roughly 30-40 Angstroms diameter.
    # Output size 64x64 with 1 Angstrom/pixel covers it well.
    projector = CryoProjector(output_size=(64, 64), pixel_size=1.0, sigma_noise=0.5)
    
    # Generate batch of projections
    # Input expects (B, N, 3).
    x = coords.unsqueeze(0) # (1, N, 3)
    
    # Generate 4 random views
    noisy_proj, rot_matrices, clean_proj = projector(x, num_projections=4)
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        # Clean
        axes[0, i].imshow(clean_proj[i].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f"Clean View {i+1}")
        axes[0, i].axis('off')
        
        # Noisy
        axes[1, i].imshow(noisy_proj[i].cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f"Noisy View {i+1}")
        axes[1, i].axis('off')
        
    os.makedirs("experiments/projections", exist_ok=True)
    plt.tight_layout()
    plt.savefig("experiments/projections/synthetic_data.png")
    print("Saved projection visualization to experiments/projections/synthetic_data.png")

if __name__ == "__main__":
    generate_projections()
