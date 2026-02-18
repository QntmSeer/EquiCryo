import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projection.projector import CryoProjector

def visualize_ctf():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    # Load Lysozyme
    data_path = os.path.join(base_dir, "data", "processed", "1hel_ca.pt")
    data = torch.load(data_path, weights_only=False)
    x = data['coords'].to(device).unsqueeze(0)
    
    # Projectors
    # Case 1: No CTF
    projector_clean = CryoProjector(output_size=(128, 128), pixel_size=1.0).to(device)
    # Case 2: With CTF (defocus 2um)
    projector_ctf = CryoProjector(output_size=(128, 128), pixel_size=1.0, defocus=2.0).to(device)
    
    # Fixed rotation
    rot = projector_clean.random_rotation_matrix(1, device=device)
    
    with torch.no_grad():
        # Compute CTF kernel for visualization
        ctf_kernel = projector_ctf.compute_ctf(device).cpu().numpy()
        
        # Clean projection
        proj_clean = projector_clean.project(x, rot).squeeze(0).cpu().numpy()
        # CTF projection
        proj_ctf = projector_ctf.project(x, rot).squeeze(0).cpu().numpy()
        
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(proj_clean, cmap='gray')
    axes[0].set_title("Clean Projection")
    plt.colorbar(im0, ax=axes[0])
    
    # Plot CTF kernel (Fourier space)
    im1 = axes[1].imshow(np.fft.fftshift(ctf_kernel), cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title("CTF Kernel (Fourier Space)")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(proj_ctf, cmap='gray')
    axes[2].set_title("Projection + CTF (Defocus 2um)")
    plt.colorbar(im2, ax=axes[2])
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    os.makedirs(os.path.join(base_dir, "assets"), exist_ok=True)
    out_path = os.path.join(base_dir, "assets", "ctf_visualization.png")
    plt.savefig(out_path, dpi=150)
    print(f"CTF visualization saved to {out_path}")

if __name__ == '__main__':
    visualize_ctf()
