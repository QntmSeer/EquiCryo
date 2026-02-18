import torch
import matplotlib.pyplot as plt
import os
import sys

def visualize_structure(file_path):
    # Load data
    try:
        data = torch.load(file_path, weights_only=False)
    except TypeError:
        # Fallback for older pytorch versions
        data = torch.load(file_path)
    coords = data['coords'].numpy()
    name = data.get('name', 'Unknown')
    
    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='b', marker='o', s=20, alpha=0.6)
    
    # Draw lines connecting consecutive backbone atoms (C-alpha trace)
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], c='k', alpha=0.5)
    
    ax.set_title(f"Processed Structure: {name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Ensure equal aspect ratio for correct structural perception
    max_range = np.array([coords[:, 0].max()-coords[:, 0].min(), 
                          coords[:, 1].max()-coords[:, 1].min(), 
                          coords[:, 2].max()-coords[:, 2].min()]).max() / 2.0
    
    mid_x = (coords[:, 0].max()+coords[:, 0].min()) * 0.5
    mid_y = (coords[:, 1].max()+coords[:, 1].min()) * 0.5
    mid_z = (coords[:, 2].max()+coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    output_path = file_path.replace('.pt', '.png')
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    import numpy as np
    file_path = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior\data\processed\ubiquitin_ca.pt"
    if os.path.exists(file_path):
        visualize_structure(file_path)
    else:
        print(f"File not found: {file_path}")
