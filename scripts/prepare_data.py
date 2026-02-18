import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pdb_loader import PDBLoader, center_and_scale

def prepare_ubiquitin():
    print("Initializing Data Preparation for Ubiquitin...")
    
    # Setup paths
    base_dir = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw_pdb")
    processed_dir = os.path.join(data_dir, "processed")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Download and Load
    loader = PDBLoader(data_dir=raw_dir)
    file_path = loader.download_pdb("1ubq")
    
    # 2. Extract C-alpha backbone
    raw_coords = loader.extract_backbone(file_path, atoms=['CA'])
    print(f"Extracted {len(raw_coords)} C-alpha atoms.")
    
    # 3. Preprocess (Center & Scale)
    coords, centroid, scale = center_and_scale(raw_coords)
    
    # 4. Save
    output_path = os.path.join(processed_dir, "ubiquitin_ca.pt")
    torch.save({
        'coords': torch.from_numpy(coords),
        'centroid': torch.from_numpy(centroid),
        'scale': scale,
        'name': '1ubq',
        'atom_type': 'CA'
    }, output_path)
    
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    prepare_ubiquitin()
