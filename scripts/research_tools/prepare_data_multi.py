import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pdb_loader import PDBLoader, center_and_scale
from data.ensemble_generator import prepare_ensemble_dataset

def process_protein(pdb_id):
    print(f"Processing {pdb_id}...")
    base_dir = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw_pdb")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    loader = PDBLoader(data_dir=raw_dir)
    file_path = loader.download_pdb(pdb_id)
    if not file_path:
        return None

    raw_coords = loader.extract_backbone(file_path, atoms=['CA'])
    print(f"Extracted {len(raw_coords)} C-alpha atoms.")

    coords, centroid, scale = center_and_scale(raw_coords)
    
    pt_path = os.path.join(processed_dir, f"{pdb_id}_ca.pt")
    torch.save({
        'coords': torch.from_numpy(coords),
        'centroid': torch.from_numpy(centroid),
        'scale': scale,
        'name': pdb_id,
        'atom_type': 'CA'
    }, pt_path)
    
    # Generate ensemble
    ensemble_path = os.path.join(processed_dir, f"{pdb_id}_ensemble.pt")
    prepare_ensemble_dataset(pt_path, ensemble_path, n_conformers=500, n_modes=10, amplitude_scale=0.3)
    
    return pt_path

if __name__ == '__main__':
    # Ubiquitin (already done, but for completeness)
    # process_protein('1ubq')
    # Lysozyme (HEWL)
    process_protein('1hel')
