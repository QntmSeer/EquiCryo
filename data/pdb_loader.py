import os
import urllib.request
import numpy as np
import torch

class PDBLoader:
    def __init__(self, data_dir="data/pdb"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_pdb(self, pdb_id):
        """Downloads a PDB file from RCSB."""
        pdb_id = pdb_id.lower()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        filepath = os.path.join(self.data_dir, f"{pdb_id}.pdb")
        
        if not os.path.exists(filepath):
            print(f"Downloading {pdb_id} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Saved to {filepath}")
            except Exception as e:
                print(f"Failed to download {pdb_id}: {e}")
                return None
        else:
            print(f"File {filepath} already exists.")
        
        return filepath

    def extract_backbone(self, filepath, atoms=['CA']):
        """
        Extracts coordinates of specified atoms (default C-alpha) from the first model.
        Returns a numpy array of shape (N, 3).
        """
        coords = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    if atom_name in atoms:
                        # PDB format: x=30-38, y=38-46, z=46-54
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                        except ValueError:
                            continue
                elif line.startswith("ENDMDL"):
                    # Stop after the first model
                    break
        
        return np.array(coords, dtype=np.float32)

def center_and_scale(coords):
    """
    Centers the point cloud at the origin and scales it to fit within a unit sphere.
    """
    if len(coords) == 0:
        raise ValueError("No coordinates found to process.")

    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    
    # Scale so that the maximum distance from origin is 1.0
    max_norm = np.max(np.linalg.norm(centered, axis=1))
    if max_norm == 0:
        max_norm = 1.0 # Handle single atom or zero vector case
        
    scaled = centered / max_norm
    
    return scaled, centroid, max_norm

if __name__ == "__main__":
    # Test
    loader = PDBLoader(data_dir=r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior\data\raw_pdb")
    pdb_path = loader.download_pdb("1ubq")
    coords = loader.extract_backbone(pdb_path)
    print(f"Extracted {len(coords)} atoms.")
    
    processed, centroid, scale = center_and_scale(coords)
    print(f"Centered and scaled. Max norm: {np.max(np.linalg.norm(processed, axis=1))}")
