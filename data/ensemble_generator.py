"""
Conformational Ensemble Generator
==================================
Generates a synthetic ensemble of protein conformations from a single 
reference structure using Normal Mode Analysis (NMA)-like perturbations.

This moves the project from "memorization of one structure" to 
"learning a structural manifold" — a critical upgrade for generative modeling.

Methods:
1. Coarse-grained NMA via the Elastic Network Model (ENM)
2. Random linear combinations of low-frequency modes
"""

import numpy as np
import torch
import os
from scipy.spatial.distance import pdist, squareform


def build_hessian_enm(coords, cutoff=7.0, gamma=1.0):
    """
    Build the Hessian matrix for the Anisotropic Network Model (ANM).
    
    Args:
        coords: (N, 3) numpy array of C-alpha coordinates
        cutoff: distance cutoff in Angstroms for spring connections
        gamma: spring constant
    
    Returns:
        hessian: (3N, 3N) numpy array
    """
    N = len(coords)
    hessian = np.zeros((3*N, 3*N))
    
    for i in range(N):
        for j in range(i+1, N):
            diff = coords[j] - coords[i]
            dist = np.linalg.norm(diff)
            
            if dist < cutoff:
                # Spring constant weighted by distance
                k = -gamma / (dist**2)
                
                # 3x3 sub-matrix
                for a in range(3):
                    for b in range(3):
                        val = k * diff[a] * diff[b]
                        hessian[3*i+a, 3*j+b] = val
                        hessian[3*j+b, 3*i+a] = val
                        hessian[3*i+a, 3*i+b] -= val
                        hessian[3*j+a, 3*j+b] -= val
    
    return hessian


def compute_normal_modes(coords, cutoff=7.0, n_modes=20):
    """
    Compute normal modes from the ANM Hessian.
    
    Args:
        coords: (N, 3) numpy array
        cutoff: distance cutoff for ENM
        n_modes: number of non-trivial modes to return
    
    Returns:
        eigenvalues: (n_modes,) array of eigenvalues (frequencies^2)
        eigenvectors: (n_modes, N, 3) array of mode shapes
    """
    N = len(coords)
    hessian = build_hessian_enm(coords, cutoff=cutoff)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    
    # First 6 eigenvalues are ~0 (rigid body modes: 3 translation + 3 rotation)
    # We want the next n_modes
    start_idx = 6
    end_idx = start_idx + n_modes
    
    if end_idx > len(eigenvalues):
        end_idx = len(eigenvalues)
        n_modes = end_idx - start_idx
    
    selected_evals = eigenvalues[start_idx:end_idx]
    selected_evecs = eigenvectors[:, start_idx:end_idx]  # (3N, n_modes)
    
    # Reshape eigenvectors to (n_modes, N, 3)
    modes = []
    for k in range(n_modes):
        mode = selected_evecs[:, k].reshape(N, 3)
        # Normalize
        mode = mode / np.linalg.norm(mode)
        modes.append(mode)
    
    return selected_evals, np.array(modes)


def generate_ensemble(coords, n_conformers=500, n_modes=10, amplitude_scale=0.3, 
                      cutoff=7.0, seed=42):
    """
    Generate an ensemble of conformations by sampling along normal modes.
    
    Each conformer is: x_new = x_ref + sum_k(c_k * mode_k)
    where c_k ~ N(0, amplitude / sqrt(eigenvalue_k))
    
    This produces physically plausible deformations concentrated in 
    low-frequency collective motions (the ones relevant to biology).
    
    Args:
        coords: (N, 3) numpy array, reference structure
        n_conformers: number of conformers to generate
        n_modes: number of normal modes to use
        amplitude_scale: overall amplitude of deformation
        cutoff: ENM cutoff distance
        seed: random seed
    
    Returns:
        ensemble: (n_conformers, N, 3) numpy array
    """
    rng = np.random.RandomState(seed)
    
    eigenvalues, modes = compute_normal_modes(coords, cutoff=cutoff, n_modes=n_modes)
    
    # Avoid division by zero for near-zero eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    
    ensemble = []
    for _ in range(n_conformers):
        # Sample coefficients: lower frequency modes get larger amplitudes
        # c_k ~ N(0, amplitude / sqrt(lambda_k))
        coefficients = rng.randn(len(eigenvalues)) * amplitude_scale / np.sqrt(eigenvalues)
        
        # Linear combination of modes
        displacement = np.zeros_like(coords)
        for k, c in enumerate(coefficients):
            displacement += c * modes[k]
        
        conformer = coords + displacement
        ensemble.append(conformer)
    
    return np.array(ensemble, dtype=np.float32)


def prepare_ensemble_dataset(pdb_data_path, output_path, n_conformers=500, 
                              n_modes=10, amplitude_scale=0.3):
    """
    Load a processed structure, generate an ensemble, and save as a dataset.
    
    Args:
        pdb_data_path: path to the processed .pt file (single structure)
        output_path: path to save the ensemble .pt file
        n_conformers: number of conformers
        n_modes: number of NMA modes
        amplitude_scale: deformation amplitude
    """
    try:
        data = torch.load(pdb_data_path, weights_only=False)
    except TypeError:
        data = torch.load(pdb_data_path)
    
    coords = data['coords'].numpy()  # Already centered and scaled
    scale = data['scale']
    centroid = data['centroid']
    
    print(f"Reference structure: {len(coords)} atoms")
    print(f"Generating {n_conformers} conformers using {n_modes} normal modes...")
    
    # We need to work in original (unscaled) space for ENM cutoffs to make sense
    # Scale back to Angstroms
    coords_angstrom = coords * scale + centroid.numpy()
    
    # Generate ensemble in Angstrom space
    ensemble_angstrom = generate_ensemble(
        coords_angstrom, 
        n_conformers=n_conformers, 
        n_modes=n_modes, 
        amplitude_scale=amplitude_scale
    )
    
    # Re-center and re-scale each conformer individually
    ensemble_scaled = []
    for conf in ensemble_angstrom:
        c = np.mean(conf, axis=0)
        centered = conf - c
        max_norm = np.max(np.linalg.norm(centered, axis=1))
        if max_norm == 0:
            max_norm = 1.0
        scaled = centered / max_norm
        ensemble_scaled.append(scaled)
    
    ensemble_tensor = torch.from_numpy(np.array(ensemble_scaled, dtype=np.float32))
    
    torch.save({
        'ensemble': ensemble_tensor,  # (n_conformers, N, 3)
        'reference': data['coords'],
        'n_conformers': n_conformers,
        'n_modes': n_modes,
        'amplitude_scale': amplitude_scale,
        'name': data.get('name', 'unknown'),
    }, output_path)
    
    print(f"Saved ensemble dataset ({ensemble_tensor.shape}) to {output_path}")
    return ensemble_tensor


if __name__ == "__main__":
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    input_path = os.path.join(base, "data", "processed", "ubiquitin_ca.pt")
    output_path = os.path.join(base, "data", "processed", "ubiquitin_ensemble.pt")
    
    prepare_ensemble_dataset(input_path, output_path, n_conformers=500, n_modes=10, amplitude_scale=0.3)
