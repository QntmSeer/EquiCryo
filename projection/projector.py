import torch
import torch.nn as nn
import numpy as np
import math

class CryoProjector(nn.Module):
    def __init__(self, output_size=(64, 64), pixel_size=1.0, sigma_noise=0.1, 
                 voltage=300, defocus=None, Cs=2.7, amplitude_contrast=0.1):
        """
        Args:
            output_size: (H, W) of the projection image
            pixel_size: angstroms per pixel
            sigma_noise: standard deviation of Gaussian noise
            voltage: accelerating voltage in kV (default 300)
            defocus: defocus in micrometers (if None, no CTF is applied)
            Cs: spherical aberration in mm
            amplitude_contrast: fraction of amplitude contrast
        """
        super().__init__()
        self.output_size = output_size
        self.pixel_size = pixel_size
        self.sigma_noise = sigma_noise
        
        # CTF Parameters
        self.defocus = defocus
        self.Cs = Cs
        self.amplitude_contrast = amplitude_contrast
        
        # Calculate Electron Wavelength (Relativistic)
        h = 6.626e-34
        m0 = 9.109e-31
        e = 1.602e-19
        c = 2.998e8
        V = voltage * 1000
        self.wavelength = h / math.sqrt(2 * m0 * e * V * (1 + (e * V) / (2 * m0 * c**2))) * 1e10 # Angstroms

    def compute_ctf(self, device):
        """Precomputes the CTF kernel for the given output size."""
        H, W = self.output_size
        fy = torch.fft.fftfreq(H, d=self.pixel_size, device=device)
        fx = torch.fft.fftfreq(W, d=self.pixel_size, device=device)
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')
        S2 = FX**2 + FY**2 # squared spatial frequency magnitude
        
        # Phase shift: chi(s) = pi * lambda * s^2 * (df - 0.5 * Cs * lambda^2 * s^2)
        # Note: units must match. df is in vacuum-angstroms (usually um * 1e4)
        df_ang = self.defocus * 1e4
        Cs_ang = self.Cs * 1e7
        
        gamma = math.pi * self.wavelength * S2 * (df_ang - 0.5 * Cs_ang * self.wavelength**2 * S2)
        
        # CTF = - w * sin(gamma) + (1-w^2)^0.5 * cos(gamma)
        # Simplified: w * sin(gamma) + Q * cos(gamma) where Q is amplitude contrast
        ctf = - (self.amplitude_contrast * torch.cos(gamma) + torch.sqrt(torch.tensor(1 - self.amplitude_contrast**2)) * torch.sin(gamma))
        return ctf

    def apply_ctf(self, projections):
        """Applies the CTF to a batch of projections in the Fourier domain."""
        if self.defocus is None:
            return projections
        
        device = projections.device
        ctf = self.compute_ctf(device)
        
        # Forward FFT
        proj_fft = torch.fft.fft2(projections)
        # Apply CTF
        proj_ctf_fft = proj_fft * ctf
        # Inverse FFT
        proj_ctf = torch.fft.ifft2(proj_ctf_fft).real
        return proj_ctf

    def random_rotation_matrix(self, batch_size, device='cpu'):
        """
        Generates random 3D rotation matrices sampling uniformly from SO(3).
        """
        A = torch.randn(batch_size, 3, 3, device=device)
        Q, R = torch.linalg.qr(A)
        # Fix sign to ensure determinant is 1 (rotation, not reflection)
        sign = torch.diagonal(R, dim1=1, dim2=2).sign()
        Q = Q * sign.unsqueeze(1)
        return Q

    def project(self, x, rot_matrices):
        """
        x: (B, N, 3) Point Cloud
        rot_matrices: (B, 3, 3)
        Returns: 
            projections: (B, H, W)
        """
        B, N, D = x.shape
        H, W = self.output_size
        device = x.device
        
        x_rot = torch.bmm(rot_matrices, x.transpose(1, 2)) # (B, 3, N)
        
        # Normalize to image coordinates
        scale_factor = (min(H, W) / 2.0) * 0.8
        proj_coords = x_rot[:, :2, :] * scale_factor
        
        proj_coords[:, 0, :] += W / 2.0
        proj_coords[:, 1, :] += H / 2.0
        
        # Grid
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1) # (1, H, W, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1)
        
        px = proj_coords[:, 0, :].unsqueeze(1).unsqueeze(1) # (B, 1, 1, N)
        py = proj_coords[:, 1, :].unsqueeze(1).unsqueeze(1)
        
        # Sigma for the "atom" size in the projection
        atom_sigma = 1.5 
        
        dist_sq = (grid_x - px)**2 + (grid_y - py)**2
        gaussians = torch.exp(-dist_sq / (2 * atom_sigma**2))
        
        projection = torch.sum(gaussians, dim=-1) # (B, H, W)
        
        # Research Upgrade 3: Apply CTF
        if self.defocus is not None:
             projection = self.apply_ctf(projection)
             
        return projection

    def forward(self, x, num_projections=1):
        """
        x: (B, N, 3) or (N, 3)
        Returns: (B*num_projections, H, W) noisy projections, and rotations
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        B = x.shape[0]
        device = x.device
        
        # Repeat input if generating multiple projections per structure
        if num_projections > 1:
            x = x.repeat_interleave(num_projections, dim=0)
            B = B * num_projections
            
        rot_matrices = self.random_rotation_matrix(B, device=device)
        clean_proj = self.project(x, rot_matrices)
        
        # Add noise
        noise = torch.randn_like(clean_proj) * self.sigma_noise
        noisy_proj = clean_proj + noise
        
        return noisy_proj, rot_matrices, clean_proj
