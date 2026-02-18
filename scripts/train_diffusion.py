"""
Training script for the 3D Point Cloud Diffusion Model.

Supports two modes:
1. Single structure (original, for sanity check / overfitting)
2. Conformational ensemble (research-grade, Upgrade 1)

Usage:
    python scripts/train_diffusion.py                     # ensemble mode (default)
    python scripts/train_diffusion.py --mode single       # single structure mode
"""

import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import PointDiffusionTransformer, DiffusionModel
from data.dataset import ProteinStructureDataset, ConformationalEnsembleDataset, MultiProteinDataset, collate_fn


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    print(f"Mode: {args.mode}")
    
    # Data
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    if args.mode == 'ensemble':
        # Automatically find all ensemble files
        processed_dir = os.path.join(base, "data", "processed")
        ensemble_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith('_ensemble.pt')]
        print(f"Found ensembles: {[os.path.basename(f) for f in ensemble_files]}")
        
        if len(ensemble_files) > 1:
            dataset = MultiProteinDataset(ensemble_files, augment_noise=0.005)
            checkpoint_name = "ddpm_multi_protein.pth"
            collate = collate_fn
        else:
            dataset = ConformationalEnsembleDataset(ensemble_files[0], augment_noise=0.005)
            checkpoint_name = "ddpm_ensemble.pth"
            collate = None
    else:
        data_path = os.path.join(base, "data", "processed", "ubiquitin_ca.pt")
        dataset = ProteinStructureDataset(data_path, augment_noise=0.01)
        checkpoint_name = "ddpm_ubiquitin.pth"
        collate = None
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    
    # Model — use the more capable PointDiffusionTransformer
    net = PointDiffusionTransformer(
        in_dim=3, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers
    )
    model = DiffusionModel(net, timesteps=args.timesteps).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    loss_history = []
    best_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x_0, mask = [b.to(device) for b in batch]
            else:
                x_0 = batch.to(device)
                mask = None
            
            optimizer.zero_grad()
            loss = model.get_loss(x_0, mask=mask)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("experiments/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"experiments/checkpoints/{checkpoint_name}")
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}, "
                  f"Best: {best_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print(f"Training complete. Best loss: {best_loss:.6f}")
    
    # Save Loss Curve
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, 'b-', alpha=0.7)
    plt.title(f"Training Loss ({args.mode} mode)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"experiments/training_loss_{args.mode}.png", dpi=150)
    print("Loss curve saved.")

    # Validation: Generate samples
    print("Sampling from trained model...")
    model.eval()
    
    example_item = dataset[0]
    if isinstance(example_item, tuple):
        n_atoms = example_item[0].shape[0]
    else:
        n_atoms = example_item.shape[0]
    n_samples = 4
    
    with torch.no_grad():
        samples = model.sample((n_samples, n_atoms, 3), device=device)
    
    # Visualize
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4), 
                              subplot_kw={'projection': '3d'})
    
    for i in range(n_samples):
        s = samples[i].cpu().numpy()
        axes[i].scatter(s[:,0], s[:,1], s[:,2], c='steelblue', s=15, alpha=0.7)
        axes[i].plot(s[:,0], s[:,1], s[:,2], c='coral', alpha=0.4)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_zticks([])
    
    plt.suptitle(f"Generated Conformers ({args.mode} mode)")
    plt.tight_layout()
    plt.savefig(f"experiments/generated_samples_{args.mode}.png", dpi=150)
    print("Sample visualization saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='ensemble', 
                        choices=['single', 'ensemble'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--timesteps', type=int, default=1000)
    args = parser.parse_args()
    
    train(args)
