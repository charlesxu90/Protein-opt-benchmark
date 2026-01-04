#!/usr/bin/env python
"""
train_GB1_VED.py - Train Variant Encoder-Decoder (VED) for GB1 dataset

This script trains the ESM-2 based VED model on GB1 sequences for use with LatProtRL.

Usage:
    python train_GB1_VED.py --device cuda:0 --batch 32 --reduce_dim 16
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net.seq_lm import VED
from utils.datasets import SequenceDataset
from utils.constants import generate_random_mutant

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# GB1 constants
GB1_WILDTYPE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
GB1_LENGTH = 56

GB1_REFSEQ = {
    'hard': GB1_WILDTYPE,
    'medium': GB1_WILDTYPE,
}


def get_gb1_ved_config(device: str, level: str, reduce_dim: int = 16, num_trainable_layers: int = 4):
    """Get VED configuration for GB1."""
    args = argparse.Namespace()
    args.name = 'GB1'
    args.device = device
    args.level = level
    args.embed_dim = 1280
    args.num_layers = 33
    args.hidden_dim = 256
    args.length = GB1_LENGTH
    args.num_tokens = GB1_LENGTH + 2
    args.reduce_dim = reduce_dim
    args.num_trainable_layers = num_trainable_layers
    return args


def train_gb1_ved(
    device: str = "cuda:0",
    level: str = "medium",
    batch_size: int = 32,
    reduce_dim: int = 16,
    num_trainable_layers: int = 4,
    augment: int = 4,
    total_epochs: int = 32,
    use_wandb: bool = True,
    data_dir: str = "data/GB1",
    esm_checkpoint: str = "ckpt/esm2_t33_650M_UR50D.pt",
):
    """
    Train VED model for GB1.

    Args:
        device: Device to use
        level: Difficulty level ('hard' or 'medium')
        batch_size: Training batch size
        reduce_dim: Latent dimension
        num_trainable_layers: Number of trainable ESM layers
        augment: Number of augmentation rounds
        total_epochs: Total training epochs
        use_wandb: Whether to use wandb logging
        data_dir: Data directory
        esm_checkpoint: Path to ESM-2 checkpoint
    """
    save_name = f'GB1_{level}_{time.strftime("%d-%H:%M", time.localtime())}'

    if not os.path.exists('saved/'):
        os.makedirs('saved/')

    # Create configuration
    config = get_gb1_ved_config(device, level, reduce_dim, num_trainable_layers)

    # Check ESM checkpoint
    if not os.path.exists(esm_checkpoint):
        raise FileNotFoundError(
            f"ESM-2 checkpoint not found at {esm_checkpoint}. "
            "Please download from: https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"
        )

    # Create model
    print("Initializing VED model...")
    model = VED(config, esm_pretrained=esm_checkpoint)
    model = model.to(device)

    # Get reference sequence
    ref_seq = GB1_REFSEQ[level]
    wt_tokens = model.compose_input([('protein', ref_seq)]).cpu()
    model.set_wt_tokens(ref_seq)

    # Load data
    data_path = os.path.join(data_dir, f'{level}.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Run prepare_GB1_data.py first."
        )

    data = pd.read_csv(data_path)
    val_size = test_size = int(len(data) * 0.05)

    # Shuffle and split
    data = data.sample(frac=1, random_state=42)
    data.iloc[:test_size].to_csv(os.path.join(data_dir, f'{level}_test.csv'), index=False)
    data = data.iloc[test_size:]
    sequences = data["sequence"].tolist()

    print(f"Training sequences: {len(sequences)}")

    # Data augmentation
    def get_augmented(original):
        return [generate_random_mutant(seq, 3 / config.length) for seq in original]

    original = sequences.copy()
    for k in range(augment):
        sequences += get_augmented(original)

    print(f"Total sequences (with augmentation): {len(sequences)}")

    # Create dataset
    dataset = SequenceDataset(sequences, ref_seq, model.alphabet)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # Loss and optimizer
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-3,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=len(train_loader) * 8,
        gamma=0.5
    )

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project="LatProtRL-GB1-VED", name=save_name)
        wandb.watch(models=model)

    def epoch_loop(state, epoch, loader):
        desc = {key: [] for key in ['CELoss', 'Acc', 'Mutated_Acc', 'True_Dist_WT', 'Pred_Dist_WT']}

        for idx, batch_data in enumerate(tqdm(loader, desc=f'{state} {epoch}/{total_epochs}')):
            seq, mask = batch_data
            seq, mask = seq.to(device), mask.to(device)

            if state == 'train':
                optimizer.zero_grad()
                pred_seq, repr = model(seq, return_rep=True)
            else:
                with torch.no_grad():
                    pred_seq, repr = model(seq, return_rep=True)

            current_batch_size = mask.size(0)
            seq_flat = seq.flatten()
            mask_flat = mask.flatten()
            pred = pred_seq.view(-1, pred_seq.size(-1))
            loss_pos = ce_loss(pred, seq_flat)

            # Upweight mutated positions
            loss_val = (
                0.1 * torch.sum(loss_pos * mask_flat) / max(torch.sum(mask_flat), 1) +
                0.9 * torch.sum(loss_pos * ~mask_flat) / max(torch.sum(~mask_flat), 1)
            ) / 2

            if state == 'train':
                loss_val.backward()
                optimizer.step()
                scheduler.step()

            seq_np = seq_flat.detach().cpu().numpy()
            pred_np = np.argmax(pred.detach().cpu().numpy(), axis=1)
            mask_np = mask_flat.detach().cpu().numpy()

            desc["CELoss"].append(loss_val.item())
            desc["Acc"].append(accuracy_score(seq_np, pred_np))

            mutated_mask = mask_np == 1
            if np.sum(mutated_mask) > 0:
                desc["Mutated_Acc"].append(accuracy_score(seq_np[mutated_mask], pred_np[mutated_mask]))
            else:
                desc["Mutated_Acc"].append(1.0)

            desc["True_Dist_WT"].append(mask_np.sum() / current_batch_size)
            desc["Pred_Dist_WT"].append(
                config.num_tokens - accuracy_score(
                    wt_tokens.repeat(current_batch_size, 1).flatten().numpy(),
                    pred_np,
                    normalize=False
                ) / current_batch_size
            )

            repr_np = repr.detach().cpu().numpy()

            if idx == 1:
                tqdm.write('\t'.join([
                    f'{m:.2f} ({s:.2f})'
                    for m, s in zip(np.mean(repr_np, axis=0), np.std(repr_np, axis=0))
                ]))

            if state == 'train' and use_wandb and WANDB_AVAILABLE:
                desc_iter = {f"{state}/{k}": desc[k][-1] for k in desc}
                wandb.log(desc_iter)

        desc_epoch = {f"{state}/Total_{k}": np.mean(desc[k]) for k in desc}
        if use_wandb and WANDB_AVAILABLE:
            wandb.log(desc_epoch)

        return desc_epoch

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(total_epochs):
        model.train()
        epoch_loop("train", epoch, train_loader)

        model.eval()
        val_metrics = epoch_loop("val", epoch, val_loader)

        # Save checkpoint
        val_loss = val_metrics.get('val/Total_CELoss', float('inf'))
        torch.save(model.state_dict(), f'saved/{save_name}.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Also save as the standard name for easy loading
            torch.save(model.state_dict(), f'saved/GB1_{level}_LM.pt')
            print(f"  Saved best model (val_loss: {val_loss:.6f})")

    print(f"\nTraining complete!")
    print(f"Model saved to: saved/{save_name}.pt")
    print(f"Best model saved to: saved/GB1_{level}_LM.pt")

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train VED model for GB1")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--level', type=str, default='medium', choices=['hard', 'medium'],
                        help='Difficulty level')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--reduce_dim', '-r', type=int, default=16, help='Latent dimension')
    parser.add_argument('--num_trainable_layers', '-l', type=int, default=4,
                        help='Number of trainable ESM layers')
    parser.add_argument('--augment', type=int, default=4, help='Number of augmentation rounds')
    parser.add_argument('--epochs', type=int, default=32, help='Total epochs')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--data_dir', type=str, default='data/GB1', help='Data directory')
    parser.add_argument('--esm_checkpoint', type=str, default='ckpt/esm2_t33_650M_UR50D.pt',
                        help='Path to ESM-2 checkpoint')

    args = parser.parse_args()

    train_gb1_ved(
        device=args.device,
        level=args.level,
        batch_size=args.batch,
        reduce_dim=args.reduce_dim,
        num_trainable_layers=args.num_trainable_layers,
        augment=args.augment,
        total_epochs=args.epochs,
        use_wandb=not args.no_wandb,
        data_dir=args.data_dir,
        esm_checkpoint=args.esm_checkpoint,
    )


if __name__ == "__main__":
    main()
