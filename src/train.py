import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from model import Generator, Discriminator, get_device
from dataset import ImagePairDataset


class Pix2PixLoss(nn.Module):
    """Pix2Pix loss: L1 loss + adversarial loss."""
    
    def __init__(self, lambda_l1=100.0):
        super(Pix2PixLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
    
    def __call__(self, generated, target, disc_output, is_real):
        # Adversarial loss
        adversarial_loss = self.bce_loss(disc_output, torch.ones_like(disc_output) if is_real else torch.zeros_like(disc_output))
        
        # L1 loss (pixel-wise)
        l1_loss = self.l1_loss(generated, target)
        
        # Total generator loss
        total_loss = adversarial_loss + self.lambda_l1 * l1_loss
        
        return total_loss, adversarial_loss, l1_loss


def train_epoch(generator, discriminator, dataloader, gen_optimizer, disc_optimizer, 
                criterion, device, epoch):
    """Train for one epoch."""
    generator.train()
    discriminator.train()
    
    total_gen_loss = 0.0
    total_disc_loss = 0.0
    total_l1_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        raw_images = batch['raw'].to(device)
        edited_images = batch['edited'].to(device)
        
        # ===================
        # Train Discriminator
        # ===================
        disc_optimizer.zero_grad()
        
        # Real images
        disc_real_output = discriminator(raw_images, edited_images)
        # Create labels matching the discriminator output size
        real_labels = torch.ones_like(disc_real_output)
        disc_real_loss = criterion.bce_loss(disc_real_output, real_labels)
        
        # Fake images
        fake_edited = generator(raw_images)
        disc_fake_output = discriminator(raw_images, fake_edited.detach())
        fake_labels = torch.zeros_like(disc_fake_output)
        disc_fake_loss = criterion.bce_loss(disc_fake_output, fake_labels)
        
        # Total discriminator loss
        disc_loss = (disc_real_loss + disc_fake_loss) * 0.5
        disc_loss.backward()
        disc_optimizer.step()
        
        # ===================
        # Train Generator
        # ===================
        gen_optimizer.zero_grad()
        
        # Generate fake images
        fake_edited = generator(raw_images)
        disc_fake_output = discriminator(raw_images, fake_edited)
        
        # Generator loss
        gen_loss, adv_loss, l1_loss = criterion(fake_edited, edited_images, disc_fake_output, is_real=True)
        gen_loss.backward()
        gen_optimizer.step()
        
        # Update statistics
        total_gen_loss += gen_loss.item()
        total_disc_loss += disc_loss.item()
        total_l1_loss += l1_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Gen Loss': f'{gen_loss.item():.4f}',
            'Disc Loss': f'{disc_loss.item():.4f}',
            'L1 Loss': f'{l1_loss.item():.4f}'
        })
    
    avg_gen_loss = total_gen_loss / len(dataloader)
    avg_disc_loss = total_disc_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    
    return avg_gen_loss, avg_disc_loss, avg_l1_loss


def save_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, 
                   epoch, loss, checkpoint_dir):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train EditGenerationModel (EGM) - Pix2Pix model for image editing')
    parser.add_argument('--raw_dir', type=str, default='train_data/Raw',
                       help='Directory containing raw training images')
    parser.add_argument('--edited_dir', type=str, default='train_data/Edited',
                       help='Directory containing edited training images')
    parser.add_argument('--checkpoint_dir', type=str, default='models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Beta1 for Adam optimizer')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for training')
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                       help='Weight for L1 loss')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = ImagePairDataset(
        raw_dir=args.raw_dir,
        edited_dir=args.edited_dir,
        image_size=args.image_size,
        mode='train'
    )
    print(f"Found {len(dataset)} image pairs")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type != 'mps' else False
    )
    
    # Initialize models
    print("Initializing models...")
    generator = Generator(in_channels=3, out_channels=3).to(device)
    discriminator = Discriminator(in_channels=6).to(device)
    
    # Initialize optimizers
    gen_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999)
    )
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999)
    )
    
    # Loss function
    criterion = Pix2PixLoss(lambda_l1=args.lambda_l1)
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        gen_loss, disc_loss, l1_loss = train_epoch(
            generator, discriminator, dataloader,
            gen_optimizer, disc_optimizer, criterion,
            device, epoch
        )
        
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Gen Loss: {gen_loss:.4f}, "
              f"Disc Loss: {disc_loss:.4f}, "
              f"L1 Loss: {l1_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_checkpoint(
                generator, discriminator,
                gen_optimizer, disc_optimizer,
                epoch, gen_loss, args.checkpoint_dir
            )
    
    print("Training completed!")


if __name__ == '__main__':
    main()

