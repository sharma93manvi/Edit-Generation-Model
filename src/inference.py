import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm

from model import Generator, get_device
from dataset import ImagePairDataset


def denormalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1] range."""
    return (tensor + 1) / 2.0


def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    # Denormalize
    tensor = denormalize(tensor)
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to numpy and transpose
    image = tensor.cpu().detach().squeeze(0)
    if image.dim() == 3:
        image = image.permute(1, 2, 0)
    image = image.numpy()
    # Convert to [0, 255] and uint8
    image = (image * 255).astype('uint8')
    return Image.fromarray(image)


def save_image(image, output_path, format='JPEG'):
    """Save PIL Image to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.upper() == 'JPEG' or format.upper() == 'JPG':
        image.save(output_path, 'JPEG', quality=95)
    elif format.upper() == 'PNG':
        image.save(output_path, 'PNG')
    else:
        image.save(output_path)


def get_output_filename(raw_path, output_dir):
    """Generate output filename based on raw image name."""
    raw_path = Path(raw_path)
    raw_name = raw_path.stem
    raw_ext = raw_path.suffix.lower()
    
    # Determine output format based on original extension
    if raw_ext in ['.jpg', '.jpeg']:
        output_ext = '.jpg'
        format = 'JPEG'
    elif raw_ext == '.png':
        output_ext = '.png'
        format = 'PNG'
    else:
        # Default to JPG for other formats
        output_ext = '.jpg'
        format = 'JPEG'
    
    # Apply prefix/suffix pattern if present
    if raw_name.startswith('raw_'):
        output_name = 'edited_' + raw_name[4:]
    elif raw_name.startswith('Raw_'):
        output_name = 'Edited_' + raw_name[4:]
    elif raw_name.endswith('_raw'):
        output_name = raw_name[:-4] + '_edited'
    elif raw_name.endswith('_Raw'):
        output_name = raw_name[:-4] + '_Edited'
    else:
        # No pattern found, just add _edited suffix
        output_name = raw_name + '_edited'
    
    output_path = Path(output_dir) / f"{output_name}{output_ext}"
    return output_path, format


def main():
    parser = argparse.ArgumentParser(description='Run inference on raw images')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--raw_dir', type=str, default='execution_data/Raw',
                       help='Directory containing raw images to edit')
    parser.add_argument('--output_dir', type=str, default='execution_data/Edited',
                       help='Directory to save edited images')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for processing (should match training)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize generator
    generator = Generator(in_channels=3, out_channels=3).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print("Model loaded successfully")
    
    # Create dataset
    print("Loading images...")
    dataset = ImagePairDataset(
        raw_dir=args.raw_dir,
        edited_dir=None,
        image_size=args.image_size,
        mode='inference'
    )
    print(f"Found {len(dataset)} images to process")
    
    if len(dataset) == 0:
        print("No images found in raw directory. Exiting.")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Set to 0 for inference to avoid issues
    )
    
    # Process images
    print("Generating edited images...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            raw_images = batch['raw'].to(device)
            raw_paths = batch['raw_path']
            
            # Generate edited images
            if isinstance(raw_paths, str):
                raw_paths = [raw_paths]
            
            fake_edited = generator(raw_images)
            
            # Save each image
            for i in range(fake_edited.size(0)):
                output_tensor = fake_edited[i:i+1]
                output_image = tensor_to_image(output_tensor)
                
                raw_path = raw_paths[i] if isinstance(raw_paths, list) else raw_paths
                output_path, format = get_output_filename(raw_path, args.output_dir)
                
                save_image(output_image, output_path, format)
                print(f"Saved: {output_path}")
    
    print(f"\nInference completed! Edited images saved to {args.output_dir}")


if __name__ == '__main__':
    main()

