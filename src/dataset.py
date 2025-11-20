import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path


class ImagePairDataset(Dataset):
    """
    Dataset for paired raw/edited images with prefix/suffix pattern matching.
    
    Supports training mode (requires both raw and edited) and inference mode (only raw).
    """
    
    def __init__(self, raw_dir, edited_dir=None, image_size=256, mode='train'):
        """
        Args:
            raw_dir: Directory containing raw images
            edited_dir: Directory containing edited images (None for inference mode)
            image_size: Size to resize images to (default: 256)
            mode: 'train' or 'inference'
        """
        self.raw_dir = Path(raw_dir)
        self.edited_dir = Path(edited_dir) if edited_dir else None
        self.image_size = image_size
        self.mode = mode
        
        # Supported formats
        self.raw_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.raw', '.cr2', '.nef', '.arw'}
        self.edited_formats = {'.jpg', '.jpeg', '.png'}
        
        # Get all raw images
        self.raw_images = self._get_images(self.raw_dir, self.raw_formats)
        
        if mode == 'train':
            if not self.edited_dir or not self.edited_dir.exists():
                raise ValueError(f"Edited directory must exist for training mode: {edited_dir}")
            # Get all edited images
            self.edited_images = self._get_images(self.edited_dir, self.edited_formats)
            # Match pairs by prefix/suffix
            self.pairs = self._match_pairs()
            if len(self.pairs) == 0:
                raise ValueError("No matching image pairs found. Check prefix/suffix patterns.")
        else:
            # Inference mode: only raw images needed
            self.pairs = [(img, None) for img in self.raw_images]
        
        # Transform for training
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        # Transform for inference (no normalization needed for saving)
        self.transform_inference = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    def _get_images(self, directory, formats):
        """Get all image files from directory with specified formats."""
        images = []
        if not directory.exists():
            return images
        
        for ext in formats:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        
        return sorted(images)
    
    def _extract_base_name(self, filename):
        """Extract base name without extension for matching."""
        return Path(filename).stem
    
    def _match_pairs(self):
        """
        Match raw and edited images by prefix/suffix pattern.
        Tries multiple matching strategies:
        1. Exact base name match
        2. Prefix match (e.g., raw_photo1.jpg <-> edited_photo1.jpg)
        3. Suffix match (e.g., photo1_raw.jpg <-> photo1_edited.jpg)
        """
        pairs = []
        edited_dict = {self._extract_base_name(ed): ed for ed in self.edited_images}
        
        for raw_img in self.raw_images:
            raw_base = self._extract_base_name(raw_img)
            matched_edited = None
            
            # Strategy 1: Exact match
            if raw_base in edited_dict:
                matched_edited = edited_dict[raw_base]
            else:
                # Strategy 2: Prefix match (raw_xxx <-> edited_xxx)
                if raw_base.startswith('raw_'):
                    edited_base = 'edited_' + raw_base[4:]
                    if edited_base in edited_dict:
                        matched_edited = edited_dict[edited_base]
                elif raw_base.startswith('Raw_'):
                    edited_base = 'Edited_' + raw_base[4:]
                    if edited_base in edited_dict:
                        matched_edited = edited_dict[edited_base]
                
                # Strategy 3: Suffix match (xxx_raw <-> xxx_edited)
                if not matched_edited:
                    if raw_base.endswith('_raw'):
                        edited_base = raw_base[:-4] + '_edited'
                        if edited_base in edited_dict:
                            matched_edited = edited_dict[edited_base]
                    elif raw_base.endswith('_Raw'):
                        edited_base = raw_base[:-4] + '_Edited'
                        if edited_base in edited_dict:
                            matched_edited = edited_dict[edited_base]
            
            if matched_edited:
                pairs.append((raw_img, matched_edited))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        raw_path, edited_path = self.pairs[idx]
        
        # Load raw image
        try:
            raw_img = Image.open(raw_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading raw image {raw_path}: {e}")
        
        if self.mode == 'train':
            # Load edited image for training
            try:
                edited_img = Image.open(edited_path).convert('RGB')
            except Exception as e:
                raise ValueError(f"Error loading edited image {edited_path}: {e}")
            
            # Apply transforms
            raw_tensor = self.transform(raw_img)
            edited_tensor = self.transform(edited_img)
            
            return {
                'raw': raw_tensor,
                'edited': edited_tensor,
                'raw_path': str(raw_path),
                'edited_path': str(edited_path)
            }
        else:
            # Inference mode: only return raw image
            raw_tensor = self.transform_inference(raw_img)
            
            return {
                'raw': raw_tensor,
                'raw_path': str(raw_path)
            }

