# Edit Generation Model (EGM)

A deep learning-based image editing tool that learns user editing preferences from raw/edited image pairs and automatically applies those preferences to new images.

## Overview

EditGenerationModel (EGM) uses a Pix2Pix (conditional GAN) architecture to learn how you edit images. By training on pairs of raw and edited images, the model learns your editing style and can automatically apply it to new raw images.

## Features

- **Automatic Style Learning**: Trains on your raw/edited image pairs to learn your editing preferences
- **Pattern-Based Matching**: Automatically matches raw and edited images using prefix/suffix patterns
- **GPU Acceleration**: Supports MPS (Apple Silicon), CUDA (NVIDIA), and CPU
- **Flexible Input Formats**: Supports various image formats for raw images (JPG, PNG, TIFF, RAW, etc.)
- **Standardized Output**: Generates edited images in JPG or PNG format

## Project Structure

```
EditGenerationModel/
├── train_data/
│   ├── Raw/          # Raw training images
│   └── Edited/       # User-edited training images
├── execution_data/
│   ├── Raw/          # New raw images to edit
│   └── Edited/       # Auto-generated edited images
├── models/
│   └── (saved model checkpoints)
├── src/
│   ├── model.py      # Pix2Pix model definitions
│   ├── dataset.py    # Dataset class with pattern matching
│   ├── train.py      # Training script
│   └── inference.py  # Inference script
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sharma93manvi/EditGenerationModel.git
cd EditGenerationModel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Training Data

Place your training images in the following structure:

- **Raw images**: `train_data/Raw/`
- **Edited images**: `train_data/Edited/`

#### Image Pair Matching

The model automatically matches raw and edited images using prefix/suffix patterns:

**Prefix Pattern:**
- `raw_photo1.jpg` ↔ `edited_photo1.jpg`
- `Raw_image1.png` ↔ `Edited_image1.png`

**Suffix Pattern:**
- `photo1_raw.jpg` ↔ `photo1_edited.jpg`
- `image1_Raw.png` ↔ `image1_Edited.png`

**Exact Match (fallback):**
- `photo1.jpg` ↔ `photo1.jpg` (same base name)

**Supported Formats:**
- Raw images: JPG, PNG, TIFF, BMP, RAW, CR2, NEF, ARW, etc.
- Edited images: JPG, PNG only

### 2. Train the Model

Train the model on your image pairs:

```bash
python src/train.py --raw_dir train_data/Raw --edited_dir train_data/Edited
```

**Training Options:**
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.0002)
- `--image_size`: Image size for training (default: 256)
- `--lambda_l1`: Weight for L1 loss (default: 100.0)
- `--save_interval`: Save checkpoint every N epochs (default: 10)

**Example:**
```bash
python src/train.py \
    --raw_dir train_data/Raw \
    --edited_dir train_data/Edited \
    --epochs 300 \
    --batch_size 8 \
    --lr 0.0001 \
    --image_size 512
```

Checkpoints are saved to `models/` directory.

### 3. Run Inference

After training, generate edited images from new raw images:

1. Place raw images in `execution_data/Raw/`

2. Run inference:
```bash
python src/inference.py --checkpoint models/checkpoint_epoch_200.pth
```

**Inference Options:**
- `--checkpoint`: Path to model checkpoint (required)
- `--raw_dir`: Directory with raw images (default: `execution_data/Raw`)
- `--output_dir`: Directory to save edited images (default: `execution_data/Edited`)
- `--image_size`: Image size (should match training, default: 256)
- `--batch_size`: Batch size for inference (default: 1)

**Example:**
```bash
python src/inference.py \
    --checkpoint models/checkpoint_epoch_200.pth \
    --raw_dir execution_data/Raw \
    --output_dir execution_data/Edited \
    --image_size 256
```

Edited images will be saved to `execution_data/Edited/` with appropriate naming based on the pattern matching rules.

## Model Architecture

The model uses a **Pix2Pix** architecture:

- **Generator**: U-Net with encoder-decoder structure and skip connections
- **Discriminator**: PatchGAN that classifies 70x70 image patches
- **Loss Function**: Combination of L1 loss (pixel-wise) and adversarial loss

## Tips for Best Results

1. **Training Data Quality**: 
   - Use at least 50-100 image pairs for good results
   - Ensure consistent editing style across training images
   - Use high-quality images

2. **Image Pairing**:
   - Use consistent naming patterns (prefix or suffix)
   - Ensure raw and edited images are properly matched

3. **Training Parameters**:
   - Start with default parameters
   - Increase `--epochs` for better results (200-500+)
   - Adjust `--lambda_l1` to balance between pixel accuracy and style (higher = more pixel-accurate)

4. **Image Size**:
   - Larger images (512x512) may give better quality but require more memory
   - Start with 256x256 for faster training

## Hardware Requirements

- **Recommended**: GPU (MPS for Apple Silicon, CUDA for NVIDIA)
- **Minimum**: CPU (will be slower)
- **Memory**: At least 8GB RAM, 16GB+ recommended

## Troubleshooting

**No matching image pairs found:**
- Check that raw and edited images follow the naming patterns
- Verify images are in the correct directories
- Ensure edited images are in JPG or PNG format

**Out of memory errors:**
- Reduce `--batch_size`
- Reduce `--image_size`
- Close other applications

**Poor editing results:**
- Train for more epochs
- Use more training data
- Ensure consistent editing style in training data

## License

This project is open source and available for personal and commercial use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

