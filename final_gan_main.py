import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import glob
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import lpips  # Learned Perceptual Image Patch Similarity for better perceptual quality assessment


def load_config(config_path="config.json"):
    """
    Load configuration from a JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# Improved GAN-Enhanced Steganography Model with Spectral Normalization
class ImprovedGANSteganographyModel(nn.Module):
    """
    An improved GAN-enhanced steganography model with additional convolutional layers
    that embeds secret information in the cover image and can extract it from the 
    3-channel stego image ALONE (no cover needed). Includes spectral normalization 
    for GAN training stability.
    """
    def __init__(self):
        super(ImprovedGANSteganographyModel, self).__init__()

        # Embedding network - combines cover and secret to produce stego
        self.embed_net = nn.Sequential(
            # Initial processing
            nn.Conv2d(6, 64, kernel_size=3, padding=1),  # 6 = 3 (cover) + 3 (secret)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Feature extraction blocks
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Additional feature processing
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Additional processing layers
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Output 3-channel stego
            nn.Tanh()  # Keep in [-1, 1] range
        )

        # Extraction network - extracts secret from stego image ONLY (no cover needed)
        self.extract_net = nn.Sequential(
            # Initial processing
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 3 = stego image only
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Feature extraction blocks
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Additional feature processing
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Additional processing layers
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Output 3-channel secret
            nn.Tanh()  # Keep in [-1, 1] range
        )

    def embed(self, cover, secret):
        """
        Embed secret into cover to produce stego image by generating it directly.
        cover: [B, 3, H, W] - normalized to [-1, 1]
        secret: [B, 3, H, W] - normalized to [-1, 1]
        returns: stego image [B, 3, H, W]
        """
        # Concatenate cover and secret
        combined = torch.cat([cover, secret], dim=1)

        # The embedding network now generates the entire stego image directly.
        # It learns to blend the secret into the cover pixels holistically.
        stego = self.embed_net(combined)

        return stego

    def extract(self, stego):
        """
        Extract secret from stego image ONLY (no cover needed)
        stego: [B, 3, H, W] - normalized to [-1, 1]
        returns: extracted secret [B, 3, H, W]
        """
        # Extract secret information from stego image alone
        extracted = self.extract_net(stego)

        return extracted


# Improved Discriminator with Spectral Normalization
class ImprovedDiscriminator(nn.Module):
    """
    An improved discriminator network with more layers and spectral normalization
    to distinguish between cover and stego images, improving the imperceptibility 
    of generated stego images.
    """
    def __init__(self):
        super(ImprovedDiscriminator, self).__init__()
        
        from torch.nn.utils import spectral_norm

        # Input: 3 channels (cover or stego image)
        # Output: 1 channel (probability of being a cover image)
        self.features = nn.Sequential(
            # Block 1
            spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),  # 224x224 -> 112x112
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),  # 112x112 -> 56x56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),  # 56x56 -> 28x28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),  # 28x28 -> 14x14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 5
            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),  # 14x14 -> 7x7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 6 - Additional layer for more complexity
            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),  # 7x7 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to 1x1
            nn.Flatten(),
            spectral_norm(nn.Linear(512, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            spectral_norm(nn.Linear(1024, 1)),
            nn.Sigmoid()  # Output probability [0, 1] for real/fake
        )

    def forward(self, x):
        """
        Forward pass through the discriminator
        x: [B, 3, H, W] - normalized to [-1, 1]
        returns: [B, 1] - probability of being a cover image
        """
        features = self.features(x)
        output = self.classifier(features)
        return output


class ImageSteganographyDataset(Dataset):
    """
    Dataset for 224x224 image steganography - loads pairs of images to use as cover and secret.
    """
    def __init__(self, image_dir, img_size=224, transform=None):
        # Look for images in train/validation/test subfolders
        subfolders = ['train', 'validation', 'test']
        self.image_paths = []

        for subfolder in subfolders:
            subfolder_path = os.path.join(image_dir, subfolder)
            if os.path.exists(subfolder_path):
                png_files = glob.glob(os.path.join(subfolder_path, "*.png"))
                jpg_files = glob.glob(os.path.join(subfolder_path, "*.jpg"))
                jpeg_files = glob.glob(os.path.join(subfolder_path, "*.jpeg"))
                self.image_paths.extend(png_files + jpg_files + jpeg_files)

        # If no images found in subfolders, look in the main directory
        if not self.image_paths:
            png_files = glob.glob(os.path.join(image_dir, "*.png"))
            jpg_files = glob.glob(os.path.join(image_dir, "*.jpg"))
            jpeg_files = glob.glob(os.path.join(image_dir, "*.jpeg"))
            self.image_paths = png_files + jpg_files + jpeg_files

        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir} or its subdirectories")

        self.img_size = img_size

        # Default transformation to normalize images for 224x224
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the cover image
        cover_path = self.image_paths[idx]
        cover_img = Image.open(cover_path).convert('RGB')
        cover_tensor = self.transform(cover_img)

        # Select a different image as the secret
        secret_idx = random.choice([i for i in range(len(self.image_paths)) if i != idx])
        secret_path = self.image_paths[secret_idx]
        secret_img = Image.open(secret_path).convert('RGB')
        secret_tensor = self.transform(secret_img)

        return cover_tensor, secret_tensor


def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images in the range [-1, 1].
    """
    # Clamp images to valid range to prevent numerical issues
    img1 = torch.clamp(img1, -1, 1)
    img2 = torch.clamp(img2, -1, 1)

    mse = F.mse_loss(img1, img2)

    # Prevent division by zero
    if mse.item() == 0:
        return torch.tensor(float('inf'))

    max_pixel = 2.0  # Range is [-1, 1] so max difference is 2
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse + 1e-10))  # Add small epsilon to prevent log(0)
    return psnr


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM between two images in the range [-1, 1].
    """
    # Clamp images to valid range to prevent numerical issues
    img1 = torch.clamp(img1, -1, 1)
    img2 = torch.clamp(img2, -1, 1)

    # Convert from [-1, 1] to [0, 1] range for SSIM calculation
    img1 = (img1 + 1) / 2.0
    img2 = (img2 + 1) / 2.0

    # Ensure images are in the right range [0, 1]
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    def ssim(img1, img2, window_size=11, channel=3, size_average=True):
        # Define Gaussian window
        def create_window(window_size, channel):
            def _gaussian(window_size, sigma):
                gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
                return gauss/gauss.sum()

            _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        def _ssim(img1, img2, window, window_size, channel, size_average=True):
            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1*mu2

            sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

            C1 = 0.01**2
            C2 = 0.03**2

            ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

            if size_average:
                return ssim_map.mean()
            else:
                return ssim_map.mean(1).mean(1).mean(1)

        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return _ssim(img1, img2, window, window_size, channel, size_average)

    return ssim(img1, img2, window_size, img1.size()[1], size_average)


def calculate_mae(img1, img2):
    """
    Calculate MAE (Mean Absolute Error) between two images.
    """
    # Clamp images to valid range to prevent numerical issues
    img1 = torch.clamp(img1, -1, 1)
    img2 = torch.clamp(img2, -1, 1)
    mae = torch.mean(torch.abs(img1 - img2))
    return mae


def calculate_rmse(img1, img2):
    """
    Calculate RMSE (Root Mean Square Error) between two images.
    """
    # Clamp images to valid range to prevent numerical issues
    img1 = torch.clamp(img1, -1, 1)
    img2 = torch.clamp(img2, -1, 1)
    mse = F.mse_loss(img1, img2)
    rmse = torch.sqrt(mse)
    return rmse


def calculate_lpips_loss_fn():
    """
    Initialize LPIPS loss function.
    """
    return lpips.LPIPS(net='alex')  # Using AlexNet as the perceptual model


def improved_gan_steganography_loss(cover, stego, secret, recovered_secret, discriminator, lpips_loss_fn,
                                  alpha_hid=1.0, alpha_rec=1.0, alpha_adv=0.02, alpha_lpips=0.5):
    """
    Improved GAN-enhanced loss function with LPIPS for improved perceptual quality.
    - Hiding loss: L2 (MSE) difference between cover and stego.
    - Recovery loss: A combination of L2 (MSE) and LPIPS loss.
    - Adversarial loss: Discriminator loss to improve imperceptibility.
    """
    # Clamp images to valid range to prevent numerical issues
    cover = torch.clamp(cover, -1, 1)
    stego = torch.clamp(stego, -1, 1)
    secret = torch.clamp(secret, -1, 1)
    recovered_secret = torch.clamp(recovered_secret, -1, 1)

    # Hiding loss: L2 loss between cover image and stego image
    hiding_loss = F.mse_loss(stego, cover)

    # Recovery loss: L2 loss between original secret and recovered secret
    recovery_mse_loss = F.mse_loss(recovered_secret, secret)
    
    # Perceptual loss (LPIPS) for recovery
    recovery_lpips_loss = lpips_loss_fn(recovered_secret, secret).mean() # .mean() is important

    # Combine recovery losses
    recovery_loss = alpha_rec * recovery_mse_loss + alpha_lpips * recovery_lpips_loss

    # Adversarial loss: encourage stego images to fool the discriminator
    # Discriminator should output ~1 for real images (covers) and ~0 for fake (stegos)
    disc_on_stego = discriminator(stego)
    adversarial_loss = F.binary_cross_entropy(disc_on_stego, torch.ones_like(disc_on_stego))

    # Combined generator loss
    total_loss = alpha_hid * hiding_loss + recovery_loss + alpha_adv * adversarial_loss

    return total_loss, hiding_loss, recovery_loss, adversarial_loss


def train_improved_gan_steganography_model(generator, discriminator, dataset, num_epochs=100, 
                                         val_dataset=None, log_dir="runs/improved_gan_steganography",
                                         checkpoint_dir="checkpoints", config=None):
    """
    Train the improved GAN-enhanced steganography model with enhanced logging and tensorboard visualization.
    """
    print(f"\nTraining Improved GAN-Enhanced Steganography model with enhanced logging...")

    # Use config values or defaults
    if config:
        batch_size = config['training']['batch_size']
        learning_rate = config['training']['learning_rate']
        alpha_hid = config['training']['alpha_hid']
        alpha_rec = config['training']['alpha_rec']
        alpha_adv = config['training']['alpha_adv']  # New parameter for adversarial loss
        alpha_lpips = config['training'].get('alpha_lpips', 0.5) # Get LPIPS weight from config
        num_epochs = config['training']['num_epochs']
    else:
        batch_size = 2
        learning_rate = 1e-4
        alpha_hid = 1.0
        alpha_rec = 1.0
        alpha_adv = 0.02  # Weight for adversarial loss
        alpha_lpips = 0.5 # Default LPIPS weight

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        generator = generator.to(device)
        discriminator = discriminator.to(device)

    # Initialize LPIPS loss function for training
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

    # Create dataloader with more workers and pin_memory for better performance
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Create validation dataloader if validation dataset is provided
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizers for generator (embedding + extraction) and discriminator
    gen_optimizer = optim.Adam(
        list(generator.embed_net.parameters()) + list(generator.extract_net.parameters()), 
        lr=learning_rate, weight_decay=1e-5, eps=1e-8
    )
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5, eps=1e-8)

    # Learning rate schedulers
    gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=max(1, num_epochs//4), gamma=0.9)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size=max(1, num_epochs//4), gamma=0.9)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"Training on {len(dataset)} images for {num_epochs} epochs")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Loss weights - Hiding: {alpha_hid}, Recovery: {alpha_rec}, Adversarial: {alpha_adv}, LPIPS: {alpha_lpips}")

    # Initialize lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_hiding_psnrs = []
    val_hiding_psnrs = []
    train_recovery_psnrs = []
    val_recovery_psnrs = []
    train_hiding_ssims = []
    val_hiding_ssims = []
    train_recovery_ssims = []
    val_recovery_ssims = []

    # Training loop
    for epoch in range(num_epochs):
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        epoch_hiding_psnr = 0.0
        epoch_recovery_psnr = 0.0
        epoch_hiding_ssim = 0.0
        epoch_recovery_ssim = 0.0
        batch_count = 0

        generator.train()
        discriminator.train()

        # Use tqdm for progress bar to track epoch progress
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for cover_tensor, secret_tensor in progress_bar:
            # Move tensors to device
            cover_tensor = cover_tensor.to(device)
            secret_tensor = secret_tensor.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            gen_optimizer.zero_grad()

            try:
                # Forward pass through generator
                stego_output = generator.embed(cover_tensor, secret_tensor)
                recovered_secret = generator.extract(stego_output)

                # Calculate metrics
                hiding_psnr_batch = calculate_psnr(stego_output, cover_tensor)
                recovery_psnr_batch = calculate_psnr(recovered_secret, secret_tensor)
                hiding_ssim_batch = calculate_ssim(stego_output, cover_tensor)
                recovery_ssim_batch = calculate_ssim(recovered_secret, secret_tensor)

                # Compute generator loss (includes adversarial component)
                gen_loss, hiding_loss, recovery_loss, adv_loss = improved_gan_steganography_loss(
                    cover_tensor, stego_output, secret_tensor, recovered_secret, discriminator, lpips_loss_fn,
                    alpha_hid, alpha_rec, alpha_adv, alpha_lpips
                )

                # Check for NaN/Inf values and skip the batch if problematic
                if torch.isnan(gen_loss) or torch.isinf(gen_loss):
                    print(f"Skipping batch due to NaN/Inf generator loss: {gen_loss.item()}")
                    torch.cuda.empty_cache()
                    continue

                # Backward pass and update generator
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                gen_optimizer.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                disc_optimizer.zero_grad()

                # Discriminator loss on real images (covers)
                real_labels = torch.ones(cover_tensor.size(0), 1).to(device)
                real_loss = F.binary_cross_entropy(discriminator(cover_tensor), real_labels)

                # Discriminator loss on fake images (stegos)
                fake_labels = torch.zeros(stego_output.size(0), 1).to(device)
                fake_loss = F.binary_cross_entropy(discriminator(stego_output.detach()), fake_labels)

                # Total discriminator loss
                disc_loss = (real_loss + fake_loss) / 2

                # Check for NaN/Inf values in discriminator loss
                if torch.isnan(disc_loss) or torch.isinf(disc_loss):
                    print(f"Skipping batch due to NaN/Inf discriminator loss: {disc_loss.item()}")
                    torch.cuda.empty_cache()
                    continue

                # Backward pass and update discriminator
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                disc_optimizer.step()

                # Accumulate batch-level metrics for epoch averages
                epoch_gen_loss += gen_loss.item()
                epoch_disc_loss += disc_loss.item()
                epoch_hiding_psnr += hiding_psnr_batch.item()
                epoch_recovery_psnr += recovery_psnr_batch.item()
                epoch_hiding_ssim += hiding_ssim_batch.item()
                epoch_recovery_ssim += recovery_ssim_batch.item()
                batch_count += 1

                # Update progress bar description with current metrics
                progress_bar.set_postfix({
                    'Gen Loss': f'{gen_loss.item():.6f}',
                    'Disc Loss': f'{disc_loss.item():.6f}',
                    'Hiding PSNR': f'{hiding_psnr_batch.item():.2f}',
                    'Recovery PSNR': f'{recovery_psnr_batch.item():.2f}'
                })

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Out of memory error in batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"RuntimeError in batch: {e}")
                    continue
            except Exception as e:
                print(f"Unexpected error in batch: {e}")
                continue

        # Calculate average epoch metrics
        if batch_count > 0:
            avg_gen_loss = epoch_gen_loss / batch_count
            avg_disc_loss = epoch_disc_loss / batch_count
            avg_hiding_psnr = epoch_hiding_psnr / batch_count
            avg_recovery_psnr = epoch_recovery_psnr / batch_count
            avg_hiding_ssim = epoch_hiding_ssim / batch_count
            avg_recovery_ssim = epoch_recovery_ssim / batch_count

            print(f"Epoch {epoch+1} - Avg Gen Loss: {avg_gen_loss:.6f}, Avg Disc Loss: {avg_disc_loss:.6f}, "
                  f"Avg Hiding PSNR: {avg_hiding_psnr:.2f} dB, "
                  f"Avg Recovery PSNR: {avg_recovery_psnr:.2f} dB, "
                  f"Avg Hiding SSIM: {avg_hiding_ssim:.4f}, "
                  f"Avg Recovery SSIM: {avg_recovery_ssim:.4f}")

            # Log to TensorBoard
            writer.add_scalar('Loss/Generator_Train', avg_gen_loss, epoch)
            writer.add_scalar('Loss/Discriminator_Train', avg_disc_loss, epoch)
            writer.add_scalar('PSNR/Hiding_Train', avg_hiding_psnr, epoch)
            writer.add_scalar('PSNR/Recovery_Train', avg_recovery_psnr, epoch)
            writer.add_scalar('SSIM/Hiding_Train', avg_hiding_ssim, epoch)
            writer.add_scalar('SSIM/Recovery_Train', avg_recovery_ssim, epoch)

            # Store training metrics for plotting
            train_losses.append(avg_gen_loss + avg_disc_loss)  # Total loss
            train_hiding_psnrs.append(avg_hiding_psnr)
            train_recovery_psnrs.append(avg_recovery_psnr)
            train_hiding_ssims.append(avg_hiding_ssim)
            train_recovery_ssims.append(avg_recovery_ssim)

        # Step the learning rate schedulers
        gen_scheduler.step()
        disc_scheduler.step()

        # --- VALIDATION STEP (every N epochs or every epoch) ---
        if val_dataloader and (epoch + 1) % 1 == 0:  # Validate every epoch, or change 1 to N
            generator.eval()
            discriminator.eval()
            val_epoch_gen_loss = 0.0
            val_epoch_disc_loss = 0.0
            val_epoch_hiding_psnr = 0.0
            val_epoch_recovery_psnr = 0.0
            val_epoch_hiding_ssim = 0.0
            val_epoch_recovery_ssim = 0.0
            val_batch_count = 0

            with torch.no_grad():  # Disable gradient computation for validation
                val_progress_bar = tqdm(val_dataloader, desc=f"Val Epoch {epoch+1}/{num_epochs}")
                for val_cover_tensor, val_secret_tensor in val_progress_bar:
                    val_cover_tensor = val_cover_tensor.to(device)
                    val_secret_tensor = val_secret_tensor.to(device)

                    try:
                        # Forward pass on validation data
                        val_stego_output = generator.embed(val_cover_tensor, val_secret_tensor)
                        val_recovered_secret = generator.extract(val_stego_output)

                        # Calculate validation metrics
                        val_hiding_psnr_batch = calculate_psnr(val_stego_output, val_cover_tensor)
                        val_recovery_psnr_batch = calculate_psnr(val_recovered_secret, val_secret_tensor)
                        val_hiding_ssim_batch = calculate_ssim(val_stego_output, val_cover_tensor)
                        val_recovery_ssim_batch = calculate_ssim(val_recovered_secret, val_secret_tensor)

                        # Compute validation generator loss
                        val_gen_loss, val_hiding_loss, val_recovery_loss, val_adv_loss = improved_gan_steganography_loss(
                            val_cover_tensor, val_stego_output, val_secret_tensor, val_recovered_secret, 
                            discriminator, lpips_loss_fn, alpha_hid, alpha_rec, alpha_adv, alpha_lpips
                        )

                        # Compute validation discriminator loss
                        val_real_loss = F.binary_cross_entropy(
                            discriminator(val_cover_tensor), 
                            torch.ones(val_cover_tensor.size(0), 1).to(device)
                        )
                        val_fake_loss = F.binary_cross_entropy(
                            discriminator(val_stego_output.detach()), 
                            torch.zeros(val_stego_output.size(0), 1).to(device)
                        )
                        val_disc_loss = (val_real_loss + val_fake_loss) / 2

                        # Accumulate validation metrics
                        val_epoch_gen_loss += val_gen_loss.item()
                        val_epoch_disc_loss += val_disc_loss.item()
                        val_epoch_hiding_psnr += val_hiding_psnr_batch.item()
                        val_epoch_recovery_psnr += val_recovery_psnr_batch.item()
                        val_epoch_hiding_ssim += val_hiding_ssim_batch.item()
                        val_epoch_recovery_ssim += val_recovery_ssim_batch.item()
                        val_batch_count += 1

                        # Update validation progress bar
                        val_progress_bar.set_postfix({
                            'Val Gen Loss': f'{val_gen_loss.item():.6f}',
                            'Val Disc Loss': f'{val_disc_loss.item():.6f}',
                            'Val Hiding PSNR': f'{val_hiding_psnr_batch.item():.2f}',
                            'Val Recovery PSNR': f'{val_recovery_psnr_batch.item():.2f}'
                        })

                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue  # Skip problematic validation batch

            # Calculate and log average validation metrics
            if val_batch_count > 0:
                avg_val_gen_loss = val_epoch_gen_loss / val_batch_count
                avg_val_disc_loss = val_epoch_disc_loss / val_batch_count
                avg_val_hiding_psnr = val_epoch_hiding_psnr / val_batch_count
                avg_val_recovery_psnr = val_epoch_recovery_psnr / val_batch_count
                avg_val_hiding_ssim = val_epoch_hiding_ssim / val_batch_count
                avg_val_recovery_ssim = val_epoch_recovery_ssim / val_batch_count

                print(f"  Validation Epoch {epoch+1} - Avg Val Gen Loss: {avg_val_gen_loss:.6f}, "
                      f"Avg Val Disc Loss: {avg_val_disc_loss:.6f}, "
                      f"Avg Val Hiding PSNR: {avg_val_hiding_psnr:.2f} dB, "
                      f"Avg Val Recovery PSNR: {avg_val_recovery_psnr:.2f} dB, "
                      f"Avg Val Hiding SSIM: {avg_val_hiding_ssim:.4f}, "
                      f"Avg Val Recovery SSIM: {avg_val_recovery_ssim:.4f}")

                # Log validation metrics to TensorBoard
                writer.add_scalar('Loss/Generator_Validation', avg_val_gen_loss, epoch)
                writer.add_scalar('Loss/Discriminator_Validation', avg_val_disc_loss, epoch)
                writer.add_scalar('PSNR/Hiding_Validation', avg_val_hiding_psnr, epoch)
                writer.add_scalar('PSNR/Recovery_Validation', avg_val_recovery_psnr, epoch)
                writer.add_scalar('SSIM/Hiding_Validation', avg_val_hiding_ssim, epoch)
                writer.add_scalar('SSIM/Recovery_Validation', avg_val_recovery_ssim, epoch)

                # Store validation metrics for plotting
                val_losses.append(avg_val_gen_loss + avg_val_disc_loss)  # Total loss
                val_hiding_psnrs.append(avg_val_hiding_psnr)
                val_recovery_psnrs.append(avg_val_recovery_psnr)
                val_hiding_ssims.append(avg_val_hiding_ssim)
                val_recovery_ssims.append(avg_val_recovery_ssim)

            generator.train()
            discriminator.train()  # Switch back to training mode

    # Generate and save plots after training
    generate_training_plots(train_losses, val_losses, train_hiding_psnrs, val_hiding_psnrs,
                           train_recovery_psnrs, val_recovery_psnrs, train_hiding_ssims, val_hiding_ssims,
                           train_recovery_ssims, val_recovery_ssims, log_dir)

    print(f"Training completed for Improved GAN-Enhanced Steganography model!")
    
    # Close TensorBoard writer
    writer.close()

    return generator, discriminator


def generate_training_plots(train_losses, val_losses, train_hiding_psnrs, val_hiding_psnrs,
                           train_recovery_psnrs, val_recovery_psnrs, train_hiding_ssims, val_hiding_ssims,
                           train_recovery_ssims, val_recovery_ssims, log_dir):
    """
    Generate and save training progress plots.
    """
    print(f"\nGenerating training progress plots...")
    
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Plot 1: Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Hiding PSNR Over Time
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_hiding_psnrs, 'b-', label='Training Hiding PSNR')
    ax2.plot(epochs, val_hiding_psnrs, 'r-', label='Validation Hiding PSNR')
    ax2.set_title('Hiding PSNR Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Recovery PSNR Over Time
    ax3 = axes[1, 0]
    ax3.plot(epochs, train_recovery_psnrs, 'b-', label='Training Recovery PSNR')
    ax3.plot(epochs, val_recovery_psnrs, 'r-', label='Validation Recovery PSNR')
    ax3.set_title('Recovery PSNR Over Time')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('PSNR (dB)')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Hiding SSIM Over Time
    ax4 = axes[1, 1]
    ax4.plot(epochs, train_hiding_ssims, 'b-', label='Training Hiding SSIM')
    ax4.plot(epochs, val_hiding_ssims, 'r-', label='Validation Hiding SSIM')
    ax4.set_title('Hiding SSIM Over Time')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('SSIM')
    ax4.legend()
    ax4.grid(True)

    # Plot 5: Recovery SSIM Over Time
    ax5 = axes[2, 0]
    ax5.plot(epochs, train_recovery_ssims, 'b-', label='Training Recovery SSIM')
    ax5.plot(epochs, val_recovery_ssims, 'r-', label='Validation Recovery SSIM')
    ax5.set_title('Recovery SSIM Over Time')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('SSIM')
    ax5.legend()
    ax5.grid(True)

    # Plot 6: Hiding vs Recovery PSNR Comparison
    ax6 = axes[2, 1]
    ax6.plot(epochs, train_hiding_psnrs, 'b--', label='Hiding PSNR (Train)')
    ax6.plot(epochs, train_recovery_psnrs, 'g--', label='Recovery PSNR (Train)')
    ax6.plot(epochs, val_hiding_psnrs, 'b-', label='Hiding PSNR (Val)')
    ax6.plot(epochs, val_recovery_psnrs, 'g-', label='Recovery PSNR (Val)')
    ax6.set_title('Hiding vs Recovery PSNR Comparison')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('PSNR (dB)')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(log_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plots saved at: {plot_path}")


def evaluate_model(generator, test_dataset, num_samples=10):
    """
    Comprehensive evaluation of the trained Improved GAN-enhanced model.
    """
    print(f"\nEvaluating Improved GAN-Enhanced model on {num_samples} test samples...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    generator.eval()

    # Select test samples
    test_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))

    all_hiding_psnr = []
    all_recovery_psnr = []
    all_hiding_ssim = []
    all_recovery_ssim = []
    all_hiding_mae = []
    all_recovery_mae = []
    all_hiding_rmse = []
    all_recovery_rmse = []

    # Initialize LPIPS loss function
    lpips_fn = calculate_lpips_loss_fn().to(device)

    all_hiding_lpips = []
    all_recovery_lpips = []

    results_data = []  # For visualizations

    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            cover_tensor, secret_tensor = test_dataset[idx]
            cover_tensor = cover_tensor.unsqueeze(0).to(device)  # Add batch dimension
            secret_tensor = secret_tensor.unsqueeze(0).to(device)

            # Embed secret in cover to get stego
            stego_output = generator.embed(cover_tensor, secret_tensor)

            # Extract secret from stego (USING STego IMAGE ALONE - NO COVER NEEDED)
            recovered_secret = generator.extract(stego_output)

            # Calculate metrics
            hiding_psnr = calculate_psnr(stego_output, cover_tensor)
            recovery_psnr = calculate_psnr(recovered_secret, secret_tensor)
            hiding_ssim = calculate_ssim(stego_output, cover_tensor)
            recovery_ssim = calculate_ssim(recovered_secret, secret_tensor)
            hiding_mae = calculate_mae(stego_output, cover_tensor)
            recovery_mae = calculate_mae(recovered_secret, secret_tensor)
            hiding_rmse = calculate_rmse(stego_output, cover_tensor)
            recovery_rmse = calculate_rmse(recovered_secret, secret_tensor)

            # Calculate LPIPS
            hiding_lpips = lpips_fn(cover_tensor, stego_output).item()
            recovery_lpips = lpips_fn(secret_tensor, recovered_secret).item()

            # Check for NaN/Inf in metrics
            if (torch.isnan(hiding_psnr) or torch.isinf(hiding_psnr) or
                torch.isnan(recovery_psnr) or torch.isinf(recovery_psnr)):
                continue

            # Store metrics
            all_hiding_psnr.append(hiding_psnr.item())
            all_recovery_psnr.append(recovery_psnr.item())
            all_hiding_ssim.append(hiding_ssim.item())
            all_recovery_ssim.append(recovery_ssim.item())
            all_hiding_mae.append(hiding_mae.item())
            all_recovery_mae.append(recovery_mae.item())
            all_hiding_rmse.append(hiding_rmse.item())
            all_recovery_rmse.append(recovery_rmse.item())
            all_hiding_lpips.append(hiding_lpips)
            all_recovery_lpips.append(recovery_lpips)

            # Store for visualization (convert back to [0, 1] range for display)
            results_data.append({
                'cover': (cover_tensor[0] + 1) / 2.0,
                'secret': (secret_tensor[0] + 1) / 2.0,
                'stego': (stego_output[0] + 1) / 2.0,
                'recovered': (recovered_secret[0] + 1) / 2.0,
                'hiding_residual': torch.clamp(torch.abs(stego_output[0] - cover_tensor[0]) * 5, 0, 1), # Amplified for visibility
                'recovery_residual': torch.clamp(torch.abs(recovered_secret[0] - secret_tensor[0]) * 5, 0, 1), # Amplified for visibility
                'metrics': {
                    'hiding_psnr': hiding_psnr.item(),
                    'recovery_psnr': recovery_psnr.item(),
                    'hiding_ssim': hiding_ssim.item(),
                    'recovery_ssim': recovery_ssim.item(),
                    'hiding_mae': hiding_mae.item(),
                    'recovery_mae': recovery_mae.item(),
                    'hiding_rmse': hiding_rmse.item(),
                    'recovery_rmse': recovery_rmse.item(),
                    'hiding_lpips': hiding_lpips,
                    'recovery_lpips': recovery_lpips
                }
            })

            print(f"Sample {i+1}: Hiding PSNR = {hiding_psnr.item():.2f} dB, "
                  f"Recovery PSNR = {recovery_psnr.item():.2f} dB, "
                  f"Hiding SSIM = {hiding_ssim.item():.4f}, "
                  f"Recovery SSIM = {recovery_ssim.item():.4f}, "
                  f"Hiding LPIPS = {hiding_lpips:.4f}")

    # Calculate averages
    avg_hiding_psnr = np.mean(all_hiding_psnr) if all_hiding_psnr else 0
    avg_recovery_psnr = np.mean(all_recovery_psnr) if all_recovery_psnr else 0
    avg_hiding_ssim = np.mean(all_hiding_ssim) if all_hiding_ssim else 0
    avg_recovery_ssim = np.mean(all_recovery_ssim) if all_recovery_ssim else 0
    avg_hiding_mae = np.mean(all_hiding_mae) if all_hiding_mae else 0
    avg_recovery_mae = np.mean(all_recovery_mae) if all_recovery_mae else 0
    avg_hiding_rmse = np.mean(all_hiding_rmse) if all_hiding_rmse else 0
    avg_recovery_rmse = np.mean(all_recovery_rmse) if all_recovery_rmse else 0
    avg_hiding_lpips = np.mean(all_hiding_lpips) if all_hiding_lpips else 0
    avg_recovery_lpips = np.mean(all_recovery_lpips) if all_recovery_lpips else 0

    print(f"\nCOMPREHENSIVE EVALUATION RESULTS:")
    print("=" * 70)
    print(f"{'Metric':<20} {'Hiding':<15} {'Recovery':<15}")
    print("-" * 50)
    print(f"{'PSNR (dB)':<20} {avg_hiding_psnr:.2f}{'':<15} {avg_recovery_psnr:.2f}")
    print(f"{'SSIM':<20} {avg_hiding_ssim:.4f}{'':<15} {avg_recovery_ssim:.4f}")
    print(f"{'LPIPS':<20} {avg_hiding_lpips:.4f}{'':<15} {avg_recovery_lpips:.4f}")
    print(f"{'MAE':<20} {avg_hiding_mae:.4f}{'':<15} {avg_recovery_mae:.4f}")
    print(f"{'RMSE':<20} {avg_hiding_rmse:.4f}{'':<15} {avg_recovery_rmse:.4f}")

    # Calculate standard deviations
    std_hiding_psnr = np.std(all_hiding_psnr) if all_hiding_psnr else 0
    std_recovery_psnr = np.std(all_recovery_psnr) if all_recovery_psnr else 0
    std_hiding_ssim = np.std(all_hiding_ssim) if all_hiding_ssim else 0
    std_recovery_ssim = np.std(all_recovery_ssim) if all_recovery_ssim else 0
    std_hiding_lpips = np.std(all_hiding_lpips) if all_hiding_lpips else 0
    std_recovery_lpips = np.std(all_recovery_lpips) if all_recovery_lpips else 0

    print("\nStandard Deviations:")
    print(f"{'PSNR (dB)':<20} {std_hiding_psnr:.2f}{'':<15} {std_recovery_psnr:.2f}")
    print(f"{'SSIM':<20} {std_hiding_ssim:.4f}{'':<15} {std_recovery_ssim:.4f}")
    print(f"{'LPIPS':<20} {std_hiding_lpips:.4f}{'':<15} {std_recovery_lpips:.4f}")

    return results_data


def create_visualization_grid(results_data, save_path='improved_gan_steganography_results.png'):
    """
    Create a comprehensive visualization grid showing results.
    """
    print(f"\nCreating Improved GAN visualization grid at: {save_path}")

    num_samples = len(results_data)
    if num_samples == 0:
        print("No results to visualize")
        return

    fig, axes = plt.subplots(6, num_samples, figsize=(4*num_samples, 24))

    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    titles = ['Cover', 'Secret', 'Stego', 'Recovered', 'Hiding Residual', 'Recovery Residual']

    for i in range(6):  # 6 rows: Cover, Secret, Stego, Recovered, Hiding Residual, Recovery Residual
        for j in range(num_samples):  # For each sample
            img_data = None
            if i == 0:  # Cover
                img_data = results_data[j]['cover'].permute(1, 2, 0).cpu().numpy()
                title = f'{titles[i]}\nSample {j+1}'
            elif i == 1:  # Secret
                img_data = results_data[j]['secret'].permute(1, 2, 0).cpu().numpy()
                title = f'{titles[i]}\nSample {j+1}'
            elif i == 2:  # Stego
                img_data = results_data[j]['stego'].permute(1, 2, 0).cpu().numpy()
                # Include hiding metrics in the title
                metrics = results_data[j]['metrics']
                hiding_psnr = metrics['hiding_psnr']
                hiding_ssim = metrics['hiding_ssim']
                hiding_lpips = metrics['hiding_lpips']
                title = f'{titles[i]}\nSample {j+1}\nPSNR: {hiding_psnr:.2f} dB\nSSIM: {hiding_ssim:.4f}\nLPIPS: {hiding_lpips:.4f}'
            elif i == 3:  # Recovered
                img_data = results_data[j]['recovered'].permute(1, 2, 0).cpu().numpy()
                # Include recovery metrics in the title
                metrics = results_data[j]['metrics']
                recovery_psnr = metrics['recovery_psnr']
                recovery_ssim = metrics['recovery_ssim']
                recovery_lpips = metrics['recovery_lpips']
                title = f'{titles[i]}\nSample {j+1}\nPSNR: {recovery_psnr:.2f} dB\nSSIM: {recovery_ssim:.4f}\nLPIPS: {recovery_lpips:.4f}'
            elif i == 4:  # Hiding Residual
                img_data = results_data[j]['hiding_residual'].permute(1, 2, 0).cpu().numpy()
                title = f'{titles[i]}\nSample {j+1}'
            elif i == 5:  # Recovery Residual
                img_data = results_data[j]['recovery_residual'].permute(1, 2, 0).cpu().numpy()
                title = f'{titles[i]}\nSample {j+1}'

            axes[i, j].imshow(img_data)
            axes[i, j].set_title(title)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Improved GAN visualization grid saved at: {save_path}")


def main(config_path="config.json"):
    """
    Main function using config file for Improved GAN-enhanced steganography.
    """
    print("Improved GAN-Enhanced Steganography Training: Embed Secret in Cover, Extract from Stego")
    print("=" * 90)

    # Load configuration from file
    config = load_config(config_path)

    # Enhanced parameters from config
    img_size = config['data']['img_size']
    num_epochs = config['training']['num_epochs']
    log_dir = config['training']['log_dir']
    alpha_hid = config['training']['alpha_hid']
    alpha_rec = config['training']['alpha_rec'] 
    alpha_adv = config['training'].get('alpha_adv', 0.02)  # New parameter for adversarial loss
    alpha_lpips = config['training'].get('alpha_lpips', 0.5)  # New parameter for LPIPS loss

    # Load dataset
    possible_dirs = [config['data']['image_dir'], "dataset", "images", "./"]
    image_dir = None
    dataset = None

    for dir_name in possible_dirs:
        if os.path.isdir(dir_name):
            try:
                dataset = ImageSteganographyDataset(dir_name, img_size=img_size)
                if len(dataset) > 0:
                    image_dir = dir_name
                    break
            except Exception as e:
                print(f"Could not load dataset from {dir_name}: {e}")
                continue

    if dataset is None or len(dataset) == 0:
        raise ValueError("No valid dataset found in possible directories: " + str(possible_dirs))

    print(f"Loaded {len(dataset)} images from {image_dir}")
    print(f"Training for {num_epochs} epochs with image size {img_size}x{img_size}")
    print(f"Loss weights - Hiding: {alpha_hid}, Recovery: {alpha_rec}, Adversarial: {alpha_adv}, LPIPS: {alpha_lpips}")

    # Split dataset into train, validation, and test
    train_size = int(config['data']['train_ratio'] * len(dataset))
    val_size = int(config['data']['val_ratio'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")

    if val_size == 0 or test_size == 0:
        print("Warning: Dataset too small for proper train/val/test split. Using minimal splits.")
        train_size = max(1, len(dataset) - 2)
        val_size = 1 if len(dataset) >= 2 else 0
        test_size = 1 if len(dataset) >= 3 else 0

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Initialize models
    improved_gan_steg_model = ImprovedGANSteganographyModel()
    improved_discriminator = ImprovedDiscriminator()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    improved_gan_steg_model = improved_gan_steg_model.to(device)
    improved_discriminator = improved_discriminator.to(device)

    print(f"\nImproved GAN-Enhanced Steganography Model parameters: {sum(p.numel() for p in improved_gan_steg_model.parameters()):,}")
    print(f"Improved Discriminator parameters: {sum(p.numel() for p in improved_discriminator.parameters()):,}")

    # Create dataloaders with optimizations for large datasets
    batch_size = config['training']['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Train Improved GAN-Enhanced Steganography model with enhanced logging
    print("\nStarting Improved GAN-Enhanced Steganography model training (224x224) with enhanced logging...")
    trained_generator, trained_discriminator = train_improved_gan_steganography_model(
        improved_gan_steg_model,
        improved_discriminator,
        train_dataset,
        num_epochs=num_epochs,
        val_dataset=val_dataset,
        log_dir=log_dir,
        config=config
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED FOR IMPROVED GAN-ENHANCED STEGANOGRAPHY MODEL")
    print("=" * 60)

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results_data = evaluate_model(trained_generator, val_dataset, num_samples=min(5, len(val_dataset)))

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results_data = evaluate_model(trained_generator, test_dataset, num_samples=min(5, len(test_dataset)))

    # Create visualization grid using test results
    create_visualization_grid(test_results_data, save_path='improved_gan_steganography_test_results.png')

    # Save the trained models
    torch.save({
        'generator_state_dict': trained_generator.state_dict(),
        'discriminator_state_dict': trained_discriminator.state_dict(),
        'model_type': 'ImprovedGANSteganographyModel',
        'img_size': img_size,
    }, 'improved_gan_steganography_model_50k.pth')

    print(f"\nImproved GAN-Enhanced Steganography model saved as 'improved_gan_steganography_model_50k.pth'")
    print(f"TensorBoard logs saved in '{log_dir}' directory")
    print("To view TensorBoard logs, run: tensorboard --logdir=" + log_dir)

    # Demonstrate the process
    print("\n" + "="*60)
    print("DEMONSTRATION: Embed and Extract Process")
    print("="*60)

    if len(test_dataset) > 0:
        cover_tensor, secret_tensor = test_dataset[0]
        cover_tensor = cover_tensor.unsqueeze(0).to(device)
        secret_tensor = secret_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            stego_image = trained_generator.embed(cover_tensor, secret_tensor)
            recovered_secret = trained_generator.extract(stego_image)

        hiding_psnr = calculate_psnr(stego_image, cover_tensor)
        recovery_psnr = calculate_psnr(recovered_secret, secret_tensor)
        hiding_ssim = calculate_ssim(stego_image, cover_tensor)
        recovery_ssim = calculate_ssim(recovered_secret, secret_tensor)
        
        # Calculate LPIPS
        lpips_fn = calculate_lpips_loss_fn().to(device)
        hiding_lpips = lpips_fn(cover_tensor, stego_image).item()
        recovery_lpips = lpips_fn(secret_tensor, recovered_secret).item()

        print(f"Embedding & Extraction Results:")
        print(f"  Cover-Stego PSNR: {hiding_psnr.item():.2f} dB")
        print(f"  Cover-Stego SSIM: {hiding_ssim.item():.4f}")
        print(f"  Cover-Stego LPIPS: {hiding_lpips:.4f}")
        print(f"  Original-Recovered Secret PSNR: {recovery_psnr.item():.2f} dB")
        print(f"  Original-Recovered Secret SSIM: {recovery_ssim.item():.4f}")
        print(f"  Original-Recovered Secret LPIPS: {recovery_lpips:.4f}")

        # Test discriminator
        with torch.no_grad():
            disc_on_cover = trained_discriminator(cover_tensor).mean().item()
            disc_on_stego = trained_discriminator(stego_image).mean().item()
        
        print(f"  Discriminator output on cover: {disc_on_cover:.4f}")
        print(f"  Discriminator output on stego: {disc_on_stego:.4f}")


if __name__ == "__main__":
    main()  # Calls main with default config.json