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
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import math
from collections import OrderedDict

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Configuration file loading
def load_config(config_path="config.json"):
    """
    Load configuration from a JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Dataset class for steganography with 256x256 images
class SteganographyDataset(Dataset):
    """
    Dataset for steganography - loads pairs of images to use as cover and secret.
    """
    def __init__(self, image_dir, img_size=256, transform=None):
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

        # Default transformation to normalize images for 256x256
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

# Utility functions for metrics
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

# #####################################################
# APPROACH 1: High-accuracy image steganography with INN and GAN
# #####################################################

class InvBlock(nn.Module):
    """
    Invertible block for INN-based steganography
    """
    def __init__(self, subnet_constructor, channel_num, channel_split_num):
        super(InvBlock, self).__init__()
        self.split_len1 = channel_split_num 
        self.split_len2 = channel_num - channel_split_num
        self.cl = subnet_constructor(self.split_len1, self.split_len2)
        self.cr = subnet_constructor(self.split_len2, self.split_len1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        
        if not rev:
            y1 = x1 + self.cl(x2)
            y2 = x2 + self.cr(y1)
        else:
            y2 = x2 - self.cr(x1)
            y1 = x1 - self.cl(y2)
        
        return torch.cat((y1, y2), 1)

    def inverse(self, x):
        return self.forward(x, rev=True)

class InvNet(nn.Module):
    """
    Invertible Neural Network for steganography
    """
    def __init__(self, channel_in=6, channel_out=3, subnet_constructor=None):
        super(InvNet, self).__init__()
        if subnet_constructor is None:
            subnet_constructor = self.build_default_subnet
            
        # Define invertible blocks
        self.block1 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block2 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block3 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block4 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block5 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block6 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block7 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block8 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block9 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block10 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block11 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block12 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block13 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block14 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block15 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        self.block16 = InvBlock(subnet_constructor, channel_in, channel_in // 2)
        
        # Final output layer
        self.final_layer = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1)

    def build_default_subnet(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Conv2d(dim_in, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, dim_out, 3, padding=1)
        )

    def forward(self, x, rev=False):
        if not rev:
            # Forward pass: embed secret into cover
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
            x = self.block13(x)
            x = self.block14(x)
            x = self.block15(x)
            x = self.block16(x)
            x = self.final_layer(x)
            x = torch.tanh(x)  # Normalize output to [-1, 1]
        else:
            # Reverse pass: extract secret from stego
            x = self.final_layer(x)
            x = self.block16(x, rev=True)
            x = self.block15(x, rev=True)
            x = self.block14(x, rev=True)
            x = self.block13(x, rev=True)
            x = self.block12(x, rev=True)
            x = self.block11(x, rev=True)
            x = self.block10(x, rev=True)
            x = self.block9(x, rev=True)
            x = self.block8(x, rev=True)
            x = self.block7(x, rev=True)
            x = self.block6(x, rev=True)
            x = self.block5(x, rev=True)
            x = self.block4(x, rev=True)
            x = self.block3(x, rev=True)
            x = self.block2(x, rev=True)
            x = self.block1(x, rev=True)
        
        return x

class INNSteganographyModel(nn.Module):
    """
    Steganography model using Invertible Neural Networks (INN) with GAN
    """
    def __init__(self):
        super(INNSteganographyModel, self).__init__()
        
        # Use the INN for embedding and extraction
        self.inn = InvNet(channel_in=6, channel_out=3)
        
        # Discriminator for GAN
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 256x256 -> 128x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def embed(self, cover, secret):
        """
        Embed secret into cover to produce stego image
        """
        # Concatenate cover and secret
        combined = torch.cat([cover, secret], dim=1)
        
        # Generate stego image using INN
        stego = self.inn(combined)
        
        # Ensure output is close to the cover but with secret embedded
        stego = cover + 0.1 * stego  # Small perturbation to maintain cover appearance
        stego = torch.clamp(stego, -1, 1)  # Clamp to valid range
        
        return stego

    def extract(self, stego):
        """
        Extract secret from stego image (using INN in reverse)
        """
        # For extraction, we need a separate INN for reconstruction
        # In practice, we'd train a separate extraction network
        # Here we use the same architecture but in reverse
        # We'll use a simplified approach for this implementation
        
        # In a real implementation, this would use the reverse INN
        # For demonstration, we'll create a simple extraction network
        if not hasattr(self, 'extract_net'):
            self.extract_net = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.Tanh()
            ).to(stego.device)
        
        extracted = self.extract_net(stego)
        return extracted

    def discriminate(self, x):
        """
        Discriminator output for GAN training
        """
        return self.discriminator(x)


# #####################################################
# APPROACH 2: High invisibility image steganography with DWT and GAN
# #####################################################

class DWT(nn.Module):
    """
    Discrete Wavelet Transform
    """
    def __init__(self):
        super(DWT, self).__init__()
        # Haar wavelet filter
        self.requires_grad = False

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Get even and odd rows/columns
        x_even_row = x[:, :, ::2, :]
        x_odd_row = x[:, :, 1::2, :]
        x_even_col = x[:, :, :, ::2]
        x_odd_col = x[:, :, :, 1::2]
        
        # For a proper implementation, we would use convolution with Haar filters
        # For simplicity, we'll approximate this with average and difference operations
        LL = (x_even_row[:, :, :, ::2] + x_even_row[:, :, :, 1::2] + 
              x_odd_row[:, :, :, ::2] + x_odd_row[:, :, :, 1::2]) / 4
        LH = (x_odd_row[:, :, :, ::2] + x_odd_row[:, :, :, 1::2] - 
              x_even_row[:, :, :, ::2] - x_even_row[:, :, :, 1::2]) / 4
        HL = (x_even_row[:, :, :, 1::2] + x_odd_row[:, :, :, 1::2] - 
              x_even_row[:, :, :, ::2] - x_odd_row[:, :, :, ::2]) / 4
        HH = (x_even_row[:, :, :, ::2] + x_odd_row[:, :, :, 1::2] - 
              x_even_row[:, :, :, 1::2] - x_odd_row[:, :, :, ::2]) / 4
        
        # Concatenate the four subbands
        return torch.cat([LL, LH, HL, HH], dim=1)

class IWT(nn.Module):
    """
    Inverse Discrete Wavelet Transform
    """
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # x shape: [B, 4*C, H, W] where we have 4 subbands
        # For simplicity, this is a placeholder implementation
        B, C, H, W = x.shape
        C = C // 4  # Each subband has C channels
        
        # Split the input into 4 subbands
        LL = x[:, :C, :, :]
        LH = x[:, C:2*C, :, :]
        HL = x[:, 2*C:3*C, :, :]
        HH = x[:, 3*C:, :, :]
        
        # Approximate reconstruction
        # This is a simplified version; a true IWT would be more complex
        reconstructed = LL + LH + HL + HH
        return torch.clamp(reconstructed, -1, 1)

class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism for steganography
    """
    def __init__(self, channels):
        super(MultiScaleAttention, self).__init__()
        self.channels = channels
        
        # Multi-scale convolutions
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        
        # Attention weights
        self.attention_conv = nn.Conv2d(channels * 3, 3, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        feat1 = self.conv1(x)
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        
        # Concatenate features
        feats = torch.cat([feat1, feat3, feat5], dim=1)
        
        # Compute attention
        attention = self.attention_conv(feats)
        attention = self.softmax(attention)
        
        # Apply attention
        attended = feat1 * attention[:, 0:1, :, :] + \
                   feat3 * attention[:, 1:2, :, :] + \
                   feat5 * attention[:, 2:3, :, :]
        
        return attended

class FusionModule(nn.Module):
    """
    Fusion module to combine cover and secret information
    """
    def __init__(self, channels=64):
        super(FusionModule, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        self.attention = MultiScaleAttention(channels)
        
        # Processing layers
        self.process = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),  # 3 for cover, 3 for secret -> 4 subbands each
            nn.ReLU(inplace=True),
            self.attention,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, cover, secret):
        # Apply DWT to both cover and secret
        cover_dwt = self.dwt(cover)
        secret_dwt = self.dwt(secret)
        
        # Combine them
        combined = torch.cat([cover_dwt, secret_dwt], dim=1)
        
        # Process with attention
        processed = self.process(combined)
        
        # Apply IWT to get final stego
        stego = self.iwt(processed)
        
        return stego

class DWTSteganographyModel(nn.Module):
    """
    Steganography model using DWT with GAN for high invisibility
    """
    def __init__(self):
        super(DWTSteganographyModel, self).__init__()
        
        # Fusion module to combine cover and secret
        self.fusion_module = FusionModule(channels=64)
        
        # Extraction network (simplified for this implementation)
        self.extract_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Discriminator for GAN
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 256x256 -> 128x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def embed(self, cover, secret):
        """
        Embed secret into cover using DWT
        """
        stego = self.fusion_module(cover, secret)
        
        # Ensure output is close to the cover but with secret embedded
        stego = cover + 0.1 * stego  # Small perturbation to maintain cover appearance
        stego = torch.clamp(stego, -1, 1)  # Clamp to valid range
        
        return stego

    def extract(self, stego):
        """
        Extract secret from stego image
        """
        return self.extract_net(stego)

    def discriminate(self, x):
        """
        Discriminator output for GAN training
        """
        return self.discriminator(x)


# #####################################################
# APPROACH 3: Image steganography based on Wavelet Transform and U-Net GAN
# #####################################################

class UNetBlock(nn.Module):
    """
    U-Net convolutional block
    """
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False, dropout_rate=0.5):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if down else nn.ReLU()
        )
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

class UNetGenerator(nn.Module):
    """
    U-Net Generator for steganography
    """
    def __init__(self, input_channels=6, output_channels=3, features=[64, 128, 256, 512, 512, 512, 512, 512]):
        super(UNetGenerator, self).__init__()
        
        # Encoder (downsampling)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.residual_connections = nn.ModuleList()
        
        # Down sampling layers
        for feature in features:
            self.downs.append(UNetBlock(input_channels, feature, down=True, use_dropout=False))
            input_channels = feature
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features[-1], features[-1], 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(features[-1], features[-1], 4, 2, 1),
            nn.BatchNorm2d(features[-1])
        )
        
        # Up sampling layers
        features = features[::-1]  # reverse the list
        for idx, feature in enumerate(features):
            if idx == len(features) - 1:
                self.ups.append(
                    UNetBlock(2 * feature, output_channels, down=False, use_dropout=False)
                )
            else:
                self.ups.append(
                    UNetBlock(2 * feature, feature, down=False, use_dropout=True)
                )
        
        # Final layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(features[0], output_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        skip_connections = []
        
        # Forward through downsampling layers
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        
        # Reverse skip connections for upsampling
        skip_connections = skip_connections[::-1]
        
        # Through bottleneck
        x = self.bottleneck(x)
        
        # Through upsampling layers
        for idx, up in enumerate(self.ups):
            x = up(x)
            if idx < len(skip_connections) - 1:  # Don't add skip connection to final layer
                x = torch.cat((x, skip_connections[idx]), dim=1)
        
        return self.final_layer(x)

class Steganalyzer(nn.Module):
    """
    Steganalyzer for detecting hidden information (Xu-Net or Yedrouj-Net based)
    """
    def __init__(self, input_channels=3):
        super(Steganalyzer, self).__init__()
        
        # Simpler version of steganalyzer network
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class WaveletUNetSteganographyModel(nn.Module):
    """
    Steganography model using Wavelet Transform and U-Net GAN
    """
    def __init__(self):
        super(WaveletUNetSteganographyModel, self).__init__()
        
        # U-Net generator
        self.generator = UNetGenerator(input_channels=6, output_channels=3)  # 3 for cover + 3 for secret
        
        # Extraction network
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Dual steganalyzer discriminators (simplified)
        self.steganalyzer1 = Steganalyzer(input_channels=3)  # Xu-Net based
        self.steganalyzer2 = Steganalyzer(input_channels=3)  # Yedrouj-Net based

    def embed(self, cover, secret):
        """
        Embed secret into cover using U-Net
        """
        # Concatenate cover and secret
        combined = torch.cat([cover, secret], dim=1)
        
        # Generate stego using U-Net
        stego = self.generator(combined)
        
        # Apply double tanh activation (lambda=60 as specified in paper)
        # This helps in keeping the changes subtle
        stego = cover + 0.05 * torch.tanh(60 * stego)  # Using lambda=60 as specified
        
        return torch.clamp(stego, -1, 1)

    def extract(self, stego):
        """
        Extract secret from stego image
        """
        return self.extractor(stego)

    def discriminate(self, x):
        """
        Discriminate using dual steganalyzers
        """
        # Get outputs from both steganalyzers
        out1 = self.steganalyzer1(x)
        out2 = self.steganalyzer2(x)
        
        # Return average of both outputs
        return (out1 + out2) / 2


# #####################################################
# Training functions for all approaches
# #####################################################

def train_approach1(config):
    """
    Train Approach 1: High-accuracy image steganography with INN and GAN
    """
    print(f"\nStarting training for Approach 1: High-accuracy INN-based steganography...")
    
    # Load dataset
    dataset = SteganographyDataset(config['data']['image_dir'], img_size=256)
    
    # Split dataset
    train_size = int(config['data']['train_ratio'] * len(dataset))
    val_size = int(config['data']['val_ratio'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Initialize model
    model = INNSteganographyModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizers
    opt_g = optim.Adam(list(model.inn.parameters()), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=config['training']['num_epochs'])
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=config['training']['num_epochs'])
    
    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    lambda_p, lambda_f, lambda_e, lambda_a = 1.0, 1.0, 1.0, 0.1  # Loss weights
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for cover, secret in progress_bar:
            cover = cover.to(device)
            secret = secret.to(device)
            
            # Train generator
            opt_g.zero_grad()
            
            # Embed
            stego = model.embed(cover, secret)
            
            # Extract
            extracted = model.extract(stego)
            
            # Generator losses
            loss_p = F.mse_loss(stego, cover)  # Pixel loss for invisibility
            loss_e = F.mse_loss(extracted, secret)  # Extraction accuracy loss
            loss_adv = 1 - model.discriminate(stego).mean()  # Adversarial loss
            
            # Total generator loss
            g_loss = lambda_p * loss_p + lambda_e * loss_e + lambda_a * loss_adv
            
            g_loss.backward()
            opt_g.step()
            
            # Train discriminator
            opt_d.zero_grad()
            
            # Real loss
            real_pred = model.discriminate(cover)
            real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
            
            # Fake loss
            fake_pred = model.discriminate(stego.detach())
            fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()
            
            total_loss += g_loss.item()
            
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}',
                'LR_G': f'{scheduler_g.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()
    
    print("Training completed for Approach 1!")
    return model, test_dataset

def train_approach2(config):
    """
    Train Approach 2: High invisibility image steganography with DWT and GAN
    """
    print(f"\nStarting training for Approach 2: High invisibility DWT-based steganography...")
    
    # Load dataset
    dataset = SteganographyDataset(config['data']['image_dir'], img_size=256)
    
    # Split dataset
    train_size = int(config['data']['train_ratio'] * len(dataset))
    val_size = int(config['data']['val_ratio'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Initialize model
    model = DWTSteganographyModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizers
    opt_g = optim.Adam(list(model.fusion_module.parameters()) + list(model.extract_net.parameters()), 
                       lr=1e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(model.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.StepLR(opt_g, step_size=20, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(opt_d, step_size=20, gamma=0.5)
    
    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    lambda_p, lambda_s, lambda_f, lambda_e, lambda_a = 100, 1, 0.1, 1, 1  # Loss weights
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for cover, secret in progress_bar:
            cover = cover.to(device)
            secret = secret.to(device)
            
            # Train generator
            opt_g.zero_grad()
            
            # Embed
            stego = model.embed(cover, secret)
            
            # Extract
            extracted = model.extract(stego)
            
            # Generator losses
            loss_p = F.mse_loss(stego, cover)  # Pixel loss (high weight for invisibility)
            loss_e = F.mse_loss(extracted, secret)  # Extraction accuracy loss
            loss_adv = 1 - model.discriminate(stego).mean()  # Adversarial loss
            
            # Total generator loss
            g_loss = lambda_p * loss_p + lambda_e * loss_e + lambda_a * loss_adv
            
            g_loss.backward()
            opt_g.step()
            
            # Train discriminator
            opt_d.zero_grad()
            
            # Real loss
            real_pred = model.discriminate(cover)
            real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
            
            # Fake loss
            fake_pred = model.discriminate(stego.detach())
            fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()
            
            total_loss += g_loss.item()
            
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}',
                'LR_G': f'{scheduler_g.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()
    
    print("Training completed for Approach 2!")
    return model, test_dataset

def train_approach3(config):
    """
    Train Approach 3: Image steganography based on Wavelet Transform and U-Net GAN
    """
    print(f"\nStarting training for Approach 3: U-Net based steganography...")
    
    # Load dataset
    dataset = SteganographyDataset(config['data']['image_dir'], img_size=256)
    
    # Split dataset
    train_size = int(config['data']['train_ratio'] * len(dataset))
    val_size = int(config['data']['val_ratio'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Initialize model
    model = WaveletUNetSteganographyModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizers
    opt_g = optim.Adam(list(model.generator.parameters()) + list(model.extractor.parameters()), 
                       lr=1e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(list(model.steganalyzer1.parameters()) + list(model.steganalyzer2.parameters()), 
                       lr=1e-4, betas=(0.5, 0.999))
    
    # Learning rate schedulers (following paper's schedule: decay by 0.4 every 20 epochs)
    scheduler_g = optim.lr_scheduler.StepLR(opt_g, step_size=20, gamma=0.4)
    scheduler_d = optim.lr_scheduler.StepLR(opt_d, step_size=20, gamma=0.4)
    
    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    eta, alpha, beta = 1.0, 1.0, 1e-7  # Loss weights from the paper
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for cover, secret in progress_bar:
            cover = cover.to(device)
            secret = secret.to(device)
            
            # Train generator
            opt_g.zero_grad()
            
            # Embed
            stego = model.embed(cover, secret)
            
            # Extract
            extracted = model.extract(stego)
            
            # Generator losses
            loss_adv = 1 - model.discriminate(stego).mean()  # Adversarial loss
            loss_rec = F.mse_loss(extracted, secret)  # Reconstruction loss
            loss_hid = F.mse_loss(stego, cover)  # Hiding loss
            
            # Total generator loss (from the paper's formula)
            g_loss = eta * loss_adv + alpha * loss_rec + beta * loss_hid
            
            g_loss.backward()
            opt_g.step()
            
            # Train discriminator (steganalyzers)
            opt_d.zero_grad()
            
            # Real loss (cover images should be classified as "not stego")
            real_pred = model.discriminate(cover)
            real_loss = F.binary_cross_entropy(real_pred, torch.zeros_like(real_pred))
            
            # Fake loss (stego images should be classified as "stego")
            fake_pred = model.discriminate(stego.detach())
            fake_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()
            
            total_loss += g_loss.item()
            
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}',
                'LR_G': f'{scheduler_g.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()
    
    print("Training completed for Approach 3!")
    return model, test_dataset

def evaluate_model(model, test_dataset, num_samples=5, device=None):
    """
    Evaluate the trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Select test samples
    test_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    results = []
    
    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            cover_tensor, secret_tensor = test_dataset[idx]
            cover_tensor = cover_tensor.unsqueeze(0).to(device)  # Add batch dimension
            secret_tensor = secret_tensor.unsqueeze(0).to(device)
            
            # Embed secret in cover to get stego
            stego_output = model.embed(cover_tensor, secret_tensor)
            
            # Extract secret from stego
            recovered_secret = model.extract(stego_output)
            
            # Calculate metrics
            hiding_psnr = calculate_psnr(stego_output, cover_tensor)
            recovery_psnr = calculate_psnr(recovered_secret, secret_tensor)
            hiding_ssim = calculate_ssim(stego_output, cover_tensor)
            recovery_ssim = calculate_ssim(recovered_secret, secret_tensor)
            
            results.append({
                'cover': (cover_tensor[0] + 1) / 2.0,
                'secret': (secret_tensor[0] + 1) / 2.0,
                'stego': (stego_output[0] + 1) / 2.0,
                'recovered': (recovered_secret[0] + 1) / 2.0,
                'hiding_psnr': hiding_psnr.item(),
                'recovery_psnr': recovery_psnr.item(),
                'hiding_ssim': hiding_ssim.item(),
                'recovery_ssim': recovery_ssim.item()
            })
            
            print(f"Sample {i+1}: Hiding PSNR = {hiding_psnr.item():.2f} dB, "
                  f"Recovery PSNR = {recovery_psnr.item():.2f} dB, "
                  f"Hiding SSIM = {hiding_ssim.item():.4f}, "
                  f"Recovery SSIM = {recovery_ssim.item():.4f}")
    
    return results

def run_all_approaches():
    """
    Run all three approaches
    """
    print("="*80)
    print("RUNNING ALL THREE STEGANOGRAPHY APPROACHES")
    print("="*80)
    
    # Load config
    config = load_config("config.json")
    
    # Update config to use 256x256 images and 50 epochs
    config['data']['img_size'] = 256
    config['training']['num_epochs'] = 50
    config['training']['batch_size'] = 2  # Adjust based on GPU memory
    
    print(f"Using config: {config}")
    
    # Approach 1: High-accuracy INN-based steganography
    print("\n" + "="*60)
    print("APPROACH 1: High-accuracy image steganography with INN and GAN")
    print("="*60)
    model1, test_dataset1 = train_approach1(config)
    
    print("\nEvaluating Approach 1...")
    results1 = evaluate_model(model1, test_dataset1, num_samples=3)
    
    # Approach 2: High invisibility DWT-based steganography
    print("\n" + "="*60)
    print("APPROACH 2: High invisibility image steganography with DWT and GAN")
    print("="*60)
    model2, test_dataset2 = train_approach2(config)
    
    print("\nEvaluating Approach 2...")
    results2 = evaluate_model(model2, test_dataset2, num_samples=3)
    
    # Approach 3: U-Net based steganography
    print("\n" + "="*60)
    print("APPROACH 3: Image steganography based on Wavelet Transform and U-Net GAN")
    print("="*60)
    model3, test_dataset3 = train_approach3(config)
    
    print("\nEvaluating Approach 3...")
    results3 = evaluate_model(model3, test_dataset3, num_samples=3)
    
    print("\n" + "="*80)
    print("ALL THREE APPROACHES COMPLETED!")
    print("="*80)
    
    # Print summary results
    print(f"\nSUMMARY:")
    print(f"Approach 1 - Average Hiding PSNR: {np.mean([r['hiding_psnr'] for r in results1]):.2f} dB")
    print(f"Approach 1 - Average Recovery PSNR: {np.mean([r['recovery_psnr'] for r in results1]):.2f} dB")
    
    print(f"Approach 2 - Average Hiding PSNR: {np.mean([r['hiding_psnr'] for r in results2]):.2f} dB")
    print(f"Approach 2 - Average Recovery PSNR: {np.mean([r['recovery_psnr'] for r in results2]):.2f} dB")
    
    print(f"Approach 3 - Average Hiding PSNR: {np.mean([r['hiding_psnr'] for r in results3]):.2f} dB")
    print(f"Approach 3 - Average Recovery PSNR: {np.mean([r['recovery_psnr'] for r in results3]):.2f} dB")
    
    # Save models
    torch.save(model1.state_dict(), 'approach1_inn_steganography.pth')
    torch.save(model2.state_dict(), 'approach2_dwt_steganography.pth')
    torch.save(model3.state_dict(), 'approach3_unet_steganography.pth')
    
    print(f"\nModels saved as:")
    print("- approach1_inn_steganography.pth")
    print("- approach2_dwt_steganography.pth")
    print("- approach3_unet_steganography.pth")
    
    return model1, model2, model3, results1, results2, results3

if __name__ == "__main__":
    # Run all approaches
    model1, model2, model3, results1, results2, results3 = run_all_approaches()