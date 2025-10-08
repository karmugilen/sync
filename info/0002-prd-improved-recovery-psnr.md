# 0002-PRD: Improving Recovery PSNR in GAN-Enhanced Steganography

## 1. Introduction/Overview

This PRD outlines requirements for significantly improving the recovery PSNR in our GAN-enhanced steganography system. Current implementation achieves only ~18-20 dB recovery PSNR, which is insufficient for practical applications. The goal is to increase recovery PSNR to >35 dB while maintaining good hiding performance (>40 dB PSNR for stego-cover comparison).

The main issues identified:
1. Weak extraction network architecture that cannot effectively recover hidden information
2. Imbalanced loss function weighting favoring hiding over recovery
3. Insufficient attention to recovery quality during GAN training
4. Lack of specialized architectures for better feature preservation

## 2. Goals

1. Implement an improved neural network architecture that can embed AND extract information with high fidelity
2. Increase recovery PSNR from ~18 dB to >35 dB on validation set
3. Maintain hiding PSNR >40 dB for imperceptibility
4. Improve SSIM scores for both hiding (>0.95) and recovery (>0.92)
5. Implement attention mechanisms to focus on important features during extraction
6. Balance loss functions to optimize both hiding and recovery performance
7. Add specialized training techniques to improve recovery capability

## 3. User Stories

- As a user, I want to hide images in other images and recover them with high quality so that the recovered images are usable
- As a developer, I want a steganography system with balanced performance between hiding and recovery so that both aspects work well
- As a researcher, I want to evaluate the system with improved metrics so that I can compare it with other approaches
- As a security professional, I want reliable extraction even after image processing operations so that hidden information is preserved

## 4. Functional Requirements

1. The system must accept image files as cover media in common formats (PNG, JPG, BMP, TIFF, JPEG)
2. The system must accept image files as secret media in common formats (PNG, JPG, BMP, TIFF, JPEG)
3. The system must generate steganographic images that appear visually identical to the original cover image (PSNR >40 dB)
4. The system must provide a function to extract the hidden image from the stego image with high fidelity (PSNR >35 dB)
5. The system must use an enhanced GAN architecture with improved extraction network
6. The system must transform images to a standard size (224x224) for processing
7. The system must normalize images to the [-1, 1] range for neural network processing
8. The system must support loading and using enhanced pre-trained models with better recovery capability
9. The system must calculate and report quality metrics (PSNR, SSIM, MAE, RMSE, LPIPS) for both hiding and recovery
10. The system must provide visualization grids showing cover, secret, stego, recovered images, and residuals
11. The system must support GPU acceleration if available
12. The system must handle out-of-memory errors gracefully
13. The system must support dataset loading from multiple folder structures (train/validation/test subfolders or direct directory)
14. The system must save trained models in PyTorch format (.pth)
15. The system must allow configuration through a JSON file (config.json)
16. The system must include attention mechanisms in the extraction network
17. The system must implement a balanced loss function favoring both imperceptibility and recovery
18. The system must include skip connections to preserve fine details during extraction

## 5. Non-Goals (Out of Scope)

1. Audio or video steganography - this system focuses specifically on image-based steganography
2. Text-based steganography - images are used as both cover and secret
3. Steganalysis or detection of steganographic content
4. Real-time steganography for video streaming
5. Built-in encryption of the data (though this could be part of preprocessing)
6. Web-based user interface (though could be added later)
7. Blockchain-based verification of hidden data
8. Support for embedding multiple secrets in one cover image
9. Steganography with non-image data as cover (e.g., audio covers with image secrets)

## 6. Design Considerations

1. The neural network architecture must include separate embedding and extraction networks with attention mechanisms
2. The embedding network must accept concatenated cover and secret images (6 channels) and output a 3-channel stego image
3. The extraction network must accept a 3-channel stego image and output a 3-channel secret image with high fidelity
4. Attention modules should be added to help the extraction network focus on relevant features
5. Skip connections should preserve important details during the extraction process
6. The GUI interface should provide image previews for cover, secret, and stego images
7. The GUI interface should provide progress bars for long-running operations
8. The GUI interface should support both embedding and extraction operations
9. The system should normalize images to [-1, 1] range using mean=(0.5, 0.5, 0.5) and std=(0.5, 0.5, 0.5)
10. The system should handle images of various sizes by resizing to 224x224
11. Loss functions should be rebalanced to give appropriate weight to recovery performance

## 7. Technical Considerations

1. The system must use PyTorch for neural network implementation
2. The system must use PIL for image loading and saving
3. The system must use torchvision for image transformations
4. The system must use CUDA if available for GPU acceleration
5. The system must use TensorBoard for training visualization
6. The system must include proper error handling for file operations and model loading
7. The system must support configuration loading from JSON files
8. The system must implement proper dataset splitting (train/validation/test)
9. The model architecture should include batch normalization and ReLU activations
10. The loss function should balance hiding imperceptibility, recovery accuracy, and adversarial loss using appropriate weights
11. The system must support gradient clipping for training stability
12. The system must use Adam optimizer with configurable learning rates and weight decay
13. Attention mechanisms should be implemented to improve feature focusing
14. Skip connections should be added between embedding and extraction networks
15. Specialized architectures for better detail preservation should be explored

## 8. Success Metrics

1. Image quality: PSNR > 40 dB for stego-cover comparison
2. Image quality: SSIM > 0.95 for stego-cover comparison
3. Recovery quality: PSNR > 35 dB for recovered-original secret comparison
4. Recovery quality: SSIM > 0.92 for recovered-original secret comparison
5. Training stability: No NaN or infinite loss values during training
6. Processing efficiency: GPU utilization when available
7. Model performance: Consistent results across multiple test samples
8. User satisfaction: Intuitive GUI operation

## 9. Root Cause Analysis of Low Recovery PSNR

After analyzing the current implementation, the main issues causing low recovery PSNR are:

1. **Weak Extraction Network**: The current extraction network lacks sufficient capacity and specialized components to recover fine details
2. **Imbalanced Loss Functions**: The training focuses heavily on hiding performance while neglecting recovery quality
3. **Missing Attention Mechanisms**: Without attention, the extraction network cannot focus on the most important features
4. **Lack of Skip Connections**: Important details are lost during the embedding-extraction cycle
5. **Inadequate Training Signals**: The extraction network doesn't receive strong enough gradients for effective learning

## 10. Proposed Solutions

1. **Enhanced Extraction Network Architecture**:
   - Add more layers and channels to increase model capacity
   - Implement attention modules to focus on relevant features
   - Add skip connections to preserve fine details
   
2. **Balanced Loss Function**:
   - Increase weight on recovery loss components
   - Add perceptual loss to improve visual quality
   - Implement multi-scale loss for better detail preservation
   
3. **Specialized Training Techniques**:
   - Two-stage training (first optimize hiding, then jointly optimize both)
   - Curriculum learning (start with simpler images, gradually increase complexity)
   - Data augmentation specifically for improving recovery

4. **Attention Mechanisms**:
   - Self-attention modules to capture long-range dependencies
   - Channel attention to emphasize important features
   - Spatial attention to focus on relevant regions

5. **Improved Network Connectivity**:
   - Add skip connections between embedding and extraction networks
   - Implement U-Net style architectures for better feature preservation
   - Use residual connections to ease training