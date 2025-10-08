# PRD: Steganography Color Recovery Enhancement

## Introduction/Overview

The current steganography model suffers from color recovery issues where recovered secret images appear with a "coffee color" or sepia-tone wash-out instead of full colors. This is caused by two main problems: 1) the use of MSE (Mean Squared Error) loss function which doesn't understand human perception, and 2) the residual model architecture where the embedding network learns to generate a difference map rather than a complete stego image. This PRD outlines improvements to achieve full-color recovery using perceptual loss (LPIPS) and direct image generation architecture.

## Goals

1. Implement perceptual loss (LPIPS) in the training loss function to improve color recovery quality
2. Modify the architecture to use direct image generation instead of residual difference addition
3. Achieve full-color recovered secret images that closely match the original secret
4. Maintain imperceptibility of the stego images compared to the cover images
5. Preserve the ability to extract the secret from the stego image alone without requiring the cover image

## User Stories

1. As a user of the steganography system, I want the recovered secret image to maintain full-color fidelity so that I can accurately retrieve the original image information.

2. As a developer, I want the model to use perceptual loss functions that better align with human vision so that the color recovery is more accurate.

3. As a researcher, I want the steganography model to employ direct image generation architecture to avoid the limitations of residual learning that causes color degradation.

## Functional Requirements

1. The model must integrate LPIPS (Learned Perceptual Image Patch Similarity) loss into the training process for the recovery phase.
2. The embed method should generate the complete stego image directly without using residual addition.
3. The loss function must include both MSE and LPIPS components for recovery loss.
4. The model must maintain the existing capability to extract secrets from stego images without requiring the original cover.
5. The recovery PSNR and SSIM metrics should improve while maintaining hiding quality metrics.
6. The system must be configurable to adjust the weights of different loss components (hiding, recovery, adversarial, LPIPS).

## Non-Goals (Out of Scope)

1. Changing the input/output formats of the steganography process
2. Modifying the basic GAN training process beyond loss function changes
3. Implementing different model architectures beyond the embed method fix
4. Adding new image formats or data types to the pipeline

## Design Considerations

1. The LPIPS model should be initialized with the AlexNet backbone for perceptual quality assessment
2. The embed method will be modified to return the direct output of the embedding network without adding residuals
3. Loss function weights must be configurable via the config.json file
4. The solution should maintain backward compatibility with existing training and inference workflows

## Technical Considerations

1. The LPIPS loss function will need to be initialized on the device and passed to the loss calculation function
2. The model will need to be retrained from scratch due to the architectural changes
3. The changes should be implemented in a new version to preserve existing work
4. The config.json will require new parameters for LPIPS loss weight

## Success Metrics

1. Color quality of recovered images should be visibly improved (less coffee/sepia tone)
2. Recovery LPIPS scores should decrease (better perceptual similarity)
3. Recovery PSNR and SSIM should improve compared to the original implementation
4. Hiding quality (PSNR, SSIM, LPIPS) should remain acceptable
5. The model should successfully recover full-color secret images during testing

## Open Questions

1. How should the LPIPS loss be weighted relative to MSE loss in the recovery phase?
2. What training duration is appropriate for the new architecture with perceptual loss?
3. Should the discriminator also be modified to account for better perceptual quality?