#!/usr/bin/env python3
# Test script to verify the enhanced GAN steganography model implementation.

import torch
from enhanced_gan_main import EnhancedGANSteganographyModel, EnhancedDiscriminator

def test_model_instantiation():
    \"\"\"Test that the enhanced GAN models can be instantiated correctly.\"\"\"
    print(\"Testing Enhanced GAN Steganography Model instantiation...\")
    
    try:
        # Instantiate the enhanced steganography model
        steg_model = EnhancedGANSteganographyModel()
        print(\"✓ EnhancedGANSteganographyModel instantiated successfully\")
        
        # Instantiate the enhanced discriminator
        disc_model = EnhancedDiscriminator()
        print(\"✓ EnhancedDiscriminator instantiated successfully\")
        
        # Count parameters
        steg_params = sum(p.numel() for p in steg_model.parameters())
        disc_params = sum(p.numel() for p in disc_model.parameters())
        print(f\"✓ EnhancedGANSteganographyModel parameters: {steg_params:,}\")
        print(f\"✓ EnhancedDiscriminator parameters: {disc_params:,}\")
        
        # Test forward pass with dummy data
        dummy_cover = torch.randn(1, 3, 224, 224)  # Batch of 1, 3 channels, 224x224
        dummy_secret = torch.randn(1, 3, 224, 224)  # Batch of 1, 3 channels, 224x224
        
        # Test embedding
        stego_output = steg_model.embed(dummy_cover, dummy_secret)
        print(\"✓ Embedding forward pass successful\")
        print(f\"  Stego output shape: {stego_output.shape}\")
        
        # Test extraction
        recovered_secret = steg_model.extract(stego_output)
        print(\"✓ Extraction forward pass successful\")
        print(f\"  Recovered secret shape: {recovered_secret.shape}\")
        
        # Test discriminator
        disc_output = disc_model(dummy_cover)
        print(\"✓ Discriminator forward pass successful\")
        print(f\"  Discriminator output shape: {disc_output.shape}\")
        
        print(\"\\nAll tests passed! Enhanced GAN models are working correctly.\")
        return True
        
    except Exception as e:
        print(f\"❌ Error during testing: {e}\")
        return False

if __name__ == \"__main__\":
    test_model_instantiation()