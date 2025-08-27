#!/usr/bin/env python3
"""
Test script to check model output probabilities and threshold issues
"""

import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from models.config import ModelConfig
from models.inferencer import load_class_configuration

def test_model_probabilities():
    """Test model output probabilities to diagnose threshold issues"""
    print("=== Testing Model Output Probabilities ===\n")
    
    # Load class configuration
    class_names = load_class_configuration()
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    config = ModelConfig(num_classes=num_classes, class_names=class_names)
    model = smp.Unet("resnet101", classes=num_classes, activation=None)
    
    # Load trained weights
    try:
        state_dict = torch.load("models/checkpoints/final_model.pth", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create a test image (random noise for testing)
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print(f"Test image shape: {test_image.shape}")
    
    # Preprocess image
    test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image_tensor = torch.from_numpy(test_image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Run inference
    with torch.no_grad():
        logits = model(test_image_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        print(f"\nRaw logits shape: {logits.shape}")
        print(f"Probabilities shape: {probs.shape}")
        
        # Analyze probabilities for each class
        print("\n=== Probability Analysis ===")
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            class_probs = probs[i]
            
            min_prob = class_probs.min()
            max_prob = class_probs.max()
            mean_prob = class_probs.mean()
            std_prob = class_probs.std()
            
            print(f"\n{class_name}:")
            print(f"  Min probability: {min_prob:.6f}")
            print(f"  Max probability: {max_prob:.6f}")
            print(f"  Mean probability: {mean_prob:.6f}")
            print(f"  Std probability: {std_prob:.6f}")
            
            # Test different thresholds
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            print(f"  Pixels above threshold:")
            for threshold in thresholds:
                pixels_above = (class_probs > threshold).sum()
                percentage = (pixels_above / class_probs.size) * 100
                print(f"    {threshold}: {pixels_above} pixels ({percentage:.2f}%)")
    
    # Test with actual threshold
    threshold = 0.3
    preds = (probs > threshold).astype(np.uint8)
    
    print(f"\n=== Threshold Analysis (threshold={threshold}) ===")
    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class {i}"
        class_pred = preds[i]
        positive_pixels = class_pred.sum()
        percentage = (positive_pixels / class_pred.size) * 100
        print(f"{class_name}: {positive_pixels} positive pixels ({percentage:.2f}%)")
    
    # Check if model is producing any positive predictions
    total_positive = preds.sum()
    if total_positive == 0:
        print("\n⚠️  WARNING: Model is not producing any positive predictions!")
        print("This could indicate:")
        print("1. Threshold is too high")
        print("2. Model needs more training")
        print("3. Model is not learning properly")
        print("4. Data preprocessing issues")
    else:
        print(f"\n✅ Model is producing {total_positive} positive predictions")

if __name__ == "__main__":
    test_model_probabilities() 