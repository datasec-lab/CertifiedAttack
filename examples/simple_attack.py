#!/usr/bin/env python
"""
Simple attack example with minimal configuration.

This script shows how to run a basic attack programmatically.
"""

import os
import sys
import torch
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_image_classification import (
    create_model,
    create_dataloader,
    get_default_config,
    update_config
)
from attacks import get_attack


def simple_attack_example(model_name='resnet', dataset_name='cifar10', attack_name='CertifiedAttack'):
    """
    Run a simple attack example.
    
    Args:
        model_name: Name of the model architecture
        dataset_name: Name of the dataset
        attack_name: Name of the attack method
    """
    print(f"Running {attack_name} on {model_name} trained on {dataset_name}")
    print("-" * 50)
    
    # Create a basic config
    config = get_default_config()
    
    # Dataset configuration
    config.dataset.name = dataset_name.upper()
    config.dataset.data_dir = './data'
    config.dataset.download = True
    
    # Model configuration
    config.model.name = model_name
    if dataset_name == 'cifar10':
        config.model.n_classes = 10
    elif dataset_name == 'cifar100':
        config.model.n_classes = 100
    
    # Find checkpoint
    checkpoint_dir = f'./experiments/{dataset_name}/{model_name}/exp00'
    checkpoint_path = None
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.startswith('checkpoint_') and file.endswith('.pth'):
                checkpoint_path = os.path.join(checkpoint_dir, file)
                break
    
    if checkpoint_path:
        config.model.checkpoint = checkpoint_path
        print(f"Using checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found in {checkpoint_dir}")
        print("The attack will run but may not produce meaningful results without a trained model.")
    
    # Attack configuration
    config.attack.name = attack_name
    config.attack.epsilon = 0.03  # L-inf epsilon for unrestricted attacks
    config.attack.num_iterations = 100
    config.attack.batch_size = 1
    
    # Basic settings
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.attack.targeted = False
    
    # Update config
    update_config(config)
    config.freeze()
    
    # Create model
    print(f"\nCreating {model_name} model...")
    model = create_model(config)
    model = model.to(config.device)
    model.eval()
    
    # Create data loader
    print(f"Loading {dataset_name} dataset...")
    _, _, test_loader = create_dataloader(config)
    
    # Create attack
    print(f"Initializing {attack_name}...")
    attack = get_attack(config, model)
    
    # Run attack on a few samples
    print("\nRunning attack on test samples...")
    success_count = 0
    total_count = 0
    
    for i, (images, labels) in enumerate(test_loader):
        if i >= 10:  # Only test on 10 samples for demo
            break
            
        images = images.to(config.device)
        labels = labels.to(config.device)
        
        # Get original prediction
        with torch.no_grad():
            outputs = model(images)
            orig_pred = outputs.argmax(dim=1)
        
        # Run attack
        adv_images = attack(images, labels)
        
        # Get adversarial prediction
        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_pred = adv_outputs.argmax(dim=1)
        
        # Check success
        if config.attack.targeted:
            success = (adv_pred == labels).item()
        else:
            success = (adv_pred != orig_pred).item()
        
        success_count += success
        total_count += 1
        
        # Calculate perturbation size
        perturbation = (adv_images - images).abs()
        l_inf = perturbation.max().item()
        l_2 = perturbation.view(perturbation.size(0), -1).norm(2, dim=1).item()
        
        print(f"Sample {i+1}: Original: {orig_pred.item()}, "
              f"Adversarial: {adv_pred.item()}, "
              f"Success: {success}, "
              f"L∞: {l_inf:.4f}, L2: {l_2:.4f}")
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Attack Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    print("=" * 50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Simple CertifiedAttack Example')
    parser.add_argument('--model', type=str, default='resnet', 
                        choices=['resnet', 'vgg', 'resnext', 'wrn'],
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset name')
    parser.add_argument('--attack', type=str, default='CertifiedAttack',
                        help='Attack method')
    
    args = parser.parse_args()
    
    try:
        simple_attack_example(
            model_name=args.model,
            dataset_name=args.dataset,
            attack_name=args.attack
        )
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Installed all dependencies: pip install -r requirements.txt")
        print("2. Trained models in the experiments/ directory")
        print("3. CUDA available if using GPU")


if __name__ == '__main__':
    main()