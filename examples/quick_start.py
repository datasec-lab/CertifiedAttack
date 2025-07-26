#!/usr/bin/env python
"""
Quick start example for CertifiedAttack.

This script demonstrates a simple attack on a pre-trained model.
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack import load_config, main as attack_main


def quick_start_demo():
    """Run a quick demonstration of CertifiedAttack."""
    print("=" * 50)
    print("CertifiedAttack Quick Start Demo")
    print("=" * 50)
    
    # List of example configurations
    examples = [
        {
            'name': 'Certified Attack on CIFAR-10 VGG',
            'config': './configs/attack/cifar10/untargeted/unrestricted/vgg_CertifiedAttack.yaml',
            'description': 'Our proposed certified attack method on VGG model'
        },
        {
            'name': 'Attack with Blacklight Defense',
            'config': './configs/attack/cifar10_blacklight/untargeted/unrestricted/resnet_CertifiedAttack.yaml',
            'description': 'Attack against Blacklight detection defense'
        },
        {
            'name': 'Attack with RAND Defense',
            'config': './configs/attack/cifar10_RAND/untargeted/unrestricted/resnet_CertifiedAttack.yaml',
            'description': 'Attack against randomized preprocessing defense'
        }
    ]
    
    print("\nAvailable examples:")
    for i, example in enumerate(examples):
        print(f"\n{i+1}. {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Config: {example['config']}")
    
    # Get user choice
    while True:
        try:
            choice = input("\nSelect an example (1-3) or 'q' to quit: ")
            if choice.lower() == 'q':
                print("Exiting...")
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                selected = examples[idx]
                break
            else:
                print("Invalid choice. Please select 1-3.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
    
    print(f"\nRunning: {selected['name']}")
    print("-" * 50)
    
    # Check if config file exists
    config_path = selected['config']
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Please make sure you're running from the project root directory.")
        return
    
    # Run the attack
    try:
        # Simulate command line arguments
        sys.argv = ['attack.py', '--config', config_path, 'device', 'cuda:0']
        
        # You can also override specific parameters
        # sys.argv.extend(['attack.num_iterations', '10'])  # For quick demo
        
        # Load config and run
        config = load_config()
        attack_main(config)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check if CUDA is available for GPU execution")
        print("3. Ensure the model checkpoints exist in the experiments/ directory")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='CertifiedAttack Quick Start')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    args = parser.parse_args()
    
    if args.demo or len(sys.argv) == 1:
        quick_start_demo()
    else:
        print("Run with --demo flag for interactive demonstration")
        print("Or use attack.py directly with a config file")


if __name__ == '__main__':
    main()