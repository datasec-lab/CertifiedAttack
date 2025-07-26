# Usage Guide

This guide provides comprehensive examples and tutorials for using CertifiedAttack.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Attack Examples](#attack-examples)
  - [Certified Attack](#certified-attack)
  - [Black-box Attacks](#black-box-attacks)
  - [Attacks with Defenses](#attacks-with-defenses)
- [Training Models](#training-models)
  - [Standard Training](#standard-training)
  - [Adversarial Training](#adversarial-training)
- [Evaluation](#evaluation)
- [Configuration System](#configuration-system)
- [Advanced Usage](#advanced-usage)
- [Batch Processing](#batch-processing)
- [Custom Experiments](#custom-experiments)

## Basic Usage

### Command Line Interface

All main scripts support command-line arguments:

```bash
# General format
python script.py --config CONFIG_FILE [options]

# Override config options
python attack.py --config config.yaml device cuda:1 attack.epsilon 0.05
```

### Quick Start Examples

```bash
# 1. Run Certified Attack on CIFAR-10
python attack.py --config "./configs/attack/cifar10/untargeted/unrestricted/vgg_CertifiedAttack.yaml"

# 2. Train a model
python train.py --config "./configs/cifar10/resnet.yaml"

# 3. Evaluate a model
python evaluate.py --config "./configs/evaluate/vgg.yaml"
```

## Attack Examples

### Certified Attack

Our proposed certified attack method with different variants:

#### Binary Search Variant
```bash
# CIFAR-10 with VGG
python attack.py --config "./configs/attack/cifar10/untargeted/unrestricted/vgg_CertifiedAttack.yaml" \
    device cuda:0 \
    attack.num_samples 1000 \
    attack.confidence_level 0.95

# CIFAR-100 with ResNet
python attack.py --config "./configs/attack/cifar100/untargeted/unrestricted/resnet_CertifiedAttack.yaml" \
    device cuda:0 \
    attack.binary_search_steps 15
```

#### SSSP (Single-Step Single-Pixel) Variant
```bash
# SSSP variant on ImageNet
python attack.py --config "./configs/attack/imagenet_RAND/untargeted/unrestricted/resnet_CertifiedAttack_sssp.yaml" \
    device cuda:0 \
    attack.sssp_iterations 50
```

### Black-box Attacks

#### Score-based Attacks
```bash
# NES Attack
python attack.py --config "./configs/attack/cifar10/untargeted/l2/resnet_NES.yaml"

# Square Attack
python attack.py --config "./configs/attack/cifar10/untargeted/linf/resnet_Square.yaml"

# SimBA Attack
python attack.py --config "./configs/attack/cifar10/untargeted/l2/resnet_SimBA.yaml"
```

#### Decision-based Attacks
```bash
# Boundary Attack
python attack.py --config "./configs/attack/cifar10/untargeted/l2/resnet_Boundary.yaml"

# HSJA Attack
python attack.py --config "./configs/attack/cifar10/untargeted/linf/resnet_HSJA.yaml"

# GeoDA Attack
python attack.py --config "./configs/attack/cifar10/untargeted/l2/resnet_GeoDA.yaml"
```

#### Sparse Attacks
```bash
# PointWise Attack
python attack.py --config "./configs/attack/cifar10/untargeted/unrestricted/resnet_PointWise.yaml"

# SparseEvo Attack
python attack.py --config "./configs/attack/cifar10/untargeted/unrestricted/resnet_SparseEvo.yaml"
```

### Attacks with Defenses

#### Against Blacklight Defense
```bash
# CIFAR-10
python attack.py --config "./configs/attack/cifar10_blacklight/untargeted/unrestricted/vgg_CertifiedAttack.yaml"

# CIFAR-100
python attack.py --config "./configs/attack/cifar100_blacklight/untargeted/unrestricted/resnet_CertifiedAttack.yaml"
```

#### Against Randomized Defenses
```bash
# Pre-processing defense (RAND)
python attack.py --config "./configs/attack/cifar10_RAND/untargeted/unrestricted/resnet_CertifiedAttack.yaml"

# Post-processing defense
python attack.py --config "./configs/attack/cifar10_post_RAND/untargeted/unrestricted/wrn_CertifiedAttack.yaml"
```

#### Against Adversarial Training
```bash
# L2 adversarial training
python attack.py --config "./configs/attack/cifar10_AT/untargeted/unrestricted/resnet_CertifiedAttack_l2.yaml"

# L-infinity adversarial training
python attack.py --config "./configs/attack/cifar10_AT/untargeted/unrestricted/resnet_CertifiedAttack.yaml"
```

## Training Models

### Standard Training

#### Basic Training
```bash
# Train ResNet on CIFAR-10
python train.py --config "./configs/cifar10/resnet.yaml" \
    train.output_dir "./experiments/my_resnet" \
    train.epochs 200 \
    train.batch_size 128
```

#### Different Architectures
```bash
# VGG
python train.py --config "./configs/cifar10/vgg.yaml"

# Wide ResNet
python train.py --config "./configs/cifar10/wrn.yaml"

# ResNeXt
python train.py --config "./configs/cifar10/resnext.yaml"

# DenseNet
python train.py --config "./configs/cifar10/densenet.yaml"
```

#### With Data Augmentation
```bash
# CutMix
python train.py --config "./configs/cifar10/resnet.yaml" \
    augmentation.use_cutmix True \
    augmentation.cutmix_alpha 1.0

# MixUp
python train.py --config "./configs/cifar10/resnet.yaml" \
    augmentation.use_mixup True \
    augmentation.mixup_alpha 0.2
```

### Adversarial Training

#### TRADES Training
```bash
# L-infinity adversarial training
python train.py --config "./configs/AT/cifar10/resnet_linf.yaml" \
    train.adv_epsilon 8/255 \
    train.adv_step_size 2/255 \
    train.adv_steps 10

# L2 adversarial training
python train.py --config "./configs/AT/cifar10/resnet_l2.yaml" \
    train.adv_epsilon 0.5 \
    train.adv_beta 6.0
```

#### Custom Adversarial Training
```bash
# Custom epsilon schedule
python train.py --config "./configs/AT/cifar100/resnet_linf.yaml" \
    train.adv_epsilon_schedule "linear" \
    train.adv_epsilon_start 0 \
    train.adv_epsilon_end 8/255
```

## Evaluation

### Model Evaluation
```bash
# Basic evaluation
python evaluate.py --config "./configs/evaluate/vgg.yaml" \
    model.checkpoint "./experiments/cifar10/vgg/exp00/checkpoint_00160.pth"

# Evaluation with different metrics
python evaluate.py --config "./configs/evaluate/vgg.yaml" \
    evaluate.metrics ["accuracy", "robustness", "confidence"]
```

### Robustness Evaluation
```bash
# Evaluate against multiple attacks
python evaluate_robustness.py \
    --model-config "./configs/cifar10/resnet.yaml" \
    --attack-configs "./configs/attack/cifar10/untargeted/*/*.yaml" \
    --output-dir "./results/robustness_evaluation"
```

## Configuration System

### Understanding Config Files

Configuration files use YAML format with hierarchical structure:

```yaml
# Example config structure
dataset:
  name: CIFAR10
  data_dir: ./data
  
model:
  name: resnet
  depth: 18
  
attack:
  name: CertifiedAttack
  epsilon: 0.03
  num_iterations: 100
  
device: cuda:0
```

### Config Override System

Override any config parameter from command line:

```bash
# Override single parameter
python attack.py --config base.yaml attack.epsilon 0.05

# Override multiple parameters
python attack.py --config base.yaml \
    attack.epsilon 0.05 \
    attack.num_iterations 200 \
    device cuda:1
```

### Creating Custom Configs

1. **Extend existing config**:
```yaml
# my_config.yaml
_base_: ./configs/attack/cifar10/untargeted/unrestricted/vgg_CertifiedAttack.yaml

# Override specific settings
attack:
  num_samples: 2000
  confidence_level: 0.99
  
experiment:
  name: "high_confidence_attack"
```

2. **Use the custom config**:
```bash
python attack.py --config my_config.yaml
```

## Advanced Usage

### Multi-GPU Training
```bash
# Data parallel training
python train.py --config "./configs/cifar10/resnet.yaml" \
    train.distributed True \
    train.world_size 4
```

### Mixed Precision Training
```bash
# Enable automatic mixed precision
python train.py --config "./configs/cifar10/resnet.yaml" \
    train.use_amp True \
    train.amp_opt_level "O1"
```

### Custom Learning Rate Schedules
```bash
# Cosine annealing
python train.py --config "./configs/cifar10/resnet.yaml" \
    scheduler.name "cosine" \
    scheduler.t_max 200

# Multi-step LR
python train.py --config "./configs/cifar10/resnet.yaml" \
    scheduler.name "multistep" \
    scheduler.milestones [100, 150] \
    scheduler.gamma 0.1
```

### Checkpoint Management
```bash
# Resume from checkpoint
python train.py --config "./configs/cifar10/resnet.yaml" \
    train.resume "./experiments/cifar10/resnet/exp00/checkpoint_00100.pth"

# Save checkpoints at specific intervals
python train.py --config "./configs/cifar10/resnet.yaml" \
    train.checkpoint_interval 10
```

## Batch Processing

### Running Multiple Attacks
```bash
# Use provided scripts
bash run_attacks_cifar10.sh

# Or create custom batch script
for model in vgg resnet resnext wrn; do
    for attack in CertifiedAttack PointWise SparseEvo; do
        python attack.py --config "./configs/attack/cifar10/untargeted/unrestricted/${model}_${attack}.yaml"
    done
done
```

### Parallel Execution
```bash
# Run attacks in parallel
parallel -j 4 python attack.py --config {} ::: configs/attack/cifar10/untargeted/*/*.yaml
```

## Custom Experiments

### Creating New Attack Configurations

1. **Create config file**:
```yaml
# configs/attack/custom/my_attack.yaml
dataset:
  name: CIFAR10
  
model:
  name: resnet
  checkpoint: ./experiments/cifar10/resnet/checkpoint.pth
  
attack:
  name: CertifiedAttack
  epsilon: 0.05
  num_samples: 1500
  binary_search_steps: 20
  confidence_level: 0.95
  
experiment:
  output_dir: ./results/custom_attack
  save_adversarial_examples: True
```

2. **Run the experiment**:
```bash
python attack.py --config ./configs/attack/custom/my_attack.yaml
```

### Analyzing Results

```python
# Load and analyze results
import numpy as np
import matplotlib.pyplot as plt

# Load attack results
results = np.load('results/attack_results.npz')
success_rates = results['success_rates']
query_counts = results['query_counts']

# Plot success rate vs queries
plt.figure(figsize=(10, 6))
plt.plot(query_counts, success_rates)
plt.xlabel('Number of Queries')
plt.ylabel('Attack Success Rate')
plt.title('Attack Performance')
plt.grid(True)
plt.savefig('attack_performance.png')
```

### Custom Metrics and Logging

```python
# Add custom metrics to evaluation
python evaluate.py --config "./configs/evaluate/vgg.yaml" \
    evaluate.custom_metrics ["perturbation_size", "query_efficiency"] \
    evaluate.log_interval 10 \
    evaluate.save_predictions True
```

## Tips and Best Practices

1. **Start with small experiments**: Test with reduced dataset size or iterations
   ```bash
   python attack.py --config config.yaml \
       dataset.subset_size 100 \
       attack.num_iterations 10
   ```

2. **Monitor GPU memory**: Adjust batch size if OOM
   ```bash
   python attack.py --config config.yaml \
       attack.batch_size 16 \
       train.gradient_accumulation_steps 4
   ```

3. **Use tensorboard for monitoring**:
   ```bash
   # Start tensorboard
   tensorboard --logdir ./experiments/
   ```

4. **Save intermediate results**:
   ```bash
   python attack.py --config config.yaml \
       experiment.save_interval 100 \
       experiment.save_intermediate True
   ```

5. **Debug mode**:
   ```bash
   python attack.py --config config.yaml \
       debug True \
       dataset.subset_size 10
   ```

## Getting Help

- Check configuration defaults in `pytorch_image_classification/config/defaults.py`
- See example configs in `configs/` directory
- Review source code documentation
- Open an issue on GitHub for specific problems