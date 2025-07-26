# CertifiedAttack Examples

This directory contains example scripts to help you get started with CertifiedAttack.

## Quick Start Examples

### 1. Interactive Demo (`quick_start.py`)

An interactive script that guides you through running different attack scenarios:

```bash
python examples/quick_start.py --demo
```

This will present you with a menu of pre-configured attack examples to choose from.

### 2. Simple Attack (`simple_attack.py`)

A minimal example showing how to run attacks programmatically:

```bash
# Basic usage
python examples/simple_attack.py

# Specify model and dataset
python examples/simple_attack.py --model vgg --dataset cifar100

# Different attack
python examples/simple_attack.py --attack PointWise
```

### 3. Batch Attack Comparison (`compare_attacks.py`)

Compare multiple attacks on the same model:

```bash
python examples/compare_attacks.py --model resnet --dataset cifar10
```

### 4. Defense Evaluation (`evaluate_defenses.py`)

Evaluate attacks against different defense mechanisms:

```bash
python examples/evaluate_defenses.py --defense blacklight
```

### 5. Custom Configuration (`custom_config_example.py`)

Example of creating and using custom configurations:

```bash
python examples/custom_config_example.py
```

## Jupyter Notebooks

For interactive exploration, check out our Jupyter notebooks:

- `notebooks/01_getting_started.ipynb` - Introduction to CertifiedAttack
- `notebooks/02_attack_visualization.ipynb` - Visualizing attack results
- `notebooks/03_defense_comparison.ipynb` - Comparing defense mechanisms
- `notebooks/04_custom_attacks.ipynb` - Implementing custom attacks

## Running the Examples

1. **Ensure you have trained models** in the `experiments/` directory:
   ```bash
   python train.py --config configs/cifar10/resnet.yaml
   ```

2. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run from the project root directory**:
   ```bash
   cd /path/to/CertifiedAttack
   python examples/simple_attack.py
   ```

## Common Issues

- **No checkpoint found**: Train a model first using the training scripts
- **CUDA out of memory**: Reduce batch size in the examples
- **Import errors**: Make sure you're running from the project root directory

## Creating Your Own Examples

To create custom examples:

1. Copy one of the existing examples as a template
2. Modify the configuration parameters
3. Add your custom logic
4. Save in the examples directory

Example structure:
```python
import sys
sys.path.append('..')  # Add parent directory to path

from pytorch_image_classification import create_model, create_dataloader
from attacks import get_attack

# Your custom code here
```

## Need Help?

- Check the [Usage Guide](../USAGE.md) for detailed documentation
- Review the [configuration files](../configs/) for more options
- Open an issue on GitHub for specific questions