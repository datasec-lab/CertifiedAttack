# Installation Guide

This guide provides detailed instructions for installing CertifiedAttack on various platforms.

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Installation](#quick-installation)
- [Detailed Installation](#detailed-installation)
  - [Using pip](#using-pip)
  - [Using conda](#using-conda)
  - [From source](#from-source)
- [GPU Support](#gpu-support)
- [Platform-Specific Instructions](#platform-specific-instructions)
  - [Linux](#linux)
  - [macOS](#macos)
  - [Windows](#windows)
- [Docker Installation](#docker-installation)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8 GB RAM
- 10 GB free disk space

### Recommended Requirements
- Python 3.9 or 3.10
- 16 GB RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.0 or higher
- 20 GB free disk space

### Supported Operating Systems
- Ubuntu 18.04/20.04/22.04
- CentOS 7/8
- macOS 10.15+
- Windows 10/11

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CertifiedAttack.git
cd CertifiedAttack

# Install dependencies
pip install -r requirements.txt
```

## Detailed Installation

### Using pip

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv certifiedattack_env
   
   # Activate the environment
   # On Linux/macOS:
   source certifiedattack_env/bin/activate
   # On Windows:
   certifiedattack_env\Scripts\activate
   ```

2. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

3. **Install PyTorch** (with CUDA support):
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install CertifiedAttack dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Using conda

1. **Install Miniconda or Anaconda** if not already installed:
   - Download from: https://docs.conda.io/en/latest/miniconda.html

2. **Create conda environment**:
   ```bash
   conda create -n certifiedattack python=3.9
   conda activate certifiedattack
   ```

3. **Install PyTorch with CUDA**:
   ```bash
   # For CUDA 11.8
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CUDA 12.1
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. **Install remaining dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### From source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/CertifiedAttack.git
   cd CertifiedAttack
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

## GPU Support

### NVIDIA GPU Setup

1. **Check CUDA version**:
   ```bash
   nvidia-smi
   ```

2. **Install appropriate PyTorch version**:
   - Visit: https://pytorch.org/get-started/locally/
   - Select your configuration
   - Use the provided installation command

3. **Verify GPU availability**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   print(torch.cuda.device_count())   # Number of GPUs
   ```

### AMD GPU Support (ROCm)

```bash
# Install PyTorch for ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## Platform-Specific Instructions

### Linux

#### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip git

# Install CUDA (if using GPU)
# Follow NVIDIA's official guide: https://developer.nvidia.com/cuda-downloads
```

#### CentOS/RHEL
```bash
# Install system dependencies
sudo yum install -y python3-devel python3-pip git

# Install CUDA (if using GPU)
# Follow NVIDIA's official guide
```

### macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install dependencies
pip3 install -r requirements.txt
```

**Note**: macOS doesn't support CUDA. Use CPU-only PyTorch or consider using cloud services.

### Windows

1. **Install Python**:
   - Download from: https://www.python.org/downloads/
   - Check "Add Python to PATH" during installation

2. **Install Visual Studio Build Tools**:
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Select "C++ build tools" workload

3. **Install Git**:
   - Download from: https://git-scm.com/download/win

4. **Open Command Prompt or PowerShell as Administrator**:
   ```powershell
   # Clone repository
   git clone https://github.com/yourusername/CertifiedAttack.git
   cd CertifiedAttack
   
   # Create virtual environment
   python -m venv certifiedattack_env
   certifiedattack_env\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## Docker Installation

1. **Build Docker image**:
   ```bash
   docker build -t certifiedattack:latest .
   ```

2. **Run container**:
   ```bash
   # With GPU support
   docker run --gpus all -it -v $(pwd):/workspace certifiedattack:latest
   
   # CPU only
   docker run -it -v $(pwd):/workspace certifiedattack:latest
   ```

## Troubleshooting

### Common Issues

1. **CUDA/cuDNN version mismatch**:
   ```bash
   # Check versions
   python -c "import torch; print(torch.version.cuda)"
   
   # Reinstall matching PyTorch version
   pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. **Out of memory errors**:
   - Reduce batch size in config files
   - Use gradient accumulation
   - Enable mixed precision training

3. **ImportError for specific packages**:
   ```bash
   # Reinstall the specific package
   pip install --force-reinstall package_name
   ```

4. **Permission denied errors**:
   ```bash
   # Use --user flag
   pip install --user -r requirements.txt
   ```

### Getting Help

If you encounter issues:
1. Check the [FAQ](docs/FAQ.md)
2. Search existing [GitHub Issues](https://github.com/yourusername/CertifiedAttack/issues)
3. Create a new issue with:
   - System information (OS, Python version, CUDA version)
   - Complete error message
   - Steps to reproduce

## Verification

After installation, verify everything is working:

```bash
# Check Python version
python --version

# Test imports
python -c "import torch, torchvision, numpy, matplotlib; print('All imports successful!')"

# Run a simple test
python attack.py --help

# Run unit tests (if available)
pytest tests/
```

### Test Attack

Run a quick test attack to ensure everything is properly installed:

```bash
# Download a small test dataset (if needed)
python scripts/download_test_data.py

# Run test attack
python attack.py --config configs/test/quick_test.yaml
```

If all tests pass, you're ready to use CertifiedAttack!

## Next Steps

- Read the [Usage Guide](USAGE.md) for detailed examples
- Check [examples/](examples/) for sample scripts
- Review [configs/](configs/) for configuration options