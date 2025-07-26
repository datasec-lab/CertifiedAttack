# Contributing to CertifiedAttack

We welcome contributions to CertifiedAttack! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/CertifiedAttack.git
   cd CertifiedAttack
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/originaluser/CertifiedAttack.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

We appreciate many types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new attack methods, defenses, or models
- **Documentation**: Improve or add documentation
- **Tests**: Add test cases to improve coverage
- **Examples**: Create new example scripts or notebooks
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure and readability

### Contribution Workflow

1. **Check existing issues** and pull requests to avoid duplicates
2. **Open an issue** to discuss significant changes before starting
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request** with a clear description

## Development Setup

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

# Install package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Pre-commit Hooks

We use pre-commit hooks to maintain code quality:

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- Line length: 100 characters (instead of 79)
- Use type hints where appropriate
- Use descriptive variable names

### Code Formatting

We use `black` for automatic code formatting:

```bash
# Format a file
black path/to/file.py

# Format all files
black .
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief description of function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: If param1 is negative
    """
    pass
```

### Import Order

1. Standard library imports
2. Third-party imports
3. Local imports

Each group separated by a blank line.

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_attacks.py

# Run with coverage
pytest --cov=attacks --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Include both positive and negative test cases

Example test:

```python
import pytest
from attacks import CertifiedAttack

def test_certified_attack_initialization():
    """Test CertifiedAttack initialization."""
    attack = CertifiedAttack(epsilon=0.1)
    assert attack.epsilon == 0.1
    
def test_certified_attack_invalid_epsilon():
    """Test CertifiedAttack with invalid epsilon."""
    with pytest.raises(ValueError):
        CertifiedAttack(epsilon=-0.1)
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include type hints
- Add inline comments for complex logic

### User Documentation

- Update relevant `.md` files for user-facing changes
- Add examples for new features
- Keep the README.md up to date

### API Documentation

We use Sphinx for API documentation:

```bash
# Build documentation
cd docs
make html
```

## Submitting Changes

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Rebase your changes**:
   ```bash
   git checkout feature/your-feature
   git rebase main
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature
   ```

4. **Create Pull Request** on GitHub

### Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: 
  - What changes were made
  - Why the changes are needed
  - How the changes were tested
- **Link issues**: Reference related issues
- **Small PRs**: Keep pull requests focused

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] PR description is complete

## Reporting Issues

### Bug Reports

Include:
- Python version
- PyTorch version
- Operating system
- Complete error message
- Minimal code to reproduce

### Feature Requests

Include:
- Use case description
- Expected behavior
- Why it would be useful
- Possible implementation approach

### Issue Templates

Use our issue templates on GitHub for consistency.

## Recognition

Contributors will be recognized in:
- The contributors list
- Release notes
- Project documentation

Thank you for contributing to CertifiedAttack!