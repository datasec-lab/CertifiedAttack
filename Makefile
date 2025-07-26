.PHONY: help install install-dev test format lint clean attack train evaluate

help:
	@echo "CertifiedAttack Makefile Commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make test         - Run tests"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Run linting checks"
	@echo "  make clean        - Clean temporary files"
	@echo "  make attack       - Run quick attack demo"
	@echo "  make train        - Train a model (example)"
	@echo "  make evaluate     - Evaluate a model (example)"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/

format:
	black .
	isort .

lint:
	flake8 .
	mypy .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.swp" -delete
	find . -type f -name "*.swo" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

attack:
	python examples/quick_start.py --demo

train:
	python train.py --config configs/cifar10/resnet.yaml

evaluate:
	python evaluate.py --config configs/evaluate/vgg.yaml

# Development helpers
setup-gpu:
	@echo "Checking GPU availability..."
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

download-data:
	@echo "Downloading CIFAR-10 dataset..."
	@python -c "import torchvision.datasets as datasets; datasets.CIFAR10('./data', download=True)"

quick-test:
	python attack.py --config configs/attack/cifar10/untargeted/unrestricted/vgg_CertifiedAttack.yaml \
		dataset.subset_size 10 \
		attack.num_iterations 10 \
		device cpu