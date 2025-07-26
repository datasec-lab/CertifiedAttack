#!/usr/bin/env python
"""Setup script for CertifiedAttack package."""

import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    # Remove comments and empty lines
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name='certifiedattack',
    version='1.0.0',
    author='Your Name',
    author_email='your-email@example.com',
    description='Certifiable Black-Box Attacks with Randomized Adversarial Examples',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/CertifiedAttack',
    project_urls={
        'Bug Tracker': 'https://github.com/yourusername/CertifiedAttack/issues',
        'Documentation': 'https://github.com/yourusername/CertifiedAttack/wiki',
        'Source Code': 'https://github.com/yourusername/CertifiedAttack',
    },
    packages=find_packages(exclude=['tests', 'experiments', 'configs', 'docs']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'pre-commit>=2.17.0',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.18.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'certifiedattack=attack:main',
            'certifiedattack-train=train:main',
            'certifiedattack-evaluate=evaluate:main',
        ],
    },
    include_package_data=True,
    package_data={
        'certifiedattack': ['configs/**/*.yaml'],
    },
    zip_safe=False,
)