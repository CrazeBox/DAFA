#!/usr/bin/env python3
"""
DAFA: Directionally Aligned Federated Aggregation Framework

A comprehensive framework for federated learning experiments with
directionally aligned aggregation methods.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

script_requirements = [req for req in requirements if not req.startswith("torch")]

setup(
    name="dafa-fl",
    version="1.0.0",
    author="DAFA Research Team",
    author_email="research@dafa-fl.org",
    description="Directionally Aligned Federated Aggregation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dafa-fl/dafa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=script_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dafa-run=scripts.run_experiment:main",
            "dafa-stages=scripts.run_five_stages:main",
            "dafa-data=scripts.download_datasets:main",
            "dafa-analyze=scripts.analyze_results:main",
        ],
    },
)
