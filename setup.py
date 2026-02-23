"""
Setup script for the Quant-FINN package.
Physics-Informed Neural Networks for Quantitative Finance.
"""

from setuptools import setup, find_packages

setup(
    name="quant-finn",
    version="0.1.0",
    description="Physics-Informed Neural Networks for Quantitative Finance",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
        ],
        "sentiment": [
            "transformers>=4.30.0",
        ],
    },
)
