#!/usr/bin/env python
"""
Setup script for ML Sandbox.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="rediff-ml-sandbox",
    version="0.1.0",
    author="Ibraheem Amin",
    author_email="ibraheem@princeton.edu",
    description="A comprehensive sandbox for graph and vector machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DIodide/rediff-ml-sandbox",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
            "mypy>=1.4.0",
        ],
        "gpu": [
            "torch-geometric>=2.3.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
