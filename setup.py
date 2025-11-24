from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="instacart-next-purchase",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready ML system for predicting Instacart customer reorders with API and CLI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/instacart_next_purchase",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.20.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pre-commit>=3.6.0",
        ],
        "cloud": [
            "boto3>=1.34.0",
            "s3fs>=2023.12.2",
            "awscli>=1.32.17",
        ],
        "docs": [
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.15.0",
            "psutil>=5.9.0",
        ],
        "all": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0", 
            "pytest-asyncio>=0.20.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pre-commit>=3.6.0",
            "boto3>=1.34.0",
            "s3fs>=2023.12.2",
            "awscli>=1.32.17",
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
            "prometheus-client>=0.15.0",
            "psutil>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "instacart-cli=scripts.cli:cli",
            "instacart-api=uvicorn:main",
            "instacart-etl=etl.extract:main",
            "instacart-train=models.train_xgb:main",
            "instacart-predict=models.inference:main",
        ],
    },
    package_data={
        "": ["config/*.yaml"],
    },
    include_package_data=True,
    zip_safe=False,
)