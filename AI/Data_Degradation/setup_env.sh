#!/bin/bash

# Setup Environment for Time-Series Data Augmentation Benchmarking
# Usage: ./setup_env.sh

ENV_NAME="ts_aug_env"
PYTHON_VERSION="3.10"

echo "=========================================================="
echo " Setting up Conda Environment: $ENV_NAME"
echo "=========================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create a new conda environment
echo "Creating conda environment with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
# Note: we need to use conda run or source activate to ensure the script acts within the env
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Environment activated."

# Install dependencies from requirements.txt using pip
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found! Skipping dependency installation."
fi

echo "=========================================================="
echo " Setup Complete!"
echo " To activate this environment in the future, run:"
echo " conda activate $ENV_NAME"
echo "=========================================================="
