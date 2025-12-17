#!/bin/bash

ENV_NAME="ant_navigation"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Building environment from environment.yml..."
conda env create -f environment.yml || conda env update -f environment.yml

conda activate $ENV_NAME
echo "Environment '$ENV_NAME' activated."
echo "Detecting hardware for PyTorch installation..."

if lspci | grep -i 'vga.*nvidia' > /dev/null; then
    echo "NVIDIA GPU Detected. Installing CUDA PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 #Change CUDA version here
elif lspci | grep -i 'vga.*amd' > /dev/null; then
    echo "AMD GPU Detected. Installing ROCm PyTorch (6.2)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4   #Change ROCm version here
    #export HSA_OVERRIDE_GFX_VERSION=10.3.0 
else
    echo "No GPU detected. Installing CPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
echo "Installing Stable Baselines3..."
pip install stable-baselines3[extra]

echo "Hardware Verification:"
python3 -c "import torch; print(f' - Torch version: {torch.__version__}'); \
print(f' - ROCm/CUDA available: {torch.cuda.is_available()}'); \
print(f' - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo "------------------------------------------------"
echo "Setup Complete! Use 'conda activate $ENV_NAME' to begin."