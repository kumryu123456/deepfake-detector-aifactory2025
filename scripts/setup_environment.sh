#!/bin/bash
# Environment Setup Script for Deepfake Detection Project
# This script installs all required dependencies for the project

set -e  # Exit on error

echo "================================================================================"
echo "DEEPFAKE DETECTION - ENVIRONMENT SETUP"
echo "================================================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

REQUIRED_VERSION="3.8"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${GREEN}✓${NC} Python version is compatible (>= 3.8)"
else
    echo -e "${RED}✗${NC} Python version must be >= 3.8"
    exit 1
fi

echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${YELLOW}Warning:${NC} No virtual environment detected"
    echo "It's recommended to use a virtual environment (conda or venv)"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please create and activate a virtual environment first."
        exit 1
    fi
fi

echo ""

# Check if pip is available
echo "Checking pip availability..."
if command -v pip &> /dev/null; then
    PIP_CMD="pip"
    echo -e "${GREEN}✓${NC} pip found"
elif python3 -m pip --version &> /dev/null; then
    PIP_CMD="python3 -m pip"
    echo -e "${GREEN}✓${NC} pip found (via python3 -m pip)"
else
    echo -e "${RED}✗${NC} pip not found"
    echo ""
    echo "Please install pip first:"
    echo "  sudo apt update"
    echo "  sudo apt install -y python3-pip python3-venv"
    echo ""
    echo "Or on other systems:"
    echo "  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py"
    echo "  python3 get-pip.py"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 1: Installing PyTorch with CUDA 11.8 support"
echo "================================================================================"
echo ""
echo "This will download ~2-3 GB of packages. It may take several minutes."
echo ""

# Install PyTorch with CUDA 11.8
$PIP_CMD install torch==1.13.1+cu118 torchvision==0.14.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} PyTorch installed successfully"
else
    echo -e "${RED}✗${NC} PyTorch installation failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 2: Installing project dependencies"
echo "================================================================================"
echo ""

# Install other dependencies
$PIP_CMD install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Dependencies installed successfully"
else
    echo -e "${RED}✗${NC} Dependency installation failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 3: Verifying installation"
echo "================================================================================"
echo ""

# Verify PyTorch installation
python3 -c "
import torch
import torchvision
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} PyTorch verification passed"
else
    echo -e "${RED}✗${NC} PyTorch verification failed"
    exit 1
fi

echo ""

# Verify key dependencies
python3 -c "
try:
    import timm
    import cv2
    import albumentations
    import pandas
    import sklearn
    import yaml
    import tqdm
    import facenet_pytorch
    import mediapipe
    print('✓ All key dependencies imported successfully')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Dependency verification passed"
else
    echo -e "${RED}✗${NC} Dependency verification failed"
    exit 1
fi

echo ""

# Verify project imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python3 -c "
try:
    from models import create_model_from_config
    from data import DeepfakeDataset
    from training import Trainer
    from inference import InferenceEngine
    print('✓ All project modules imported successfully')
except ImportError as e:
    print(f'✗ Project import failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Project module verification passed"
else
    echo -e "${YELLOW}⚠${NC}  Project module verification failed (this is expected if modules aren't fully implemented yet)"
fi

echo ""
echo "================================================================================"
echo "INSTALLATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Create a demo checkpoint for testing:"
echo "   python scripts/create_demo_checkpoint.py --config configs/baseline_config.yaml --output checkpoints/demo.pth"
echo ""
echo "2. Test inference pipeline:"
echo "   python scripts/inference.py --checkpoint checkpoints/demo.pth --data ./data --output submission.csv"
echo ""
echo "3. Test submission notebook:"
echo "   jupyter notebook task.ipynb"
echo ""
echo "For more information, see:"
echo "  - SETUP_GUIDE.md - Complete setup instructions"
echo "  - README.md - Project overview"
echo "  - README_NOTEBOOK.md - Notebook usage guide"
echo ""
echo "================================================================================"
