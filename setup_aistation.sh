#!/bin/bash
# ============================================================================
# Setup Script for Inspur AIStation V5.0 (Corrected)
# Handles missing python3.8-venv and pip installation issues
# ============================================================================

set -e

echo "=========================================="
echo "CDG Setup: Inspur AIStation V5.0 (Fixed)"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Option 1: Install missing packages (if apt works)
echo "Attempting to install python3.8-venv..."
apt update 2>/dev/null || echo "apt update had issues, continuing..."
apt install -y python3.8-venv python3-pip 2>/dev/null || echo "apt install had issues, trying alternative..."

# Option 2: Use Python's built-in venv module directly (doesn't require ensurepip)
echo "Creating virtual environment using built-in venv..."
python3 -m venv venv --without-pip

# Activate venv
source venv/bin/activate

# Option 3: Install pip manually using get-pip.py
echo "Installing pip manually..."
cd /tmp
curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py 2>/dev/null || {
    # If curl fails, use wget
    wget -q https://bootstrap.pypa.io/get-pip.py -O get-pip.py 2>/dev/null || {
        echo "⚠ Warning: Could not download get-pip.py"
        echo "Attempting to use system pip if available..."
        which pip3 && cp $(which pip3) /tmp/pip_fallback || true
    }
}

# Install pip
if [ -f get-pip.py ]; then
    python3 get-pip.py --no-warn-script-location
    rm get-pip.py
fi

cd -

# Upgrade pip in venv
python3 -m pip install --upgrade pip setuptools wheel 2>&1 | grep -v "already satisfied" || true

# ============================================================================
# Install Core Dependencies (PyTorch for Python 3.8.10)
# ============================================================================

echo ""
echo "Installing PyTorch..."

# For CPU only (faster, lighter)
# pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.7 (A800 GPUs)
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
    --index-url https://download.pytorch.org/whl/cu117

# ============================================================================
# Install Project Dependencies (Python 3.8.10 compatible)
# ============================================================================

echo ""
echo "Installing project dependencies..."

pip install \
    numpy==1.21.6 \
    pyyaml==6.0 \
    tensorboard==2.10.0 \
    scipy==1.7.3 \
    networkx==2.6.3 \
    tqdm==4.62.3 \
    matplotlib==3.5.3 \
    Pillow==9.3.0 \
    opencv-python-headless==4.6.0.66 \
    scikit-learn==1.0.2 \
    pandas==1.3.5

# ============================================================================
# Verify Installation
# ============================================================================

echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="

python3 << 'EOF'
import sys
print(f"✓ Python: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
except Exception as e:
    print(f"✗ PyTorch error: {e}")

try:
    import yaml
    print(f"✓ PyYAML: {yaml.__version__}")
except:
    print("✗ PyYAML not available")

try:
    from tensorboard.version import __version__
    print(f"✓ TensorBoard: {__version__}")
except:
    print("✗ TensorBoard not available")

print("")
print("✓ Setup complete!")
EOF

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Activate venv: source venv/bin/activate"
echo "2. Run tests: python test_correctness.py"
echo "3. Start training: python train_corrected.py"
echo "4. Monitor: tensorboard --logdir=./runs --host=0.0.0.0 --port=6006"
echo "=========================================="
