#!/usr/bin/env bash
# Setup script for qualisys-fbx-to-g1
# Installs all dependencies in one shot.

set -e  # stop on first error

echo "============================================"
echo "  Qualisys FBX → G1 Setup"
echo "============================================"
echo ""

# ── 1. Python packages ────────────────────────────────────────────────────────
echo "[1/3] Installing Python packages ..."
pip install numpy scipy tqdm
echo "      Done."
echo ""

# ── 2. GMR retargeting library ───────────────────────────────────────────────
echo "[2/3] Installing GMR (General Motion Retargeting) ..."
if python -c "import general_motion_retargeting" 2>/dev/null; then
    echo "      GMR already installed, skipping."
else
    if [ ! -d "GMR" ]; then
        git clone https://github.com/YanjieZe/GMR.git
    fi
    cd GMR
    pip install -e .
    cd ..
    echo "      GMR installed."
fi
echo ""

# ── 3. Check Blender ─────────────────────────────────────────────────────────
echo "[3/3] Checking for Blender ..."
if command -v blender &>/dev/null; then
    BLENDER_VER=$(blender --version 2>/dev/null | head -1)
    echo "      Found: $BLENDER_VER"
else
    echo ""
    echo "  WARNING: 'blender' command not found."
    echo "  Blender is required to read FBX files."
    echo ""
    echo "  Install options:"
    echo "    Ubuntu/Debian:  sudo snap install blender --classic"
    echo "    or download from: https://www.blender.org/download/"
    echo "    (Blender 3.6 LTS or newer recommended)"
    echo ""
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Quick test:"
echo "    python fbx_to_g1.py your_recording.fbx"
echo "============================================"
