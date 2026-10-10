#!/usr/bin/env bash
# Setup script for g1-qtm-teleop
# Installs all dependencies in one shot.

set -e  # stop on first error

GMR_COMMIT=bb1bbe4   # GMR version the QTM patch was made against

echo "============================================"
echo "  QTM Teleop G1 Setup"
echo "============================================"
echo ""

# ── 1. Python packages ────────────────────────────────────────────────────────
echo "[1/4] Installing Python packages ..."
pip install -r requirements.txt
echo "      Done."
echo ""

# ── 2. GMR retargeting library (patched for QTM) ─────────────────────────────
echo "[2/4] Installing GMR (General Motion Retargeting) ..."
if [ ! -d "GMR" ]; then
    git clone https://github.com/YanjieZe/GMR.git
    git -C GMR checkout -q "$GMR_COMMIT"
fi
if git -C GMR apply --check ../gmr_patch/gmr_qtm.patch 2>/dev/null; then
    git -C GMR apply ../gmr_patch/gmr_qtm.patch
    echo "      Applied QTM patch."
else
    echo "      QTM patch already applied (or GMR has local changes), skipping."
fi
cp gmr_patch/ik_configs/*.json GMR/general_motion_retargeting/ik_configs/
pip install -e GMR
echo "      GMR installed."
echo ""

# ── 3. Unitree SDK (only needed to drive the real robot) ─────────────────────
echo "[3/4] Checking for unitree_sdk2py ..."
if python -c "import unitree_sdk2py" 2>/dev/null; then
    echo "      Found."
else
    echo "      Not found. Dry runs (--dry-run) work without it."
    echo "      To drive the real G1, install it from:"
    echo "        https://github.com/unitreerobotics/unitree_sdk2_python"
fi
echo ""

# ── 4. Check Blender (only needed for offline FBX retargeting) ───────────────
echo "[4/4] Checking for Blender ..."
if command -v blender &>/dev/null; then
    BLENDER_VER=$(blender --version 2>/dev/null | head -1)
    echo "      Found: $BLENDER_VER"
else
    echo ""
    echo "  WARNING: 'blender' command not found."
    echo "  Blender is only needed for offline FBX retargeting (fbx_to_g1.py)."
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
echo "  Quick test (no robot, MuJoCo viewer only):"
echo "    python scripts/teleop_session.py --dry-run --retargeter gmr"
echo "============================================"
