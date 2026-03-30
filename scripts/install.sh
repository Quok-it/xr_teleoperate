#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-$HOME/waldo}"

echo "Installing waldo to $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Unpack conda environment
echo "Unpacking Python environment..."
mkdir -p "$INSTALL_DIR/env"
tar xzf "$SCRIPT_DIR/tv-env.tar.gz" -C "$INSTALL_DIR/env"

# Fix prefixes for this machine
echo "Fixing environment prefixes..."
source "$INSTALL_DIR/env/bin/activate"
conda-unpack
deactivate

# Unpack application source
echo "Unpacking application..."
tar xzf "$SCRIPT_DIR/xr_teleoperate.tar.gz" -C "$INSTALL_DIR"

# Install system deps if needed
if ! dpkg -s libturbojpeg0-dev &>/dev/null; then
    echo ""
    echo "NOTE: libturbojpeg-dev is not installed. Install it with:"
    echo "  sudo apt-get install -y libturbojpeg-dev"
fi

# Create launcher script
cat > "$INSTALL_DIR/run.sh" << 'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

WALDO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$WALDO_DIR/env/bin/activate"
cd "$WALDO_DIR/teleop"
python teleop_hand_and_arm.py "$@"
LAUNCHER
chmod +x "$INSTALL_DIR/run.sh"

echo ""
echo "Done. Run with:"
echo "  $INSTALL_DIR/run.sh --arm=G1_29 --ee=dex3"
echo ""
echo "Or activate the environment manually:"
echo "  source $INSTALL_DIR/env/bin/activate"
