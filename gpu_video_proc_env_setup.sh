#!/bin/bash

# 1. Check and install ffmpeg if it's not installed
if ! command -v ffmpeg &> /dev/null; then
    echo "[INFO] ffmpeg not found. Installing..."
    sudo apt install -y ffmpeg
else
    echo "[INFO] ffmpeg is already installed: $(ffmpeg -version | head -n 1)"
fi

# 2. Install nvtop
echo "[INFO] Installing nvtop..."
sudo apt install -y nvtop

# 3. Install pip if it's not installed
if ! command -v pip3 &> /dev/null; then
    echo "[INFO] pip not found. Installing..."
    sudo apt install -y python3-pip
else
    echo "[INFO] pip is already installed: $(pip3 --version)"
fi

# 4. Check and install venv module properly
if ! python3 -m venv test_env 2>/dev/null; then
    echo "[INFO] venv module not found or incomplete. Installing..."
    sudo apt install -y python3.12-venv
else
    echo "[INFO] venv is already available"
    rm -rf test_env
fi

# 5. Create ~/projects directory if it doesn't exist and navigate into it
mkdir -p ~/projects
cd ~/projects || exit 1

# 6. Create python directory inside projects if it doesn't exist and navigate into it
mkdir -p python
cd python || exit 1

# 7. Clone the repository if it hasn't been cloned already
if [ ! -d "UniframeStudio" ]; then
    git clone https://github.com/theblacksmithlab/UniframeStudio.git
else
    echo "[INFO] Uniframe_studio repository already exists"
fi

# 8. Navigate into the smart_dubbing repository
cd UniframeStudio || exit 1

# 9. Create virtual environment
python3 -m venv venv

# 10. Activate virtual environment and install requirements
source venv/bin/activate
pip install -r requirements.txt

# 11. Create required directories if they don't exist
mkdir -p video_input
mkdir -p output/timestamped_transcriptions

echo "[INFO] Environment setup complete."