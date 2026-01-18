#!/bin/bash
# Nemotron Speech Streaming to MLX Conversion
#
# This script runs the complete conversion pipeline.
# Steps 1-4 can run on any platform with Python and PyTorch.
# Step 5 requires Apple Silicon Mac with MLX.

set -e  # Exit on first error
set -u  # Error on undefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Nemotron Speech Streaming MLX Conversion${NC}"
echo -e "${BLUE}======================================${NC}"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "Python: ${GREEN}$PYTHON_VERSION${NC}"

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo -e "${YELLOW}WARNING: Not in a virtual environment${NC}"
    echo "Consider: python3 -m venv .venv && source .venv/bin/activate"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "Virtual env: ${GREEN}$VIRTUAL_ENV${NC}"
fi

echo

# Check for required packages
echo -e "${BLUE}Checking dependencies...${NC}"
python3 -c "import torch" 2>/dev/null || {
    echo -e "${RED}ERROR: torch not installed${NC}"
    echo "Run: pip install -r requirements.txt"
    exit 1
}
python3 -c "import huggingface_hub" 2>/dev/null || {
    echo -e "${RED}ERROR: huggingface-hub not installed${NC}"
    echo "Run: pip install -r requirements.txt"
    exit 1
}
python3 -c "import safetensors" 2>/dev/null || {
    echo -e "${RED}ERROR: safetensors not installed${NC}"
    echo "Run: pip install -r requirements.txt"
    exit 1
}
python3 -c "import rich" 2>/dev/null || {
    echo -e "${RED}ERROR: rich not installed${NC}"
    echo "Run: pip install -r requirements.txt"
    exit 1
}
echo -e "${GREEN}Core dependencies OK${NC}"
echo

# Run conversion steps
run_step() {
    local step_num=$1
    local script=$2
    local description=$3

    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}Step $step_num: $description${NC}"
    echo -e "${BLUE}======================================${NC}"

    python3 "scripts/$script"

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}Step $step_num completed successfully${NC}"
    else
        echo -e "${RED}Step $step_num FAILED${NC}"
        exit 1
    fi
    echo
}

# Parse arguments
SKIP_DOWNLOAD=false
SKIP_EXTRACT=false
SKIP_ANALYZE=false
SKIP_CONVERT=false
SKIP_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
        --skip-analyze)
            SKIP_ANALYZE=true
            shift
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --skip-download  Skip model download (use existing .nemo)"
            echo "  --skip-extract   Skip weight extraction"
            echo "  --skip-analyze   Skip architecture analysis"
            echo "  --skip-convert   Skip MLX conversion"
            echo "  --skip-test      Skip inference testing"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Step 1: Download
if [[ "$SKIP_DOWNLOAD" == false ]]; then
    run_step 1 "01_download_model.py" "Download Model"
else
    echo -e "${YELLOW}Skipping Step 1: Download${NC}"
fi

# Step 2: Extract
if [[ "$SKIP_EXTRACT" == false ]]; then
    run_step 2 "02_extract_weights.py" "Extract Weights"
else
    echo -e "${YELLOW}Skipping Step 2: Extract${NC}"
fi

# Step 3: Analyze
if [[ "$SKIP_ANALYZE" == false ]]; then
    run_step 3 "03_analyze_architecture.py" "Analyze Architecture"
else
    echo -e "${YELLOW}Skipping Step 3: Analyze${NC}"
fi

# Step 4: Convert
if [[ "$SKIP_CONVERT" == false ]]; then
    run_step 4 "04_convert_to_mlx.py" "Convert to MLX"
else
    echo -e "${YELLOW}Skipping Step 4: Convert${NC}"
fi

# Step 5: Test (only on Mac)
if [[ "$SKIP_TEST" == false ]]; then
    if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        # Check for MLX
        if python3 -c "import mlx" 2>/dev/null; then
            run_step 5 "05_test_inference.py" "Test Inference"
        else
            echo -e "${YELLOW}MLX not installed - skipping Step 5${NC}"
            echo "Install with: pip install -r requirements-mlx.txt"
        fi
    else
        echo -e "${YELLOW}Not on Apple Silicon - skipping Step 5${NC}"
        echo "Transfer output/mlx to your Mac to test"
    fi
else
    echo -e "${YELLOW}Skipping Step 5: Test${NC}"
fi

echo
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Conversion Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo
echo "Output directory: output/mlx/"
echo
echo "Files:"
ls -la output/mlx/ 2>/dev/null || echo "(Run on Mac to generate MLX files)"
