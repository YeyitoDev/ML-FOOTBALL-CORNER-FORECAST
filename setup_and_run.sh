#!/bin/bash

# Football Prediction Model - Setup and Run Script
# This script sets up a virtual environment, installs dependencies, 
# and runs the complete ML pipeline

set -e  # Exit on error

echo "=================================================="
echo "⚽ Football Prediction Model - Setup & Run"
echo "=================================================="

# Configuration
VENV_DIR="venv"
PYTHON_VERSION="python3"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check Python version
echo -e "\n${BLUE}📋 Step 1: Checking Python version...${NC}"
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi
PYTHON_VER=$($PYTHON_VERSION --version)
echo -e "${GREEN}✅ Found: $PYTHON_VER${NC}"

# Step 2: Create virtual environment
echo -e "\n${BLUE}📦 Step 2: Creating virtual environment...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists. Removing old one...${NC}"
    rm -rf $VENV_DIR
fi

$PYTHON_VERSION -m venv $VENV_DIR
echo -e "${GREEN}✅ Virtual environment created${NC}"

# Step 3: Activate virtual environment
echo -e "\n${BLUE}🔌 Step 3: Activating virtual environment...${NC}"
source $VENV_DIR/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Step 4: Upgrade pip
echo -e "\n${BLUE}⬆️  Step 4: Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✅ pip upgraded${NC}"

# Step 5: Install dependencies
echo -e "\n${BLUE}📚 Step 5: Installing dependencies...${NC}"
echo "   This may take a few minutes..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}✅ All dependencies installed${NC}"

# Step 6: Create necessary directories
echo -e "\n${BLUE}📁 Step 6: Creating necessary directories...${NC}"
mkdir -p data/models
mkdir -p data/output
mkdir -p data/predictions
echo -e "${GREEN}✅ Directories created${NC}"

# Step 7: Run main data processing pipeline
echo -e "\n${BLUE}🔄 Step 7: Running data processing pipeline...${NC}"
cd src
python main.py
cd ..
echo -e "${GREEN}✅ Data processing complete${NC}"

# Step 8: Train ML models
echo -e "\n${BLUE}🤖 Step 8: Training ML models...${NC}"
cd src
python train_model.py
cd ..
echo -e "${GREEN}✅ Model training complete${NC}"

# Step 9: Summary
echo -e "\n=================================================="
echo -e "${GREEN}✅ SETUP AND TRAINING COMPLETE!${NC}"
echo "=================================================="
echo ""
echo "📊 Generated files:"
echo "   - Data: data/output/"
echo "   - Models: data/models/"
echo ""
echo "🎯 To use the trained model:"
echo "   source venv/bin/activate"
echo "   cd src"
echo "   python predict.py"
echo ""
echo "🐳 To run with Docker instead:"
echo "   docker-compose up --build"
echo ""
echo "=================================================="

# Deactivate virtual environment
deactivate
