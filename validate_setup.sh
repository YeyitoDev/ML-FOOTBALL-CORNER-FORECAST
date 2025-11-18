#!/bin/bash

# Quick validation script to check if all files are in place

echo "🔍 Validating Football Prediction Model Setup..."
echo "=================================================="

errors=0

# Check required files
echo -e "\n📋 Checking required files..."

files=(
    "Dockerfile"
    "docker-compose.yml"
    "requirements.txt"
    "Makefile"
    "setup_and_run.sh"
    "src/main.py"
    "src/utils.py"
    "src/models.py"
    "src/train_model.py"
    "src/predict.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - MISSING"
        ((errors++))
    fi
done

# Check required directories
echo -e "\n📁 Checking required directories..."

dirs=(
    "data"
    "src"
    "pipelines"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/"
    else
        echo "  ❌ $dir/ - MISSING"
        ((errors++))
    fi
done

# Check Python
echo -e "\n🐍 Checking Python..."
if command -v python3 &> /dev/null; then
    version=$(python3 --version)
    echo "  ✅ $version"
else
    echo "  ❌ Python 3 not found"
    ((errors++))
fi

# Check Docker (optional)
echo -e "\n🐳 Checking Docker (optional)..."
if command -v docker &> /dev/null; then
    version=$(docker --version)
    echo "  ✅ $version"
else
    echo "  ⚠️  Docker not found (optional)"
fi

if command -v docker-compose &> /dev/null; then
    version=$(docker-compose --version)
    echo "  ✅ $version"
else
    echo "  ⚠️  docker-compose not found (optional)"
fi

# Check make (optional)
echo -e "\n🔨 Checking Make (optional)..."
if command -v make &> /dev/null; then
    version=$(make --version | head -n1)
    echo "  ✅ $version"
else
    echo "  ⚠️  Make not found (optional)"
fi

# Summary
echo -e "\n=================================================="
if [ $errors -eq 0 ]; then
    echo "✅ All required files and dependencies are present!"
    echo ""
    echo "You can now run:"
    echo "  make all              # Complete pipeline with venv"
    echo "  ./setup_and_run.sh    # Automated script"
    echo "  docker-compose up     # Docker deployment"
else
    echo "❌ Found $errors error(s). Please fix before proceeding."
    exit 1
fi
echo "=================================================="
