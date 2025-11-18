@echo off
REM Football Prediction Model - Setup and Run Script (Windows)
REM This script sets up a virtual environment, installs dependencies, 
REM and runs the complete ML pipeline

echo ==================================================
echo ⚽ Football Prediction Model - Setup ^& Run
echo ==================================================

REM Configuration
set VENV_DIR=venv
set PYTHON=python

REM Step 1: Check Python version
echo.
echo 📋 Step 1: Checking Python version...
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.9 or higher.
    exit /b 1
)
for /f "tokens=*" %%i in ('%PYTHON% --version') do set PYTHON_VER=%%i
echo ✅ Found: %PYTHON_VER%

REM Step 2: Create virtual environment
echo.
echo 📦 Step 2: Creating virtual environment...
if exist %VENV_DIR% (
    echo ⚠️  Virtual environment already exists. Removing old one...
    rmdir /s /q %VENV_DIR%
)
%PYTHON% -m venv %VENV_DIR%
echo ✅ Virtual environment created

REM Step 3: Activate virtual environment
echo.
echo 🔌 Step 3: Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
echo ✅ Virtual environment activated

REM Step 4: Upgrade pip
echo.
echo ⬆️  Step 4: Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✅ pip upgraded

REM Step 5: Install dependencies
echo.
echo 📚 Step 5: Installing dependencies...
echo    This may take a few minutes...
pip install -r requirements.txt --quiet
echo ✅ All dependencies installed

REM Step 6: Create necessary directories
echo.
echo 📁 Step 6: Creating necessary directories...
if not exist data\models mkdir data\models
if not exist data\output mkdir data\output
if not exist data\predictions mkdir data\predictions
echo ✅ Directories created

REM Step 7: Run main data processing pipeline
echo.
echo 🔄 Step 7: Running data processing pipeline...
cd src
python main.py
cd ..
echo ✅ Data processing complete

REM Step 8: Train ML models
echo.
echo 🤖 Step 8: Training ML models...
cd src
python train_model.py
cd ..
echo ✅ Model training complete

REM Step 9: Summary
echo.
echo ==================================================
echo ✅ SETUP AND TRAINING COMPLETE!
echo ==================================================
echo.
echo 📊 Generated files:
echo    - Data: data\output\
echo    - Models: data\models\
echo.
echo 🎯 To use the trained model:
echo    %VENV_DIR%\Scripts\activate.bat
echo    cd src
echo    python predict.py
echo.
echo 🐳 To run with Docker instead:
echo    docker-compose up --build
echo.
echo ==================================================

REM Deactivate virtual environment
call %VENV_DIR%\Scripts\deactivate.bat
pause
