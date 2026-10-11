@echo off
title AI-Powered Aquifer Groundwater Decision-Support System
color 0B

echo ==============================================================================
echo   AI-POWERED AQUIFER GROUNDWATER FORECASTING & SMART WATER MANAGEMENT SYSTEM
echo ==============================================================================
echo.

:: 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found in system PATH.
    echo Please install Python 3.10 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python environment...
python --version

:: 2. Set up virtual environment
if not exist ".venv" (
    echo [2/4] Creating virtual environment (.venv)...
    python -m venv .venv
) else (
    echo [2/4] Virtual environment (.venv) already exists.
)

echo Activating virtual environment...
call .venv\Scripts\activate

:: 3. Install / Verify Dependencies
echo [3/4] Verifying and installing required packages...
pip install --upgrade pip --quiet
pip install -r backend\requirements.txt --quiet

:: 4. Ensure datasets are initialized
if not exist "backend\app\data\aquifer_multistation.csv" (
    echo Initializing hydrogeological dataset...
    cd backend
    python -m app.core.data_generator
    cd ..
)

echo [4/4] Starting Aquifer Forecasting & Smart Water Management Server...
echo.
echo ==============================================================================
echo   LOCAL ACCESS:          http://127.0.0.1:8000/
echo   LOCAL NETWORK (LAN):   Access via your machine's IP (e.g. http://192.168.x.x:8000/)
echo   API DOCUMENTATION:     http://127.0.0.1:8000/docs
echo ==============================================================================
echo Press Ctrl+C in this terminal window to stop the server.
echo.

:: Launch default browser automatically after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000/"

:: Run the FastAPI server bound to 0.0.0.0
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

pause
