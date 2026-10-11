#!/usr/bin/env bash
# ==============================================================================
# AI-Powered Aquifer Groundwater Decision-Support System
# Cross-Platform Linux & macOS Auto-Launcher
# ==============================================================================

set -e

echo "=============================================================================="
echo "  AI-POWERED AQUIFER GROUNDWATER FORECASTING & SMART WATER MANAGEMENT SYSTEM"
echo "=============================================================================="
echo ""

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

echo "[1/4] Python Version: $(python3 --version)"

# 2. Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "[2/4] Virtual environment exists."
fi

source .venv/bin/activate

# 3. Dependencies
echo "[3/4] Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r backend/requirements.txt --quiet

# 4. Generate data if missing
if [ ! -f "backend/app/data/aquifer_multistation.csv" ]; then
    echo "Initializing datasets..."
    cd backend
    python3 -m app.core.data_generator
    cd ..
fi

echo "[4/4] Starting server on 0.0.0.0:8000..."
echo ""
echo "=============================================================================="
echo "  LOCAL ACCESS:        http://127.0.0.1:8000/"
echo "  LOCAL NETWORK (LAN): http://$(hostname -I | awk '{print $1}'):8000/"
echo "  SWAGGER API DOCS:    http://127.0.0.1:8000/docs"
echo "=============================================================================="
echo "Press Ctrl+C to stop."
echo ""

# Auto-open browser on macOS or Linux if available
if command -v xdg-open &> /dev/null; then
    (sleep 2 && xdg-open http://127.0.0.1:8000/) &
elif command -v open &> /dev/null; then
    (sleep 2 && open http://127.0.0.1:8000/) &
fi

cd backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
