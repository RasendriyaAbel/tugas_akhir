#!/bin/bash
# Quick start script untuk menjalankan NILM v9 web dashboard (Linux/Mac)
# Usage: bash quick_start.sh

echo ""
echo "======================================"
echo "  NILM v9 Web Dashboard - Quick Start"
echo "======================================"
echo ""

PROJECT_ROOT=$(pwd)
ML_SERVICE_DIR="$PROJECT_ROOT/ml_service"
VENV_PATH="$ML_SERVICE_DIR/.venv"
MODEL_DIR="$PROJECT_ROOT/src/nilm_models_v9"

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model directory not found: $MODEL_DIR"
    exit 1
fi

echo "✓ Model directory: $MODEL_DIR"

# Verify model files
echo ""
for file in "best_nilm_model.keras" "meta_nilm.json"; do
    if [ -f "$MODEL_DIR/$file" ]; then
        SIZE=$(du -h "$MODEL_DIR/$file" | cut -f1)
        echo "  ✓ $file ($SIZE)"
    else
        echo "  ❌ $file missing!"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "  TERMINAL 1: ML Service (Flask Backend)"
echo "=========================================="
echo ""
echo "Open new terminal and run:"
echo ""
echo "  cd ml_service"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo ""
echo "Service will run at: http://127.0.0.1:5001"
echo ""

echo "=========================================="
echo "  TERMINAL 2: Frontend (Next.js)"
echo "=========================================="
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ npm install failed!"
        exit 1
    fi
fi

echo "Starting frontend development server..."
echo ""

npm run dev

echo ""
echo "Frontend running at: http://localhost:3000"
