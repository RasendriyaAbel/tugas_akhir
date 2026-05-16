#!/usr/bin/env powershell
# Quick start script untuk menjalankan NILM v9 web dashboard
# Usage: .\quick_start.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  NILM v9 Web Dashboard - Quick Start" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = Get-Location
$mlServiceDir = Join-Path $projectRoot "ml_service"
$venvPath = Join-Path $mlServiceDir ".venv"

# Check if .env file exists
$envFile = Join-Path $projectRoot ".env"
if (Test-Path $envFile) {
    Write-Host "✓ .env file ditemukan" -ForegroundColor Green
} else {
    Write-Host "⚠ .env file tidak ditemukan. Pastikan sudah setup NILM_MODEL_DIR" -ForegroundColor Yellow
}

# Check if model directory exists
$modelDir = Join-Path $projectRoot "src" "nilm_models_v9"
if (Test-Path $modelDir) {
    Write-Host "✓ Model directory: $modelDir" -ForegroundColor Green
} else {
    Write-Host "❌ Model directory tidak ditemukan!" -ForegroundColor Red
    exit 1
}

# Verify model files
$requiredFiles = @("best_nilm_model.keras", "meta_nilm.json")
$allFilesExist = $true
foreach ($file in $requiredFiles) {
    $filePath = Join-Path $modelDir $file
    if (Test-Path $filePath) {
        $size = (Get-Item $filePath).Length / 1MB
        Write-Host "  ✓ $file ($([math]::Round($size, 2)) MB)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file missing!" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Host ""
    Write-Host "Missing model files! Cannot continue." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  TERMINAL 1: ML Service (Flask Backend)" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Untuk menjalankan ML Service, buka terminal baru dan jalankan:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  cd ml_service" -ForegroundColor White
Write-Host "  .\.venv\Scripts\activate" -ForegroundColor White
Write-Host "  python app.py" -ForegroundColor White
Write-Host ""
Write-Host "Service akan berjalan di: http://127.0.0.1:5001" -ForegroundColor Cyan
Write-Host ""

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  TERMINAL 2: Frontend (Next.js)" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Check if node_modules exists
$nodeModules = Join-Path $projectRoot "node_modules"
if (Test-Path $nodeModules) {
    Write-Host "✓ node_modules ditemukan" -ForegroundColor Green
} else {
    Write-Host "Installing npm dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ npm install gagal!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Menjalankan frontend development server..." -ForegroundColor Yellow
Write-Host ""

npm run dev

Write-Host ""
Write-Host "Frontend berjalan di: http://localhost:3000" -ForegroundColor Cyan
