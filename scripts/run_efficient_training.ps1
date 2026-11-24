param([switch]$Resume, [string]$Python = 'C:\Users\yzdys\miniconda3\envs\py312\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
Push-Location -LiteralPath $projectRoot
try {
    if ($Resume) { & $Python scripts/train_efficient_strip.py --epochs 12 --resume }
    else { & $Python scripts/train_efficient_strip.py --epochs 12 }
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally { Pop-Location }
