param([string]$Python = 'C:\Users\yzdys\miniconda3\envs\py312\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'runs/strip_inspector_refined/last.pt'))) {
    throw 'Local training checkpoint is missing. Restore local runs before resuming.'
}
Push-Location -LiteralPath $projectRoot
try {
    & $Python scripts/train_strip_inspector.py --resume `
        --run-name strip_inspector_refined `
        --warm-checkpoint runs/strip_inspector/best.pt `
        --epochs 8 --batch-size 16 --crop-width 800 --crop-probability 0.5 `
        --mask-every 1 --auxiliary-classes `
        --encoder-lr 0.00003 --decoder-lr 0.0002 --detector-lr 0.0001
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally { Pop-Location }
