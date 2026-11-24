param([string]$Python = 'C:\Users\yzdys\miniconda3\envs\py312\python.exe', [int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
Push-Location -LiteralPath $projectRoot
try {
    & $Python scripts/serve_steel_inspection.py --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Service exited with code $LASTEXITCODE" }
} finally { Pop-Location }
