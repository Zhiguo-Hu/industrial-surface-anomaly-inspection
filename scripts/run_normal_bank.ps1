param(
    [ValidateSet('verify','build','predict')][string]$Mode = 'verify',
    [string]$Image,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $env:USERPROFILE 'miniconda3\envs\py312\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Specify the configured Python executable with -Python.'
}
$argumentList = @((Join-Path $PSScriptRoot 'build_steel_normal_bank.py'))
if ($Mode -eq 'verify') {
    $argumentList += '--verify'
} elseif ($Mode -eq 'predict') {
    if (-not $Image -or -not (Test-Path -LiteralPath $Image -PathType Leaf)) {
        throw 'Use -Image with an existing image path.'
    }
    $resolvedImage = (Resolve-Path -LiteralPath $Image).Path
    $argumentList += @('--image', $resolvedImage)
}
Push-Location -LiteralPath $projectRoot
try {
    & $Python @argumentList
    if ($LASTEXITCODE -ne 0) { throw 'Normal-bank operation did not complete; model outputs remain in runs/steel_normal_bank.' }
    Write-Output ('Output directory: ' + (Join-Path $projectRoot 'runs\steel_normal_bank'))
} finally {
    Pop-Location
}
