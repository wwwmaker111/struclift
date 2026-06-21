# 精简后恢复本地运行环境：重建 .venv 并安装依赖（在 E:\structlift 根目录执行）
#   powershell -ExecutionPolicy Bypass -File .\restore_env.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Venv = Join-Path $Root ".venv"
if (-not (Test-Path $Venv)) {
    Write-Host "Creating venv: $Venv"
    python -m venv $Venv
} else {
    Write-Host "Venv already exists: $Venv"
}

$Pip = Join-Path $Venv "Scripts\pip.exe"
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install -U pip
& $Pip install -r (Join-Path $Root "requirements.txt")
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
