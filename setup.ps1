<#
.SYNOPSIS
  Bootstrap script for vietdub on Windows. Idempotent.

.DESCRIPTION
  Creates .venv, installs the project (editable) via pyproject.toml with the upgraded stack.
  Re-run is a no-op if pyproject.toml hash matches build/installed-stack.txt.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# --- Prune old logs (older than 7 days) ---
$logDir = Join-Path $PSScriptRoot 'build'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("setup-{0}-{1}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID)
Get-ChildItem -Path $logDir -Filter 'setup-*.log' -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
    Remove-Item -Force

# Helper: tee to both stdout and log file
function Tee-Log {
    param([string]$Message)
    Write-Host $Message
    Add-Content -Path $logFile -Value $Message
}

Tee-Log "=== vietdub setup.ps1 starting at $(Get-Date -Format 'o') ==="

# --- Step 1: Python version check ---
$pyLauncher = (Get-Command 'py' -ErrorAction SilentlyContinue)
if (-not $pyLauncher) { throw "Python launcher 'py' not found in PATH. Install Python 3.11 from https://www.python.org/" }
$pyVerOutput = & py -3.11 --version 2>&1
if ($LASTEXITCODE -ne 0) { throw "'py -3.11' failed. Ensure Python 3.11 is installed and on PATH." }
$pyVer = ($pyVerOutput -replace 'Python ', '').Trim()
$pyMajor, $pyMinor = $pyVer.Split('.')[0..1] | ForEach-Object { [int]$_ }
if ($pyMajor -lt 3 -or ($pyMajor -eq 3 -and $pyMinor -lt 9) -or ($pyMajor -eq 3 -and $pyMinor -ge 14)) {
    throw "Python $pyVer not supported. Need >=3.9,<3.14 (paddlepaddle 3.x wheel availability)."
}
Tee-Log "Step 1 OK: Python $pyVer"

# --- Step 2: Architecture check ---
$arch = (python -c "import platform; print(platform.machine())" 2>&1).Trim()
if ($LASTEXITCODE -ne 0) { throw "Failed to detect architecture" }
if ($arch -ne 'AMD64') { throw "Architecture $arch not supported. paddlepaddle 3.x Windows wheels are amd64-only." }
Tee-Log "Step 2 OK: Architecture $arch"

# --- Step 3: VC++ runtime check ---
$vcCheck = python -c "import ctypes; ctypes.CDLL('vcruntime140.dll'); ctypes.CDLL('msvcp140.dll'); print('ok')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Tee-Log "VC++ runtime missing. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe"
    throw "VC++ 2019/2022 redistributable not found. Install and re-run."
}
Tee-Log "Step 3 OK: VC++ runtime present"

# --- Step 4: Hash check vs installed-stack.txt ---
$reqFile = Join-Path $PSScriptRoot 'pyproject.toml'
$venvDir = Join-Path $PSScriptRoot '.venv'
$hashFile = Join-Path $PSScriptRoot 'build\installed-stack.txt'

if (-not (Test-Path $reqFile)) { throw "pyproject.toml not found at $reqFile" }
$reqHash = (Get-FileHash $reqFile -Algorithm SHA256).Hash
if ((Test-Path $venvDir) -and (Test-Path $hashFile) -and ((Get-Content $hashFile -Raw) -eq $reqHash)) {
    Tee-Log "Step 4 OK: .venv already matches pyproject.toml hash. Skipping install."
    Print-Activation
    exit 0
}
if (Test-Path $venvDir) {
    Tee-Log "Hash mismatch or partial install. Removing .venv..."
    Remove-Item -Recurse -Force $venvDir
}

# --- Step 5: Create .venv ---
& py -3.11 -m venv $venvDir
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
Tee-Log "Step 5 OK: Created .venv"

# --- Step 6: Upgrade pip ---
& "$venvDir\Scripts\python.exe" -m pip install --upgrade pip --upgrade-strategy only-if-needed
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
Tee-Log "Step 6 OK: pip upgraded"

# --- Step 7: Install project (editable, from pyproject.toml) ---
& "$venvDir\Scripts\python.exe" -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "pip install -e . failed" }
Tee-Log "Step 7 OK: requirements installed"

# --- Step 8: PS version warning ---
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Tee-Log "Step 8 WARN: PowerShell 5.1 detected. PowerShell 7+ recommended for better error propagation."
}

# --- Step 9: Record hash and print activation ---
Set-Content -Path $hashFile -Value $reqHash -NoNewline
Tee-Log "Step 9 OK: Activation instructions below"
Tee-Log "  PowerShell: .\\.venv\\Scripts\\Activate.ps1"
Tee-Log "  cmd:        .venv\\Scripts\\activate.bat"
Tee-Log "  bash:       source .venv/bin/activate"
Tee-Log "=== setup.ps1 complete at $(Get-Date -Format 'o') ==="