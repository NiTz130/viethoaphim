# setup-2026.ps1 — create venv-2026 with the new pinned stack.
# Idempotent: if venv-2026 exists with matching hash, skip; otherwise nuke and recreate.
$ErrorActionPreference = 'Stop'
$venvDir = Join-Path $PSScriptRoot '..\.venv-2026'
$reqFile = Join-Path $PSScriptRoot '..\requirements-2026.txt'
$hashFile = Join-Path $PSScriptRoot '..\build\installed-stack-2026.txt'

if (-not (Test-Path $reqFile)) {
    throw "requirements-2026.txt not found at $reqFile"
}

$reqHash = (Get-FileHash $reqFile -Algorithm SHA256).Hash
if (Test-Path $venvDir) {
    if ((Test-Path $hashFile) -and ((Get-Content $hashFile -Raw) -eq $reqHash)) {
        Write-Host "venv-2026 already up to date. Skipping."
        exit 0
    }
    Write-Host "Hash mismatch or partial install. Removing venv-2026..."
    Remove-Item -Recurse -Force $venvDir
}

py -3.11 -m venv $venvDir
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }

& "$venvDir\Scripts\python.exe" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

# Install vietdub editable so `import vietdub.ocr.regression` works in venv-2026.
# (Required for tests/ocr_regression.py and tests/integration/test_full_pipeline_2026.py.)
& "$venvDir\Scripts\python.exe" -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "pip install -e . failed" }

& "$venvDir\Scripts\python.exe" -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements-2026.txt failed" }

New-Item -ItemType Directory -Force -Path (Split-Path $hashFile) | Out-Null
Set-Content -Path $hashFile -Value $reqHash -NoNewline
Write-Host "venv-2026 ready at $venvDir"
