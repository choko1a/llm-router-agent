$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

Set-Location $ProjectRoot

if (-not (Test-Path $VenvPython)) {
    Write-Host ".venv was not found or does not contain Scripts\python.exe."
    Write-Host "Run ./scripts/bootstrap_env.ps1 first."
    exit 1
}

try {
    & $VenvPython -c "import sys; print(sys.executable)" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "venv interpreter failed"
    }
}
catch {
    Write-Host ".venv exists but its Python interpreter is not usable."
    Write-Host "Run ./scripts/bootstrap_env.ps1 to recreate it."
    exit 1
}

. $ActivateScript

Write-Host "Running tests with:"
& python -c "import sys; print('Interpreter: ' + sys.executable); print('Version: ' + sys.version.replace(chr(10), ' '))"

$pytestOutput = & python -m pytest 2>&1
$pytestExitCode = $LASTEXITCODE
$pytestOutput | ForEach-Object { Write-Host $_ }

$joined = $pytestOutput -join "`n"
$passed = "unknown"
if ($joined -match "(\d+)\s+passed") {
    $passed = $Matches[1]
}

Write-Host ""
if ($pytestExitCode -eq 0) {
    Write-Host "Result: success"
}
else {
    Write-Host "Result: fail"
}
Write-Host "Passed tests: $passed"

exit $pytestExitCode
