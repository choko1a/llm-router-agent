$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Test-PythonExecutable {
    param([string]$PythonCommand)

    try {
        $output = & $PythonCommand -c "import sys; print(sys.executable); print(sys.version.replace(chr(10), ' '))" 2>&1
        if ($LASTEXITCODE -ne 0 -or $output.Count -lt 2) {
            return $null
        }

        return [pscustomobject]@{
            Command = $PythonCommand
            Path = [string]$output[0]
            Version = [string]$output[1]
        }
    }
    catch {
        return $null
    }
}

function Find-BasePython {
    Write-Step "Checking Python candidates"

    foreach ($candidate in @("python", "py")) {
        Write-Host "Trying '$candidate'..."
        $python = Test-PythonExecutable -PythonCommand $candidate
        if ($null -ne $python) {
            Write-Host "Found Python via '$candidate'"
            Write-Host "Interpreter: $($python.Path)"
            Write-Host "Version: $($python.Version)"
            return $python
        }

        Write-Host "'$candidate' is not usable in this shell."
    }

    Write-Host ""
    Write-Host "Python was not found."
    Write-Host "Install Python 3.9+ and ensure either 'python' or 'py' works in PowerShell."
    Write-Host "If 'py' points to an old removed interpreter, repair the Python Launcher registration or reinstall Python."
    exit 1
}

function Test-Venv {
    if (-not (Test-Path $VenvPython)) {
        return $false
    }

    try {
        & $VenvPython -c "import sys; print(sys.executable)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Reset-BrokenVenv {
    if (-not (Test-Path $VenvPath)) {
        return
    }

    $resolvedProject = (Resolve-Path $ProjectRoot).Path
    $resolvedVenv = (Resolve-Path $VenvPath).Path
    if (-not $resolvedVenv.StartsWith($resolvedProject, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove venv outside project root: $resolvedVenv"
    }

    Write-Step "Removing broken .venv"
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

function Ensure-Venv {
    param([pscustomobject]$BasePython)

    if (Test-Venv) {
        Write-Step "Using existing .venv"
        return
    }

    if (Test-Path $VenvPath) {
        Reset-BrokenVenv
    }

    Write-Step "Creating fresh .venv"
    & $BasePython.Command -m venv $VenvPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Venv)) {
        throw "Failed to create a working .venv."
    }
}

function Install-Requirements {
    Write-Step "Activating .venv"
    . (Join-Path $VenvPath "Scripts\Activate.ps1")

    Write-Step "Python environment"
    & python -c "import sys; print('Interpreter: ' + sys.executable); print('Version: ' + sys.version.replace(chr(10), ' '))"

    Write-Step "Upgrading pip"
    & python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed."
    }

    Write-Step "Installing requirements"
    & python -m pip install -r $RequirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "requirements installation failed."
    }
}

function Invoke-Pytest {
    Write-Step "Running pytest"
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
}

Set-Location $ProjectRoot

Write-Host "LLM Router Agent Windows bootstrap"
Write-Host "Project root: $ProjectRoot"

$basePython = Find-BasePython
Ensure-Venv -BasePython $basePython
Install-Requirements
Invoke-Pytest
