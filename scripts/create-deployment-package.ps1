# =============================================================================
# Create Lambda Deployment Package
# Creates a zip file with all dependencies and application code
# =============================================================================

param(
    [string]$OutputFile = "deployment-package.zip"
)

$ErrorActionPreference = "Stop"
# In PowerShell 7+, native stderr can be promoted to terminating errors.
# pip emits non-fatal dependency warnings on stderr, so disable this behavior
# for the packaging script and rely on $LASTEXITCODE instead.
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "  Creating Lambda Deployment Package" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# Get script directory
$ScriptDir = Split-Path -Parent $PSCommandPath
$LambdaDir = Split-Path -Parent $ScriptDir
$PackageDir = Join-Path $LambdaDir "package"

# Clean previous package
if (Test-Path $PackageDir) {
    Write-Host "[1/5] Cleaning previous package..." -ForegroundColor Yellow
    Remove-Item $PackageDir -Recurse -Force
}

New-Item -ItemType Directory -Path $PackageDir | Out-Null

# Install dependencies
# Prefer requirements-lambda.txt (lighter, excludes dev-only packages like mlflow)
$ReqFile = Join-Path $LambdaDir "requirements-lambda.txt"
if (-not (Test-Path $ReqFile)) {
    $ReqFile = Join-Path $LambdaDir "requirements.txt"
}
Write-Host "[2/5] Installing dependencies (Linux compatible)..." -ForegroundColor Yellow
Write-Host "   Requirements: $(Split-Path $ReqFile -Leaf)" -ForegroundColor Gray
Write-Host "   Using manylinux2014_x86_64 platform for Lambda compatibility" -ForegroundColor Gray

# Install Linux-compatible wheels for Lambda (Amazon Linux runtime)
# NOTE: pip may warn about conflicts with OTHER packages installed in the current
# venv (e.g. python-jobspy from SupaBackend). These warnings are harmless because
# requirements-lambda.txt only installs exactly what Lambda needs.
$PrevErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $PipOutput = python -m pip install -r $ReqFile `
        -t $PackageDir `
        --platform manylinux2014_x86_64 `
        --only-binary=:all: `
        --upgrade `
        --quiet 2>&1
} finally {
    $ErrorActionPreference = $PrevErrorActionPreference
}

$ExitCode = $LASTEXITCODE

# Print output, suppressing noisy cross-project dependency conflict warnings
$PipOutput | Where-Object {
    $_ -notmatch "dependency conflicts" -and
    $_ -notmatch "python-jobspy" -and
    $_ -notmatch "incompatible"
} | ForEach-Object { Write-Host $_ }

if ($ExitCode -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Write-Host "Ensure pip >= 20.3 for --platform support" -ForegroundColor Yellow
    exit 1
}

# Copy application code
Write-Host "[3/5] Copying application code..." -ForegroundColor Yellow
Copy-Item (Join-Path $LambdaDir "lambda_handler.py") $PackageDir -Force
Copy-Item (Join-Path $LambdaDir "app") $PackageDir -Recurse -Force

# Remove unnecessary files to reduce package size
Write-Host "[4/5] Optimizing package size..." -ForegroundColor Yellow
Get-ChildItem $PackageDir -Recurse -Include "__pycache__", "*.pyc", "*.pyo", "tests", "test", "*.dist-info" |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Remove root-level files that could shadow Python stdlib modules.
Write-Host "   Checking for module shadowing conflicts..." -ForegroundColor Gray
$ConflictingFiles = @("typing.py", "http.py", "types.py", "abc.py", "collections.py")
$ConflictsFound = $false

foreach ($file in $ConflictingFiles) {
    $RootFile = Join-Path $PackageDir $file
    if (Test-Path $RootFile) {
        Write-Host "   Removing conflicting $file from package root" -ForegroundColor Yellow
        Remove-Item $RootFile -Force
        $ConflictsFound = $true
    }
}

if ($ConflictsFound) {
    Write-Host "   Module conflicts resolved" -ForegroundColor Green
} else {
    Write-Host "   No module conflicts detected" -ForegroundColor Green
}

# Create zip file
Write-Host "[5/5] Creating deployment package..." -ForegroundColor Yellow
$OutputPath = Join-Path $LambdaDir $OutputFile

if (Test-Path $OutputPath) {
    Remove-Item $OutputPath -Force
}

# Compress-Archive can produce duplicate flat entries in some environments.
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $PackageDir,
    $OutputPath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)

Write-Host "Package created: $OutputPath" -ForegroundColor Green

# Show package size
$Size = (Get-Item $OutputPath).Length / 1MB
Write-Host "Package size: $([math]::Round($Size, 2)) MB" -ForegroundColor Cyan

if ($Size -gt 50) {
    Write-Host "WARNING: Package is larger than 50MB, deployment will use S3 upload." -ForegroundColor Yellow
}

# Clean up package directory
Remove-Item $PackageDir -Recurse -Force

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Upload to Lambda: .\scripts\deploy.ps1" -ForegroundColor White
Write-Host ("  2. Or use AWS CLI: aws lambda update-function-code --function-name llm-backend --zip-file fileb://{0}" -f $OutputFile) -ForegroundColor White
Write-Host ""
