# =============================================================================
# Quick Lambda Update Script
# Use this for fast code updates after initial deployment
# =============================================================================

param(
    [switch]$SkipPackage,
    [switch]$SkipTest,
    [string]$PackageFile = "deployment-package.zip"
)

$ErrorActionPreference = "Stop"
$Region = "us-east-1"
$AccountId = "780822965578"
$FunctionName = "llm-backend"

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  Quick Lambda Code Update" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $PSCommandPath
$LambdaDir = Split-Path -Parent $ScriptDir

function Test-ZipContainsEntry {
    param(
        [string]$ZipPath,
        [string[]]$Patterns
    )

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($entry in $zip.Entries) {
            $entryPath = $entry.FullName -replace "\\", "/"
            foreach ($pattern in $Patterns) {
                if ($entryPath -like $pattern) {
                    return $true
                }
            }
        }
        return $false
    } finally {
        $zip.Dispose()
    }
}

# Step 1: Create deployment package (if needed)
if (-not $SkipPackage) {
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "Step 1/3: Creating Deployment Package" -ForegroundColor Cyan
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
    
    & (Join-Path $ScriptDir "create-deployment-package.ps1") -OutputFile $PackageFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create deployment package" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⏭️  Skipping package creation" -ForegroundColor Yellow
}

$DeploymentPackage = Join-Path $LambdaDir $PackageFile

# Validate package exists
if (-not (Test-Path $DeploymentPackage)) {
    Write-Host "❌ Deployment package not found: $DeploymentPackage" -ForegroundColor Red
    Write-Host "   Run without -SkipPackage to create it" -ForegroundColor Yellow
    exit 1
}

# Validate package freshness and required multipart dependency
$RequirementsFile = Join-Path $LambdaDir "requirements-lambda.txt"
if (-not (Test-Path $RequirementsFile)) {
    $RequirementsFile = Join-Path $LambdaDir "requirements.txt"
}
if ($SkipPackage -and (Test-Path $RequirementsFile)) {
    $ReqLastWrite = (Get-Item $RequirementsFile).LastWriteTimeUtc
    $PkgLastWrite = (Get-Item $DeploymentPackage).LastWriteTimeUtc
    if ($ReqLastWrite -gt $PkgLastWrite) {
        Write-Host "ERROR: requirements.txt is newer than $PackageFile." -ForegroundColor Red
        Write-Host "Rebuild package (run without -SkipPackage) before deploying." -ForegroundColor Yellow
        exit 1
    }
}

# /ai/match-cv uses request.form(), which requires python-multipart.
if (-not (Test-ZipContainsEntry -ZipPath $DeploymentPackage -Patterns @("multipart/*", "python_multipart/*"))) {
    Write-Host "ERROR: Deployment package is missing python-multipart." -ForegroundColor Red
    Write-Host "Add python-multipart to requirements.txt and rebuild package." -ForegroundColor Yellow
    exit 1
}

# Check package size
$PackageSize = (Get-Item $DeploymentPackage).Length
$PackageSizeMB = [math]::Round($PackageSize / 1MB, 2)

Write-Host ""
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "Step 2/3: Uploading Code to Lambda" -ForegroundColor Cyan
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "Package size: $PackageSizeMB MB" -ForegroundColor Gray
Write-Host ""

if ($PackageSizeMB -gt 50) {
    # Use S3 for large packages
    Write-Host "Package exceeds 50MB limit, uploading via S3..." -ForegroundColor Yellow
    
    $BucketName = "lambda-deployments-$AccountId"
    $S3Key = "lambda-backend/deployment-$(Get-Date -Format 'yyyyMMdd-HHmmss').zip"
    
    # Upload to S3
    Write-Host "Uploading to S3: s3://$BucketName/$S3Key" -ForegroundColor Gray
    aws s3 cp $DeploymentPackage "s3://$BucketName/$S3Key" --region $Region --quiet
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to upload to S3" -ForegroundColor Red
        Write-Host "   Ensure bucket exists: aws s3 mb s3://$BucketName --region $Region" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "✅ Uploaded to S3" -ForegroundColor Green
    
    # Update Lambda from S3
    Write-Host "Updating Lambda function from S3..." -ForegroundColor Yellow
    aws lambda update-function-code `
        --function-name $FunctionName `
        --s3-bucket $BucketName `
        --s3-key $S3Key `
        --region $Region `
        --output json | Out-Null
    
} else {
    # Direct upload for smaller packages
    Write-Host "Uploading directly to Lambda..." -ForegroundColor Yellow
    aws lambda update-function-code `
        --function-name $FunctionName `
        --zip-file "fileb://$DeploymentPackage" `
        --region $Region `
        --output json | Out-Null
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to update Lambda function code" -ForegroundColor Red
    Write-Host "   Check if function exists: aws lambda get-function --function-name $FunctionName --region $Region" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Lambda function code updated" -ForegroundColor Green

# Wait for function to become Active (with timeout)
Write-Host ""
Write-Host "Waiting for function to become Active..." -ForegroundColor Yellow
$WaitTimeout = 300  # seconds
$WaitStart = Get-Date
$PollInterval = 5

while ($true) {
    $Elapsed = ((Get-Date) - $WaitStart).TotalSeconds
    if ($Elapsed -gt $WaitTimeout) {
        Write-Host "❌ Timed out after ${WaitTimeout}s waiting for function to become Active" -ForegroundColor Red
        Write-Host "   Check function status: aws lambda get-function --function-name $FunctionName --region $Region" -ForegroundColor Yellow
        exit 1
    }

    $FunctionState = aws lambda get-function --function-name $FunctionName --region $Region | ConvertFrom-Json
    $State = $FunctionState.Configuration.State
    $LastUpdate = $FunctionState.Configuration.LastUpdateStatus

    if ($State -eq "Active" -and $LastUpdate -ne "InProgress") {
        Write-Host "✅ Function is Active (took $([math]::Round($Elapsed))s)" -ForegroundColor Green
        break
    }

    Write-Host "   State=$State  LastUpdateStatus=$LastUpdate  (${([math]::Round($Elapsed))}s elapsed)" -ForegroundColor Gray
    Start-Sleep -Seconds $PollInterval
}

# Step 3: Test endpoint (optional)
if (-not $SkipTest) {
    Write-Host ""
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "Step 3/3: Testing Endpoint" -ForegroundColor Cyan
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
    
    # Get API Gateway endpoint
    $ApiName = "llm-backend-api"
    $Apis = aws apigatewayv2 get-apis --region $Region | ConvertFrom-Json
    $Api = $Apis.Items | Where-Object { $_.Name -eq $ApiName } | Select-Object -First 1
    
    if ($Api) {
        $Endpoint = $Api.ApiEndpoint
        Write-Host "Endpoint: $Endpoint" -ForegroundColor Gray
        Write-Host ""
        
        # Quick health check
        Write-Host "Testing /health endpoint..." -ForegroundColor Yellow
        try {
            $Response = Invoke-RestMethod -Uri "$Endpoint/health" -Method Get -ErrorAction Stop
            if ($Response.status -eq "ok") {
                Write-Host "✅ Health check passed" -ForegroundColor Green
                Write-Host "   Status: $($Response.status)" -ForegroundColor Gray
                Write-Host "   Version: $($Response.version)" -ForegroundColor Gray
            } else {
                Write-Host "⚠️  Unexpected response: $($Response | ConvertTo-Json -Compress)" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "❌ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "   Check CloudWatch logs for details" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠️  API Gateway not found, skipping test" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "⏭️  Skipping endpoint test" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "  ✅ Lambda Update Complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  • Test your changes: .\scripts\test-endpoint.ps1" -ForegroundColor White
Write-Host "  • View logs: aws logs tail /aws/lambda/$FunctionName --follow" -ForegroundColor White
Write-Host "  • Rollback if needed: Deploy previous package.zip" -ForegroundColor White
Write-Host ""
