# =============================================================================
# Deploy Lambda Backend - Complete Deployment Script
# Creates IAM role, Lambda function, and API Gateway
# =============================================================================

param(
    [switch]$SkipPackage,
    [switch]$UpdateOnly
)

$ErrorActionPreference = "Stop"
$Region = "us-east-1"
$AccountId = "<AWS_ACCOUNT_ID>"
$FunctionName = "llm-backend"
$RoleName = "lambda-llm-backend-role"

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  Lambda Backend Deployment" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $PSCommandPath
$LambdaDir = Split-Path -Parent $ScriptDir
$DeploymentDir = Join-Path $LambdaDir "deployment"

# Check AWS CLI
Write-Host "Checking prerequisites..." -ForegroundColor Yellow
$AwsVersion = aws --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ AWS CLI not found. Please install AWS CLI." -ForegroundColor Red
    Write-Host "   Download: https://aws.amazon.com/cli/" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ AWS CLI: $AwsVersion" -ForegroundColor Green

# Verify AWS credentials
try {
    $Identity = aws sts get-caller-identity 2>&1 | ConvertFrom-Json
    if ($Identity.Account -ne $AccountId) {
        Write-Host "⚠️  Warning: AWS Account ID mismatch!" -ForegroundColor Yellow
        Write-Host "   Expected: $AccountId" -ForegroundColor Yellow
        Write-Host "   Found: $($Identity.Account)" -ForegroundColor Yellow
        Write-Host "   Update `$AccountId in deploy.ps1 or switch AWS profile" -ForegroundColor Yellow
        Write-Host ""
    } else {
        Write-Host "✅ AWS Account: $AccountId ($($Identity.Arn))" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Failed to verify AWS credentials" -ForegroundColor Red
    Write-Host "   Run: aws configure" -ForegroundColor Yellow
    exit 1
}

# Verify Python version
$PythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found. Please install Python 3.13+" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Python: $PythonVersion" -ForegroundColor Green

# Verify pip supports --platform flag (pip >= 20.3)
$PipVersion = python -m pip --version 2>&1
if ($PipVersion -match "pip (\d+)\.(\d+)") {
    $Major = [int]$matches[1]
    $Minor = [int]$matches[2]
    if ($Major -lt 20 -or ($Major -eq 20 -and $Minor -lt 3)) {
        Write-Host "⚠️  Warning: pip $Major.$Minor detected. Recommend pip >= 20.3 for --platform support" -ForegroundColor Yellow
        Write-Host "   Upgrade: python -m pip install --upgrade pip" -ForegroundColor Yellow
    } else {
        Write-Host "✅ pip: $PipVersion" -ForegroundColor Green
    }
}

# Load environment variables if .env exists
$EnvFile = Join-Path $LambdaDir ".env"
if (Test-Path $EnvFile) {
    Write-Host "Loading environment variables from .env..." -ForegroundColor Yellow
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            if ($name -and $value -and -not $name.StartsWith('#')) {
                [Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
    }
    Write-Host "✅ Environment variables loaded" -ForegroundColor Green
} else {
    Write-Host "⚠️  No .env file found. Using default values." -ForegroundColor Yellow
    Write-Host "   Copy .env.example to .env and configure before deployment." -ForegroundColor Yellow
}

# Step 1: Create deployment package
if (-not $SkipPackage) {
    Write-Host ""
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "Step 1: Creating Deployment Package" -ForegroundColor Cyan
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
    & (Join-Path $ScriptDir "create-deployment-package.ps1")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create deployment package" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⏭️  Skipping package creation" -ForegroundColor Yellow
}

$DeploymentPackage = Join-Path $LambdaDir "deployment-package.zip"
if (-not (Test-Path $DeploymentPackage)) {
    Write-Host "❌ Deployment package not found: $DeploymentPackage" -ForegroundColor Red
    exit 1
}

# Check package size
$PackageSize = (Get-Item $DeploymentPackage).Length
$PackageSizeMB = [math]::Round($PackageSize / 1MB, 2)

if ($PackageSizeMB -gt 50) {
    Write-Host "⚠️  Package is $PackageSizeMB MB, will use S3 for upload" -ForegroundColor Yellow
    $UseS3 = $true
    
    # Create S3 bucket if needed
    $BucketName = "lambda-deployments-$AccountId"
    $BucketExists = aws s3 ls "s3://$BucketName" --region $Region 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Creating S3 bucket: $BucketName" -ForegroundColor Yellow
        if ($Region -eq "us-east-1") {
            aws s3 mb "s3://$BucketName" --region $Region | Out-Null
        } else {
            aws s3 mb "s3://$BucketName" --region $Region --create-bucket-configuration LocationConstraint=$Region | Out-Null
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ S3 bucket created" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to create S3 bucket" -ForegroundColor Red
            exit 1
        }
    }
    
    # Upload to S3
    $S3Key = "lambda-backend/deployment-package-$(Get-Date -Format 'yyyyMMdd-HHmmss').zip"
    Write-Host "Uploading package to S3..." -ForegroundColor Yellow
    aws s3 cp $DeploymentPackage "s3://$BucketName/$S3Key" --region $Region
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Package uploaded to S3" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to upload to S3" -ForegroundColor Red
        exit 1
    }
} else {
    $UseS3 = $false
}

if ($UpdateOnly) {
    Write-Host ""
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "Updating Lambda Function Code" -ForegroundColor Cyan
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
    
    aws lambda update-function-code `
        --function-name $FunctionName `
        --zip-file "fileb://$DeploymentPackage" `
        --region $Region | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Lambda function code updated" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to update Lambda function" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "✅ Deployment complete!" -ForegroundColor Green
    exit 0
}

# Step 2: Create IAM Role (if not exists)
Write-Host ""
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "Step 2: Setting up IAM Role" -ForegroundColor Cyan
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan

$RoleExists = aws iam get-role --role-name $RoleName --region $Region 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating IAM role: $RoleName" -ForegroundColor Yellow
    
    aws iam create-role `
        --role-name $RoleName `
        --assume-role-policy-document "file://$DeploymentDir/trust-policy.json" `
        --region $Region | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ IAM role created" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create IAM role" -ForegroundColor Red
        exit 1
    }
    
    # Wait for role to be available
    Write-Host "Waiting for IAM role to propagate..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
} else {
    Write-Host "✅ IAM role already exists" -ForegroundColor Green
}

# Attach inline policy
Write-Host "Attaching IAM policy..." -ForegroundColor Yellow
aws iam put-role-policy `
    --role-name $RoleName `
    --policy-name "bedrock-cloudwatch-access" `
    --policy-document "file://$DeploymentDir/iam-policy.json" `
    --region $Region | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ IAM policy attached" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to attach IAM policy" -ForegroundColor Red
    exit 1
}

$RoleArn = "arn:aws:iam::${AccountId}:role/$RoleName"

# Step 3: Create or Update Lambda Function
Write-Host ""
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "Step 3: Deploying Lambda Function" -ForegroundColor Cyan
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan

$FunctionExists = aws lambda get-function --function-name $FunctionName --region $Region 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating Lambda function: $FunctionName" -ForegroundColor Yellow
    
    # Prepare environment variables
    $EnvVars = @{}
    
    # Add environment variables, removing any surrounding quotes from values
    # Note: AWS_REGION is automatically provided by Lambda runtime, don't set it
    $Keys = @("BEDROCK_MODEL_ID", "BEDROCK_MAX_TOKENS", "BEDROCK_TEMPERATURE", 
              "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "CORS_ORIGINS", 
              "LOG_LEVEL", "APP_NAME", "APP_VERSION")
    
    foreach ($key in $Keys) {
        $value = [Environment]::GetEnvironmentVariable($key, "Process")
        if ($value) {
            # Remove surrounding quotes if present
            $value = $value.Trim('"').Trim("'")
            $EnvVars[$key] = $value
        }
    }
    
    # Set defaults for missing values
    if (-not $EnvVars["BEDROCK_MODEL_ID"]) { $EnvVars["BEDROCK_MODEL_ID"] = "us.anthropic.claude-haiku-4-5-20251001-v1:0" }
    if (-not $EnvVars["BEDROCK_MAX_TOKENS"]) { $EnvVars["BEDROCK_MAX_TOKENS"] = "1024" }
    if (-not $EnvVars["BEDROCK_TEMPERATURE"]) { $EnvVars["BEDROCK_TEMPERATURE"] = "0.7" }
    if (-not $EnvVars["SUPABASE_URL"]) { $EnvVars["SUPABASE_URL"] = "" }
    if (-not $EnvVars["SUPABASE_SERVICE_ROLE_KEY"]) { $EnvVars["SUPABASE_SERVICE_ROLE_KEY"] = "" }
    if (-not $EnvVars["CORS_ORIGINS"]) { $EnvVars["CORS_ORIGINS"] = "http://localhost:3000" }
    if (-not $EnvVars["LOG_LEVEL"]) { $EnvVars["LOG_LEVEL"] = "INFO" }
    if (-not $EnvVars["APP_NAME"]) { $EnvVars["APP_NAME"] = "LLMBackend-Lambda" }
    if (-not $EnvVars["APP_VERSION"]) { $EnvVars["APP_VERSION"] = "1.0.0" }
    
    # Convert to JSON string manually to avoid escaping issues
    $EnvJson = "{"
    $first = $true
    foreach ($key in $EnvVars.Keys) {
        if (-not $first) { $EnvJson += "," }
        $EnvJson += "`"$key`":`"$($EnvVars[$key])`""
        $first = $false
    }
    $EnvJson += "}"
    
    if ($UseS3) {
        aws lambda create-function `
            --function-name $FunctionName `
            --runtime python3.13 `
            --role $RoleArn `
            --handler lambda_handler.lambda_handler `
            --code "S3Bucket=$BucketName,S3Key=$S3Key" `
            --timeout 30 `
            --memory-size 512 `
            --environment "{`"Variables`":$EnvJson}" `
            --description "FastAPI backend for JobLab AI assistant" `
            --region $Region | Out-Null
    } else {
        aws lambda create-function `
            --function-name $FunctionName `
            --runtime python3.13 `
            --role $RoleArn `
            --handler lambda_handler.lambda_handler `
            --zip-file "fileb://$DeploymentPackage" `
            --timeout 30 `
            --memory-size 512 `
            --environment "{`"Variables`":$EnvJson}" `
            --description "FastAPI backend for JobLab AI assistant" `
            --region $Region | Out-Null
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Lambda function created" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create Lambda function" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ Lambda function already exists, updating code..." -ForegroundColor Yellow
    
    if ($UseS3) {
        aws lambda update-function-code `
            --function-name $FunctionName `
            --s3-bucket $BucketName `
            --s3-key $S3Key `
            --region $Region | Out-Null
    } else {
        aws lambda update-function-code `
            --function-name $FunctionName `
            --zip-file "fileb://$DeploymentPackage" `
            --region $Region | Out-Null
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Lambda function code updated" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to update Lambda function" -ForegroundColor Red
        exit 1
    }
}

# Step 4: Create API Gateway (HTTP API)
Write-Host ""
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "Step 4: Setting up API Gateway" -ForegroundColor Cyan
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Cyan

$ApiName = "llm-backend-api"
$LambdaArn = "arn:aws:lambda:${Region}:${AccountId}:function/$FunctionName"

# Check if API exists
$ExistingApis = aws apigatewayv2 get-apis --region $Region | ConvertFrom-Json
$ExistingApi = $ExistingApis.Items | Where-Object { $_.Name -eq $ApiName } | Select-Object -First 1

if ($ExistingApi) {
    $ApiId = $ExistingApi.ApiId
    $ApiEndpoint = $ExistingApi.ApiEndpoint
    Write-Host "✅ API Gateway already exists: $ApiId" -ForegroundColor Green
} else {
    Write-Host "Creating HTTP API..." -ForegroundColor Yellow
    
    # Create API without target (will add integration separately)
    $ApiResponse = aws apigatewayv2 create-api `
        --name $ApiName `
        --protocol-type HTTP `
        --region $Region | ConvertFrom-Json
    
    if ($LASTEXITCODE -eq 0) {
        $ApiId = $ApiResponse.ApiId
        $ApiEndpoint = $ApiResponse.ApiEndpoint
        Write-Host "✅ API Gateway created: $ApiId" -ForegroundColor Green
        
        # Create Lambda integration
        Write-Host "Creating Lambda integration..." -ForegroundColor Yellow
        $IntegrationResponse = aws apigatewayv2 create-integration `
            --api-id $ApiId `
            --integration-type AWS_PROXY `
            --integration-uri $LambdaArn `
            --payload-format-version "2.0" `
            --region $Region | ConvertFrom-Json
        
        $IntegrationId = $IntegrationResponse.IntegrationId
        Write-Host "✅ Integration created: $IntegrationId" -ForegroundColor Green
        
        # Create default route
        Write-Host "Creating default route..." -ForegroundColor Yellow
        aws apigatewayv2 create-route `
            --api-id $ApiId `
            --route-key '$default' `
            --target "integrations/$IntegrationId" `
            --region $Region | Out-Null
        
        Write-Host "✅ Routes configured" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create API Gateway" -ForegroundColor Red
        exit 1
    }
}

# Add Lambda permission for API Gateway
Write-Host "Configuring API Gateway permissions..." -ForegroundColor Yellow
$StatementId = "apigateway-invoke-$ApiId"

aws lambda remove-permission `
    --function-name $FunctionName `
    --statement-id $StatementId `
    --region $Region 2>&1 | Out-Null

aws lambda add-permission `
    --function-name $FunctionName `
    --statement-id $StatementId `
    --action lambda:InvokeFunction `
    --principal apigateway.amazonaws.com `
    --source-arn "arn:aws:execute-api:${Region}:${AccountId}:${ApiId}/*/*" `
    --region $Region | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ API Gateway permissions configured" -ForegroundColor Green
} else {
    Write-Host "⚠️  Permission may already exist" -ForegroundColor Yellow
}

# Configure CORS
Write-Host "Configuring CORS..." -ForegroundColor Yellow
$CorsOrigins = ($env:CORS_ORIGINS ?? "http://localhost:3000") -split ','
$CorsOriginsJson = $CorsOrigins | ConvertTo-Json -Compress

aws apigatewayv2 update-api `
    --api-id $ApiId `
    --cors-configuration "AllowOrigins=$CorsOriginsJson,AllowMethods=['GET','POST','OPTIONS'],AllowHeaders=['content-type','authorization'],MaxAge=300" `
    --region $Region | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ CORS configured" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "  ✅ Deployment Complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "API Gateway Endpoint:" -ForegroundColor Cyan
Write-Host "  $ApiEndpoint" -ForegroundColor White
Write-Host ""
Write-Host "Available Endpoints:" -ForegroundColor Cyan
Write-Host "  GET  ${ApiEndpoint}/health" -ForegroundColor White
Write-Host "  POST ${ApiEndpoint}/ai/ask" -ForegroundColor White
Write-Host ""
Write-Host "Test your deployment:" -ForegroundColor Cyan
Write-Host "  .\scripts\test-endpoint.ps1 -Endpoint $ApiEndpoint" -ForegroundColor White
Write-Host ""
Write-Host "View logs:" -ForegroundColor Cyan
Write-Host "  aws logs tail /aws/lambda/$FunctionName --follow --region $Region" -ForegroundColor White
Write-Host ""
Write-Host "Update frontend endpoint to:" -ForegroundColor Cyan
Write-Host "  $ApiEndpoint" -ForegroundColor White
Write-Host ""
