# 🛠️ Lambda Deployment Scripts - Complete Guide

This folder contains production-ready PowerShell scripts for deploying and managing your AWS Lambda backend.

---

## 📁 Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| **deploy.ps1** | Full deployment (IAM, Lambda, API Gateway) | Initial setup or infrastructure changes |
| **update-lambda.ps1** | Quick code updates | After changing application code |
| **rollback-lambda.ps1** | Restore previous version | When deployment causes issues |
| **create-deployment-package.ps1** | Build deployment package | Manual package creation |
| **update-environment.ps1** | Update environment variables | Change config without redeploying |
| **test-endpoint.ps1** | Test all endpoints | Verify deployment health |

---

## 🚀 Quick Start

### First-Time Deployment

```powershell
# 1. Configure environment
cd lambda_backend
copy .env.example .env
# Edit .env with your Supabase credentials

# 2. Deploy everything
.\scripts\deploy.ps1
```

**What it does:**
- ✅ Creates IAM role with Bedrock + CloudWatch permissions
- ✅ Builds deployment package with Linux-compatible wheels
- ✅ Creates Lambda function (512MB, 30s timeout)
- ✅ Sets up API Gateway HTTP API with CORS
- ✅ Runs health check tests

**Time:** ~3-5 minutes

---

### Updating Code

```powershell
# Make your changes in app/

# Quick update (recommended)
.\scripts\update-lambda.ps1

# Or use deploy.ps1 in update mode
.\scripts\deploy.ps1 -UpdateOnly
```

**What it does:**
- ✅ Rebuilds package with Linux wheels
- ✅ Uploads to Lambda (via S3 if > 50MB)
- ✅ Runs health check
- ✅ Shows deployment status

**Time:** ~30-60 seconds

---

### Rolling Back

```powershell
# List available backups
.\scripts\rollback-lambda.ps1 -ListBackups

# Rollback to specific package
.\scripts\rollback-lambda.ps1 -PackageFile "deployment-package.zip"
```

**When to use:**
- ❌ New deployment causes errors
- ❌ Performance regression detected
- ❌ Need to quickly restore working version

**Time:** ~10-20 seconds

---

## 📋 Script Details

### deploy.ps1

**Full deployment with all AWS resources.**

```powershell
# Full deployment
.\scripts\deploy.ps1

# Skip package creation (use existing .zip)
.\scripts\deploy.ps1 -SkipPackage

# Update Lambda code only (no IAM/API Gateway changes)
.\scripts\deploy.ps1 -UpdateOnly
```

**Features:**
- Validates AWS credentials and Python version
- Checks pip supports `--platform` flag (>= 20.3)
- Creates IAM role if doesn't exist
- Attaches updated policies
- Handles S3 upload for large packages (> 50MB)
- Creates API Gateway with `$default` stage
- Configures CORS
- Sets environment variables (excludes `AWS_REGION`)

**Prerequisites:**
- AWS CLI installed and configured
- Python 3.13+ with pip >= 20.3
- `.env` file with Supabase credentials

---

### update-lambda.ps1

**Fast code updates without touching IAM or API Gateway.**

```powershell
# Standard update (rebuild + deploy + test)
.\scripts\update-lambda.ps1

# Use existing package
.\scripts\update-lambda.ps1 -SkipPackage -PackageFile "existing.zip"

# Deploy without testing
.\scripts\update-lambda.ps1 -SkipTest
```

**Workflow:**
1. Rebuilds package with Linux wheels
2. Checks package size (uses S3 if > 50MB)
3. Updates Lambda function code
4. Waits for activation
5. Runs health check test

**Use this for:**
- ✅ Code changes in `app/`
- ✅ Dependency updates in `requirements.txt`
- ✅ Bug fixes and features

**Time saved:** 10x faster than full deploy.ps1

---

### rollback-lambda.ps1

**Quickly restore a previous deployment.**

```powershell
# See what backups are available
.\scripts\rollback-lambda.ps1 -ListBackups

# Rollback to local file
.\scripts\rollback-lambda.ps1 -PackageFile "deployment-linux.zip"

# Rollback to S3 backup
.\scripts\rollback-lambda.ps1 -PackageFile "s3://bucket/lambda-backend/deployment-20260212.zip"
```

**Backup options:**
- **Local:** Any `.zip` file in `lambda_backend/`
- **S3:** Previous uploads in `s3://lambda-deployments-{account-id}/lambda-backend/`

**Best practice:**
Keep last 3 working packages for quick rollback.

---

### create-deployment-package.ps1

**Builds deployment package with all optimizations.**

```powershell
# Standard package
.\scripts\create-deployment-package.ps1

# Custom output name
.\scripts\create-deployment-package.ps1 -OutputFile "backup-20260212.zip"
```

**Optimizations applied:**
- ✅ Linux-compatible wheels (`--platform manylinux2014_x86_64`)
- ✅ Binary-only packages (`--only-binary=:all:`)
- ✅ Removes `__pycache__`, `*.pyc`, `*.pyo`
- ✅ Removes test files and `.dist-info`
- ✅ Checks for module shadowing conflicts
- ✅ Removes root-level `typing.py`, `http.py`, etc.

**Result:** ~21 MB package (vs 76 MB unoptimized)

---

### update-environment.ps1

**Update Lambda environment variables without redeploying code.**

```powershell
# Update all variables from .env
.\scripts\update-environment.ps1

# Update specific variable
.\scripts\update-environment.ps1 -Key "BEDROCK_MODEL_ID" -Value "us.anthropic.claude-haiku-4-5-20251001-v1:0"
```

**Common use cases:**
- Change Bedrock model ID
- Update Supabase URL/key
- Adjust CORS origins
- Change log level

**Note:** Lambda runtime provides `AWS_REGION` automatically.

---

### test-endpoint.ps1

**Comprehensive endpoint testing.**

```powershell
# Test default endpoint (from API Gateway)
.\scripts\test-endpoint.ps1

# Test specific endpoint
.\scripts\test-endpoint.ps1 -Endpoint "https://abc123.execute-api.us-east-1.amazonaws.com"
```

**Tests performed:**
1. ✅ Health check (`GET /health`)
2. ✅ Simple AI query (no database)
3. ✅ Database query with tool calling

**Expected results:**
- Response time < 2s (cold) / < 1s (warm)
- All tests pass with 200 status
- Bedrock model responds correctly

---

## 🔧 Common Workflows

### First Deployment

```powershell
# 1. Setup
cd lambda_backend
copy .env.example .env
notepad .env  # Add Supabase credentials

# 2. Deploy
.\scripts\deploy.ps1

# 3. Test
.\scripts\test-endpoint.ps1
```

---

### Daily Development Cycle

```powershell
# 1. Make code changes
# Edit files in app/

# 2. Quick deploy
.\scripts\update-lambda.ps1

# 3. Verify
# Check health automatically runs
# Or run full tests: .\scripts\test-endpoint.ps1
```

---

### Fixing Issues

```powershell
# 1. Something broke, rollback ASAP
.\scripts\rollback-lambda.ps1 -ListBackups
.\scripts\rollback-lambda.ps1 -PackageFile "deployment-linux.zip"

# 2. Verify rollback worked
.\scripts\test-endpoint.ps1

# 3. Fix code, redeploy
# Fix the bugs...
.\scripts\update-lambda.ps1

# 4. Monitor logs
aws logs tail /aws/lambda/llm-backend --follow
```

---

### Changing Configuration

```powershell
# Update environment variable
.\scripts\update-environment.ps1 -Key "BEDROCK_MODEL_ID" -Value "anthropic.claude-3-sonnet-20240229-v1:0"

# Or update .env and re-run
notepad .env
.\scripts\deploy.ps1 -UpdateOnly
```

---

### Performance Tuning

```powershell
# Increase memory (also increases CPU)
aws lambda update-function-configuration \
  --function-name llm-backend \
  --memory-size 1024 \
  --region us-east-1

# Increase timeout
aws lambda update-function-configuration \
  --function-name llm-backend \
  --timeout 60 \
  --region us-east-1

# Test after changes
.\scripts\test-endpoint.ps1
```

---

## ⚠️ Critical Deployment Rules

### 1. Always Use Linux Wheels on Windows

```powershell
# ✅ Correct
python -m pip install -r requirements.txt -t package \
  --platform manylinux2014_x86_64 \
  --only-binary=:all:

# ❌ Wrong (creates Windows binaries)
python -m pip install -r requirements.txt -t package
```

**Why:** Lambda runs Amazon Linux 2, not Windows. Windows wheels cause:
```
ImportError: No module named 'pydantic_core._pydantic_core'
```

**Fixed in:** `create-deployment-package.ps1` (automatic)

---

### 2. Never Set AWS_REGION Manually

```powershell
# ❌ Don't do this
$EnvVars = @{
    "AWS_REGION" = "us-east-1"  # Lambda provides this!
}

# ✅ Lambda provides these automatically:
# - AWS_REGION
# - AWS_LAMBDA_FUNCTION_NAME
# - AWS_LAMBDA_FUNCTION_MEMORY_SIZE
```

**Fixed in:** `deploy.ps1` (filters reserved vars)

---

### 3. Remove Module Shadowing Files

```powershell
# Files that MUST NOT exist at package root:
- typing.py
- http.py
- types.py
- abc.py

# These are OK in subdirectories:
- botocore/typing.py ✅
- package/submodule/http.py ✅
```

**Fixed in:** `create-deployment-package.ps1` (automatic)

---

### 4. Include Inference Profiles in IAM

```json
{
  "Resource": [
    "arn:aws:bedrock:*:*:foundation-model/*",
    "arn:aws:bedrock:*:*:inference-profile/*"  // ← Required!
  ]
}
```

**Fixed in:** `deployment/iam-policy.json`

---

## 📊 Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'pydantic_core'` | Windows wheels on Lambda | Rebuild with `--platform manylinux2014_x86_64` |
| `cannot import TYPE_CHECKING` | Module shadowing | Remove `package/typing.py` |
| `404 Not Found` on all endpoints | Wrong API stage | Use `$default` stage, not `prod` |
| `AccessDeniedException` Bedrock | Missing IAM permission | Add `inference-profile/*` to policy |
| Package upload fails | Size > 50MB | Use S3 upload (automatic in scripts) |
| Function timeout | Slow Bedrock response | Increase timeout to 60s |

See [DEPLOYMENT_LESSONS.md](../DEPLOYMENT_LESSONS.md) for detailed troubleshooting.

---

## 📝 Best Practices

### Keep Backups

```powershell
# Before major changes, save current package
copy deployment-package.zip deployment-backup-$(Get-Date -Format 'yyyyMMdd').zip

# S3 auto-saves with timestamps
.\scripts\deploy.ps1  # Creates s3://.../deployment-20260212-143045.zip
```

---

### Test Before Deploying

```powershell
# Test locally first
cd ../LLMBackend
uvicorn app.main:app --reload

# Then deploy to Lambda
cd ../lambda_backend
.\scripts\update-lambda.ps1
```

---

### Monitor After Deployment

```powershell
# Watch logs live
aws logs tail /aws/lambda/llm-backend --follow --region us-east-1

# Check recent errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/llm-backend \
  --filter-pattern "ERROR" \
  --start-time $(Get-Date).AddHours(-1).ToFileTimeUtc()
```

---

### Version Your Packages

```powershell
# Name packages with versions
.\scripts\create-deployment-package.ps1 -OutputFile "deployment-v1.2.0.zip"

# Keep last 3 versions
deployment-v1.2.0.zip  # Current
deployment-v1.1.0.zip  # Previous
deployment-v1.0.0.zip  # Stable
```

---

## 🔗 Related Documentation

- **[DEPLOYMENT_LESSONS.md](../DEPLOYMENT_LESSONS.md)** - Critical deployment insights
- **[DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)** - Step-by-step instructions
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - Technical architecture
- **[README.md](../README.md)** - Project overview

---

## 💡 Tips & Tricks

### Speed Up Deployments

```powershell
# Skip package rebuild if code unchanged
.\scripts\update-lambda.ps1 -SkipPackage

# Skip tests for faster deployment
.\scripts\update-lambda.ps1 -SkipTest
```

---

### Parallel Development

```powershell
# Terminal 1: Code and deploy
.\scripts\update-lambda.ps1
aws logs tail /aws/lambda/llm-backend --follow

# Terminal 2: Test endpoints
while ($true) { 
    .\scripts\test-endpoint.ps1
    Start-Sleep -Seconds 30
}
```

---

### Check What Changed

```powershell
# See current function config
aws lambda get-function-configuration \
  --function-name llm-backend \
  --region us-east-1 \
  | ConvertFrom-Json \
  | Format-List

# Compare package sizes
Get-ChildItem *.zip | Select Name, @{N='SizeMB';E={[math]::Round($_.Length/1MB,2)}} | Sort LastWriteTime -Descending
```

---

**Last Updated:** 2026-02-12  
**Scripts Version:** 2.0 (Production Ready)
