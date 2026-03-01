# =============================================================================
# Flush MLflow Lite S3 Spool
# Replays queued MLflow events from S3 via the Lambda API endpoint.
# =============================================================================

param(
    [string]$Endpoint = "",
    [int]$MaxItems = 500,
    [string]$Token = "",
    [int]$Rounds = 1
)

$ErrorActionPreference = "Stop"
$Region = "us-east-1"
$ApiName = "llm-backend-api"

if ([string]::IsNullOrWhiteSpace($Endpoint)) {
    $Apis = aws apigatewayv2 get-apis --region $Region | ConvertFrom-Json
    $Api = $Apis.Items | Where-Object { $_.Name -eq $ApiName } | Select-Object -First 1
    if (-not $Api) {
        Write-Host "ERROR: Could not resolve API endpoint (name=$ApiName)." -ForegroundColor Red
        exit 1
    }
    $Endpoint = $Api.ApiEndpoint
}

if ($MaxItems -lt 1) { $MaxItems = 1 }
if ($MaxItems -gt 5000) { $MaxItems = 5000 }
if ($Rounds -lt 1) { $Rounds = 1 }

$Headers = @{
    "Content-Type" = "application/json"
}
if (-not [string]::IsNullOrWhiteSpace($Token)) {
    $Headers["x-mlflow-spool-token"] = $Token
}

Write-Host "Endpoint : $Endpoint" -ForegroundColor Cyan
Write-Host "MaxItems : $MaxItems" -ForegroundColor Cyan
Write-Host "Rounds   : $Rounds" -ForegroundColor Cyan
Write-Host ""

for ($i = 1; $i -le $Rounds; $i++) {
    $Body = @{ max_items = $MaxItems } | ConvertTo-Json
    try {
        $Resp = Invoke-RestMethod `
            -Method POST `
            -Uri "$Endpoint/ai/mlflow/flush-spool" `
            -Headers $Headers `
            -Body $Body `
            -TimeoutSec 120

        Write-Host ("Round {0}: processed={1} succeeded={2} failed={3}" -f `
            $i, $Resp.processed, $Resp.succeeded, $Resp.failed) -ForegroundColor Green

        if ([int]$Resp.processed -eq 0) {
            Write-Host "No queued items left to flush." -ForegroundColor Yellow
            break
        }
    } catch {
        Write-Host "Flush failed: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Cyan
