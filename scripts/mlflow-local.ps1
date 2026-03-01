# =============================================================================
# Run MLflow UI locally — connects to the same Postgres DB and S3 bucket
# that your Railway production server uses.
#
# Usage:
#   cd lambda_backend
#   .\scripts\mlflow-local.ps1
#
# Then open: http://localhost:5001
# =============================================================================

$env:AWS_ACCESS_KEY_ID     = "AKIAREDACTEDREDACTED"
$env:AWS_SECRET_ACCESS_KEY = "REDACTED_AWS_SECRET_ACCESS_KEY"
$env:AWS_DEFAULT_REGION    = "eu-west-1"

$BACKEND = "postgresql://postgres:REDACTED_DB_PASSWORD@switchback.proxy.rlwy.net:35369/railway"
$ARTIFACTS = "s3://mlflow-artifacts-780822965578-eu"

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  MLflow UI — local server (same DB + S3 as Railway)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  Backend : $BACKEND" -ForegroundColor Gray
Write-Host "  Artifacts: $ARTIFACTS" -ForegroundColor Gray
Write-Host "  Open    : http://localhost:5001" -ForegroundColor Green
Write-Host ""

mlflow server `
  --backend-store-uri  $BACKEND `
  --artifacts-destination $ARTIFACTS `
  --host 0.0.0.0 `
  --port 5001
