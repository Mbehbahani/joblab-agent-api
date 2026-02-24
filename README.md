# Lambda Backend - FastAPI on AWS Lambda

This is a serverless deployment of the LLMBackend FastAPI application on AWS Lambda + API Gateway.

## Architecture

```
┌─────────────────┐
│  API Gateway    │ ← HTTP API (Public endpoint)
│   (HTTP API)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AWS Lambda     │ ← FastAPI + Mangum
│   (Python 3.13) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  AWS Bedrock (Claude)           │
│  Supabase (via HTTP)            │
└─────────────────────────────────┘
```

## Benefits over ECS

- **Cost**: ~$0.20/month for 1M requests (vs ECS ~$15/month minimum)
- **Scaling**: Automatic from 0 to thousands of concurrent executions
- **Maintenance**: No container orchestration, no task definitions
- **Cold starts**: ~1-2 seconds (acceptable for this workload)

## Folder Structure

```
lambda_backend/
├── lambda_handler.py          # Lambda entry point with Mangum
├── requirements.txt           # Optimized dependencies
├── .env.example              # Environment variable template
├── README.md                 # This file
├── app/                      # FastAPI application (copied from LLMBackend)
│   ├── __init__.py
│   ├── main.py              # FastAPI app instance
│   ├── config.py            # Environment-based settings
│   ├── routers/
│   │   ├── ai.py           # AI endpoints with Bedrock + tools
│   │   └── health.py       # Health check endpoint
│   ├── services/
│   │   ├── bedrock.py      # Bedrock service wrapper
│   │   └── joblab_tools.py # Tool definitions and executors
│   └── schemas/
│       ├── ai.py           # Request/response models
│       └── tools.py        # Tool schemas
├── deployment/
│   ├── iam-policy.json     # IAM policy for Lambda execution role
│   ├── lambda-config.json  # Lambda function configuration
│   └── api-gateway.json    # API Gateway configuration
└── scripts/
    ├── deploy.ps1          # Main deployment script
    ├── create-deployment-package.ps1  # Package Lambda code
    ├── update-environment.ps1         # Update environment variables
    └── test-endpoint.ps1              # Test API Gateway endpoint
```

## Prerequisites

- AWS CLI configured with credentials
- PowerShell 7+ (or bash for Linux/Mac)
- Python 3.13
- IAM permissions for Lambda, API Gateway, IAM roles

## Environment Variables

Create a `.env` file with:

```bash
# AWS / Bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0
BEDROCK_MAX_TOKENS=1024
BEDROCK_TEMPERATURE=0.7

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# CORS
CORS_ORIGINS=https://your-frontend.vercel.app,http://localhost:3000
```

## Deployment

### Quick Deploy

```powershell
cd lambda_backend
.\scripts\deploy.ps1
```

### Manual Deployment

1. **Create deployment package**:
   ```powershell
   .\scripts\create-deployment-package.ps1
   ```

2. **Create IAM role** (if not exists):
   ```powershell
   aws iam create-role --role-name lambda-llm-backend-role --assume-role-policy-document file://deployment/trust-policy.json
   aws iam put-role-policy --role-name lambda-llm-backend-role --policy-name bedrock-access --policy-document file://deployment/iam-policy.json
   ```

3. **Create Lambda function**:
   ```powershell
   aws lambda create-function \
     --function-name llm-backend \
     --runtime python3.13 \
     --role arn:aws:iam::<AWS_ACCOUNT_ID>:role/lambda-llm-backend-role \
     --handler lambda_handler.lambda_handler \
     --zip-file fileb://deployment-package.zip \
     --timeout 30 \
     --memory-size 512 \
     --environment Variables="{AWS_REGION=us-east-1,BEDROCK_MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0,...}"
   ```

4. **Create API Gateway**:
   ```powershell
   aws apigatewayv2 create-api --name llm-backend-api --protocol-type HTTP --target arn:aws:lambda:us-east-1:<AWS_ACCOUNT_ID>:function:llm-backend
   ```

## API Endpoints

After deployment, your API Gateway will provide a URL like:

```
https://abc123.execute-api.us-east-1.amazonaws.com
```

### Available Endpoints

- `GET /health` - Health check
- `POST /ai/ask` - AI query endpoint with tool calling

### Example Request

```bash
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/ai/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How many jobs were posted in January 2026?"}'
```

## Monitoring

View logs in CloudWatch:

```powershell
aws logs tail /aws/lambda/llm-backend --follow
```

## Cost Estimation

For 100,000 requests/month with 2s average duration:
- Lambda compute: ~$0.17
- API Gateway: ~$0.10
- Total: ~$0.27/month

Compare to ECS Fargate: ~$15-30/month minimum

## Troubleshooting

### Cold Starts
- First request after idle: 1-2 seconds
- Solution: Use provisioned concurrency (adds cost) or accept occasional cold start

### Timeout
- Default: 30 seconds
- Increase if needed: `aws lambda update-function-configuration --function-name llm-backend --timeout 60`

### Memory Issues
- Default: 512 MB
- Increase if needed: `aws lambda update-function-configuration --function-name llm-backend --memory-size 1024`

## Updating Frontend

Update your frontend's API endpoint from:
```
https://your-alb.us-east-1.elb.amazonaws.com
```

To:
```
https://your-api-id.execute-api.us-east-1.amazonaws.com
```

## Rollback to ECS

All ECS deployment files remain in `LLMBackend/` folder unchanged. To rollback:

1. Update frontend endpoint back to ALB
2. Ensure ECS service is running
3. Optionally delete Lambda function and API Gateway
