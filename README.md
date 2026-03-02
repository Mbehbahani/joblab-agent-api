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
