"""
AWS Lambda handler for FastAPI application using Mangum.
Exposes the FastAPI app as a Lambda function compatible with API Gateway.
"""

import logging
from mangum import Mangum
from app.main import app

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create the Lambda handler using Mangum adapter
# This wraps the FastAPI app and makes it compatible with AWS Lambda + API Gateway
handler = Mangum(app, lifespan="off")


def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    
    Parameters
    ----------
    event : dict
        API Gateway event object containing the HTTP request details
    context : LambdaContext
        AWS Lambda context object with runtime information
    
    Returns
    -------
    dict
        API Gateway-compatible response with statusCode, headers, and body
    """
    logger.info("Received event: %s", event.get("requestContext", {}).get("requestId", "unknown"))

    # Mangum handles the conversion between API Gateway and ASGI
    response = handler(event, context)

    # Flush MLflow async traces before Lambda freezes the execution environment.
    # Without this, buffered traces queued by async_logging=True would be lost
    # when the container is frozen/terminated.
    try:
        import mlflow
        mlflow.flush_async_logging()
    except ImportError:
        # Full mlflow not installed — flush the lightweight REST client instead
        try:
            from app.services.mlflow_lite import get_lite_client
            from app.config import get_settings
            lite = get_lite_client()
            if lite:
                lite.flush(timeout=3)
                auto_flush = max(0, int(get_settings().mlflow_spool_autoflush_max_items))
                if auto_flush > 0:
                    lite.flush_spool(max_items=auto_flush)
        except Exception:
            pass
    except Exception as exc:
        logger.warning("MLflow async flush failed (non-fatal): %s", exc)

    return response
