"""Deploy liquid-host as a custom container to HF Inference Endpoints.

Usage:
    python deploy/deploy_hf.py --image <docker-image-url>

Prerequisites:
    1. Authenticate: hf auth login
    2. Build & push the Docker image:
        docker build --platform linux/amd64 -t <your-registry>/liquid-host:latest .
        docker push <your-registry>/liquid-host:latest
    3. Run this script with the pushed image URL
"""

import argparse
import logging

from huggingface_hub import (
    create_inference_endpoint,
    get_inference_endpoint,
    whoami,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deploy")

# Defaults
DEFAULT_MODEL_REPO = "LiquidAI/LFM2-24B-A2B"
DEFAULT_ENDPOINT_NAME = "liquid-host-lfm"
DEFAULT_REGION = "us-east-1"
DEFAULT_VENDOR = "aws"
DEFAULT_ACCELERATOR = "gpu"
DEFAULT_INSTANCE_SIZE = "x1"
DEFAULT_INSTANCE_TYPE = "nvidia-l4"


def main():
    parser = argparse.ArgumentParser(description="Deploy liquid-host to HF Inference Endpoints")
    parser.add_argument("--image", required=True, help="Docker image URL (e.g. docker.io/user/liquid-host:latest)")
    parser.add_argument("--name", default=DEFAULT_ENDPOINT_NAME, help="Endpoint name")
    parser.add_argument("--repo", default=DEFAULT_MODEL_REPO, help="HF model repo to mount at /repository")
    parser.add_argument("--region", default=DEFAULT_REGION, help="Cloud region")
    parser.add_argument("--vendor", default=DEFAULT_VENDOR, help="Cloud vendor (aws, azure)")
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR, help="Accelerator type (gpu, cpu)")
    parser.add_argument("--instance-size", default=DEFAULT_INSTANCE_SIZE, help="Instance size (x1, x2, x4)")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="Instance type (nvidia-t4, nvidia-a10g, nvidia-l4)")
    parser.add_argument("--scale-to-zero", type=int, default=15, help="Minutes before scaling to zero (0 to disable)")
    parser.add_argument("--min-replica", type=int, default=0, help="Min replicas")
    parser.add_argument("--max-replica", type=int, default=1, help="Max replicas")
    parser.add_argument("--adapter", default=None, help="HF Hub adapter repo (e.g. bryanhealey/my-aiera-finetune)")
    parser.add_argument("--namespace", default=None, help="HF namespace (default: your username, or use an org name)")
    parser.add_argument("--hf-token", default=None, help="HF token to pass to the endpoint (needed for private adapter repos)")
    parser.add_argument("--training-repo", default=None, help="HF dataset repo for training data editor (e.g. user/project-data)")
    parser.add_argument("--mcp-api-key", default=None, help="API key for MCP server (e.g. Aiera API key)")
    args = parser.parse_args()

    # Verify auth
    user = whoami()
    namespace = args.namespace or user["name"]
    logger.info("Authenticated as: %s", namespace)

    logger.info("  Model repo: %s", args.repo)
    logger.info("  Image: %s", args.image)
    logger.info("  Hardware: %s / %s / %s", args.accelerator, args.instance_type, args.instance_size)

    # Try to get existing endpoint first; create if it doesn't exist
    try:
        endpoint = get_inference_endpoint(name=args.name, namespace=namespace)
        logger.info("Found existing endpoint '%s' (status: %s) — deleting and recreating...", args.name, endpoint.status)
        endpoint.delete()
        logger.info("Deleted old endpoint. Waiting for cleanup...")
        import time
        time.sleep(10)
    except Exception:
        logger.info("No existing endpoint '%s' found, creating new.", args.name)

    endpoint = create_inference_endpoint(
        name=args.name,
        namespace=namespace,
        repository=args.repo,
        framework="custom",
        accelerator=args.accelerator,
        instance_size=args.instance_size,
        instance_type=args.instance_type,
        region=args.region,
        vendor=args.vendor,
        min_replica=args.min_replica,
        max_replica=args.max_replica,
        scale_to_zero_timeout=args.scale_to_zero if args.scale_to_zero > 0 else None,
        task="custom",
        type="public",
        custom_image={
            "url": args.image,
            "health_route": "/health",
        },
        secrets={
            "MODEL_PATH": "/repository",
            "MODEL_KEY": "lfm2-24b-a2b",
            "PORT": "80",
            "DEVICE_MAP": "auto",
            **({"ADAPTER_PATH": args.adapter} if args.adapter else {}),
            **({"HF_TOKEN": args.hf_token} if args.hf_token else {}),
            **({"TRAINING_HF_REPO": args.training_repo} if args.training_repo else {}),
            **({"MCP_API_KEY": args.mcp_api_key} if args.mcp_api_key else {}),
        },
    )

    logger.info("Endpoint created!")
    logger.info("  Name: %s", endpoint.name)
    logger.info("  Status: %s", endpoint.status)
    logger.info("  URL: %s", endpoint.url)
    logger.info("")
    logger.info("Waiting for endpoint to become ready (this may take a few minutes)...")

    endpoint.wait()

    logger.info("Endpoint is ready!")
    logger.info("  URL: %s", endpoint.url)
    logger.info("")
    logger.info("Test with:")
    logger.info('  curl %s/health', endpoint.url)
    logger.info('  curl %s/v1/chat/completions -H "Content-Type: application/json" \\', endpoint.url)
    logger.info('    -d \'{"messages": [{"role": "user", "content": "Hello"}], "stream": true}\'')


if __name__ == "__main__":
    main()
