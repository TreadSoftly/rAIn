"""
S3 ➜ Lambda (container image) ➜ API Gateway (binary JPEG)

• Upload to *input* bucket triggers Lambda.
• Lambda also exposes a REST endpoint (returns JPEG).
• Bucket name & API URL are CloudFormation outputs.
"""
from __future__ import annotations

from typing import Any, Final

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigateway as apigw,
    aws_lambda as _lambda,
    aws_lambda_event_sources as events,
    aws_s3 as s3,
)
from constructs import Construct

GEO_BUCKET_ENV: Final[str] = "GEO_BUCKET"


class ArgosStack(Stack):
    """Defines the S3 → Lambda → API-Gateway pipeline."""

    def __init__(self, scope: Construct, id: str, **kwargs: Any) -> None:
        super().__init__(scope, id, **kwargs)

        # ── S3 bucket – uploads trigger inference ────────────────
        bucket_in = s3.Bucket(
            self,
            "InputBucket",
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY,
            event_bridge_enabled=False,
        )

        # ── Lambda built from lambda/Dockerfile ──────────────────
        fn = _lambda.DockerImageFunction(
            self,
            "DetectorFn",
            code=_lambda.DockerImageCode.from_image_asset(
                directory=".",              # build context = projects/argos  ✅ fixed
                file="lambda/Dockerfile",   # Dockerfile within that context
                exclude=[
                    ".git",
                    ".venv",
                    ".pytest_cache",
                    ".vscode",
                    "cdk.out",
                    "web",
                ],
            ),
            environment={GEO_BUCKET_ENV: bucket_in.bucket_name},
            memory_size=512,
            timeout=Duration.seconds(30),
        )

        # Allow Lambda to put results (GeoJSON) back into the bucket
        bucket_in.grant_put(fn)

        # Trigger Lambda on every upload
        fn.add_event_source(
            events.S3EventSource(bucket_in, events=[s3.EventType.OBJECT_CREATED])
        )

        # ── Public HTTPS endpoint (binary JPEG) ──────────────────
        api = apigw.LambdaRestApi(
            self,
            "Endpoint",
            handler=fn,                       # type: ignore[arg-type]
            binary_media_types=["image/jpeg"],
        )

        # CloudFormation outputs
        CfnOutput(self, "InputBucketName", value=bucket_in.bucket_name)
        CfnOutput(self, "ApiInvokeUrl", value=api.url)
