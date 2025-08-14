"""
S3 ➜ Lambda (container image) ➜ API Gateway (binary JPEG)

• Upload to *input* bucket can trigger Lambda (optional flag below).
• Lambda also exposes a REST endpoint (returns JPEG).
• Bucket name & API URL are CloudFormation outputs.
"""
from __future__ import annotations

from typing import Any, Final

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_lambda_event_sources as events
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from constructs import Construct

GEO_BUCKET_ENV: Final[str] = "GEO_BUCKET"

# Toggle this ON after your Lambda handler supports S3 events.
ENABLE_S3_TRIGGER: Final[bool] = False


def _latest_insights_version() -> _lambda.LambdaInsightsVersion | None:
    # Find the highest VERSION_* constant available in this CDK build
    names = [n for n in dir(_lambda.LambdaInsightsVersion) if n.startswith("VERSION_")]
    if not names:
        return None

    def _ver_key(n: str) -> tuple[int, ...]:
        # VERSION_1_0_143_0 -> (1, 0, 143, 0)
        return tuple(int(p) for p in n.removeprefix("VERSION_").split("_"))

    best = max(names, key=_ver_key)
    return getattr(_lambda.LambdaInsightsVersion, best)


class ArgosStack(Stack):
    """Defines the S3 → Lambda → API-Gateway pipeline."""

    def __init__(self, scope: Construct, id: str, **kwargs: Any) -> None:
        super().__init__(scope, id, **kwargs)

        # ── S3 bucket – uploads (optionally) trigger inference ───────────────
        bucket_in = s3.Bucket(
            self,
            "InputBucket",
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY,
            event_bridge_enabled=False,
        )

        iv = _latest_insights_version()

        # ── Lambda built from lambda/Dockerfile ──────────────────────────────
        fn = _lambda.DockerImageFunction(
            self,
            "DetectorFn",
            code=_lambda.DockerImageCode.from_image_asset(
                directory=".",
                file="lambda/Dockerfile",
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
            architecture=_lambda.Architecture.X86_64,
            tracing=_lambda.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.ONE_WEEK,
            insights_version=iv,  # safely None if not available in this CDK
        )

        # Allow Lambda to put results (GeoJSON) back into the bucket
        bucket_in.grant_put(fn)

        # Optional: trigger Lambda on image uploads (guarded until handler supports it)
        if ENABLE_S3_TRIGGER:
            fn.add_event_source(
                events.S3EventSource(
                    bucket_in,
                    events=[s3.EventType.OBJECT_CREATED],
                    filters=[
                        s3.NotificationKeyFilter(suffix=".jpg"),
                        s3.NotificationKeyFilter(suffix=".jpeg"),
                        s3.NotificationKeyFilter(suffix=".png"),
                    ],
                )
            )

        # ── Public HTTPS endpoint (binary JPEG) ──────────────────────────────
        api = apigw.LambdaRestApi(
            self,
            "Endpoint",
            handler=fn,  # type: ignore[arg-type]
            binary_media_types=["image/jpeg"],
            # CORS helps local tools call the API directly from browsers
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
        )

        # CloudFormation outputs
        CfnOutput(self, "InputBucketName", value=bucket_in.bucket_name)
        CfnOutput(self, "ApiInvokeUrl", value=api.url)
