"""
CDK stack that wires up

S3  ➜  Lambda (container image)  ➜  API Gateway (public HTTPS)

* Every object uploaded to the **input** bucket invokes the Lambda.
* The Lambda exposes a REST endpoint capable of returning binary JPEG.
* The bucket name and API URL are exported as CloudFormation outputs.
"""
from __future__ import annotations

from typing import Any
from typing import Final

from aws_cdk import CfnOutput
from aws_cdk import Duration
from aws_cdk import RemovalPolicy
from aws_cdk import Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_lambda_event_sources as events
from aws_cdk import aws_s3 as s3
from constructs import Construct

# Environment variable injected into the Lambda container - kept in one place.
GEO_BUCKET_ENV: Final[str] = "GEO_BUCKET"


class DroneStack(Stack):
    """Defines the S3 → Lambda → API-Gateway pipeline used in CI/CD."""

    def __init__(self, scope: Construct, id: str, **kwargs: Any) -> None:  # noqa: D401
        super().__init__(scope, id, **kwargs)

        # ── S3 bucket - objects dropped here trigger inference ────────────
        bucket_in = s3.Bucket(
            self,
            "InputBucket",
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY,
            event_bridge_enabled=False,
        )

        # ── Lambda (Docker image built from ../lambda) ────────────────────
        fn = _lambda.DockerImageFunction(
            self,
            "DetectorFn",
            code=_lambda.DockerImageCode.from_image_asset(
                directory="lambda",  # relative to project root
            ),
            environment={GEO_BUCKET_ENV: bucket_in.bucket_name},
            memory_size=512,
            timeout=Duration.seconds(30),
        )

        # Allow Lambda to PUT results back into the same bucket (geo-jsons)
        bucket_in.grant_put(fn)

        # Trigger Lambda on every upload to the *input* bucket
        fn.add_event_source(
            events.S3EventSource(bucket_in, events=[s3.EventType.OBJECT_CREATED])
        )

        # ── Public HTTPS endpoint (binary JPEG support) ───────────────────
        api = apigw.LambdaRestApi(
            self,
            "Endpoint",
            handler=fn,  # type: ignore[arg-type]
            binary_media_types=["image/jpeg"],
        )

        # Surface resource names/URLs in CloudFormation Outputs
        CfnOutput(self, "InputBucketName", value=bucket_in.bucket_name)
        CfnOutput(self, "ApiInvokeUrl", value=api.url)
