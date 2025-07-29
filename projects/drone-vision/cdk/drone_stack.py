from aws_cdk import Duration, Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_lambda_event_sources as events
from aws_cdk import aws_s3 as s3
from constructs import Construct


class DroneStack(Stack):
    """Defines the S3 → Lambda → API Gateway stack used in CI/CD."""

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:       # noqa: D401
        super().__init__(scope, id, **kwargs)

        # ── buckets ───────────────────────────────────────────────────────
        bucket_in = s3.Bucket(self, "InputBucket")
        bucket_out = s3.Bucket(self, "OutputBucket")

        # ── Lambda (Docker image built from ./lambda) ─────────────────────
        fn = _lambda.DockerImageFunction(
            self,
            "DetectorFn",
            code=_lambda.DockerImageCode.from_image_asset("lambda"),
            memory_size=512,
            timeout=Duration.seconds(30),
        )

        # trigger Lambda on every upload to the *input* bucket
        fn.add_event_source(
            events.S3EventSource(bucket_in, events=[s3.EventType.OBJECT_CREATED])
        )

        # ── public REST endpoint (binary JPEG support) ────────────────────
        api = apigw.LambdaRestApi(
            self,
            "Endpoint",
            handler=fn,
            binary_media_types=["image/jpeg"],
        )

        # surface the invoke URL in CloudFormation Outputs (visible in CI)
        self.api_url_output = api.url
