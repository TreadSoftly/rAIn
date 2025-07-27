from aws_cdk import (
    Stack,
    Duration,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_lambda_event_sources as events,
)
from constructs import Construct


class DroneStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs: object) -> None:
        super().__init__(scope, id, **kwargs) # type: ignore[arg-type]

        # S3 buckets for uploads and (future) processed results
        bucket_in  = s3.Bucket(self, "InputBucket")
        bucket_out = s3.Bucket(self, "OutputBucket") # type: ignore

        # Docker‑based Lambda built from ../lambda
        fn = _lambda.DockerImageFunction(
            self,
            "DetectorFn",
            code=_lambda.DockerImageCode.from_image_asset("lambda"),
            memory_size=512,
            timeout=Duration.seconds(30),
        )

        # Trigger Lambda whenever a new object is put in the input bucket
        fn.add_event_source(
            events.S3EventSource(bucket_in, events=[s3.EventType.OBJECT_CREATED])
        )

        # Simple REST endpoint -> Lambda
        api = apigw.LambdaRestApi(self, "Endpoint", handler=fn) # type: ignore[arg-type]

        # Output the invoke URL so the CI logs show it
        self.api_url_output = api.url
