from aws_cdk import (
    Stack, aws_s3 as s3,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_lambda_event_sources as events,
)
from constructs import Construct

class DroneStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        bucket_in  = s3.Bucket(self, "InputBucket")
        bucket_out = s3.Bucket(self, "OutputBucket")

        fn = _lambda.DockerImageFunction(
            self, "DetectorFn",
            code=_lambda.DockerImageCode.from_image_asset(
                directory="../lambda"
            ),
            memory_size=512,
            timeout=_lambda.Duration.seconds(30),
        )

        # trigger on upload
        fn.add_event_source(events.S3EventSource(bucket_in,
            events=[s3.EventType.OBJECT_CREATED]))

        api = apigw.LambdaRestApi(self, "Endpoint", handler=fn)
        self.api_url = api.url
