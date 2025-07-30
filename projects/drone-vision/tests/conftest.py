import warnings

warnings.filterwarnings(
    "ignore",
    message="datetime.datetime.utcnow\\(\\) is deprecated",
    category=DeprecationWarning,
    module=r".*botocore",
)
