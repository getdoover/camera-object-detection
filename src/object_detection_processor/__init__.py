from pydoover.processor import run_app

from .app_config import ObjectDetectionProcessorConfig
from .application import ObjectDetectionProcessor

__all__ = (
    "ObjectDetectionProcessor",
    "ObjectDetectionProcessorConfig",
    "handler",
)


def handler(event, context):
    """AWS Lambda entrypoint.

    A fresh ``ObjectDetectionProcessor`` per invocation is correct and cheap — the
    expensive part is the onnxruntime sessions, which are cached at module scope in
    ``application`` and so survive a warm container.
    """
    return run_app(ObjectDetectionProcessor(), event, context)
