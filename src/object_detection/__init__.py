from pydoover.docker import run_app

from .app_config import ObjectDetectionConfig
from .application import ObjectDetectionApplication

__all__ = ("ObjectDetectionApplication", "ObjectDetectionConfig", "main")


def main():
    """
    Run the application.
    """
    run_app(ObjectDetectionApplication())
