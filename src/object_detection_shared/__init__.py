"""Config and tags shared by the device app and the cloud processor.

Separate from ``common`` (which must stay free of pydoover) and from both app packages
(the processor can't import ``object_detection``, whose runtime needs grpc).
"""
