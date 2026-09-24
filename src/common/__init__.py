"""Inference shared by the on-device app and the cloud processor.

Deliberately free of any doover coupling: everything here takes and returns numpy
arrays and plain dataclasses, so the same models and the same compliance reasoning run
unchanged on a Doovit and in a Lambda. The two app shells own all the
platform-specific parts — config, subscriptions, publishing.
"""
