"""ParallelHue public package."""

from .client import (
    ClientConfig,
    ExactTelemetryError,
    ParallelHueClient,
    StreamChunk,
    UnixTelemetryReceiver,
)

__all__ = [
    "ClientConfig",
    "ExactTelemetryError",
    "ParallelHueClient",
    "StreamChunk",
    "UnixTelemetryReceiver",
]

__version__ = "0.1.0"
