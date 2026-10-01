"""PANNs inference public interface, with lazy Torch loading.

Sessions own their model independently; importing the package does not load it.
Reads: version, session and portable result contracts on request.
"""

from .__about__ import __version__

__all__ = ["PANNsSession", "TaggingResult", "__version__"]


def __getattr__(name):
    if name == "PANNsSession":
        from .session import PANNsSession

        return PANNsSession
    if name == "TaggingResult":
        from .results import TaggingResult

        return TaggingResult
    raise AttributeError(name)
