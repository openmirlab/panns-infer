"""Portable PANNs results without tensor or model ownership.

The embedding is the native post-fc1 ReLU clip vector, without normalization.
All fields survive release and serialize through dataclasses.asdict to JSON.
Reads: normalized inference outputs, source audio metadata and checkpoint identity.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    samples: int
    duration_seconds: float
    model_sample_rate: int = 32000
    normalized_samples: int | None = None
    model_samples: int | None = None
    padding_samples: int = 0
    padding_seconds: float = 0.0


@dataclass(frozen=True)
class TagScore:
    index: int
    label: str
    score: float


@dataclass(frozen=True)
class CheckpointInfo:
    repo: str
    revision: str
    source_revision: str
    artifacts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TaggingResult:
    scores: tuple[float, ...]
    labels: tuple[str, ...]
    embedding: tuple[float, ...]
    audio: AudioInfo
    checkpoint: CheckpointInfo
    label_ids: tuple[str, ...]
    embedding_shape: tuple[int, ...] = (2048,)
    embedding_dtype: str = "float32"
    embedding_meaning: str = "native CNN14 clip embedding after fc1 ReLU; not L2 normalized"

    @property
    def ranked_tags(self):
        return tuple(
            TagScore(i, self.labels[i], self.scores[i])
            for i in sorted(range(len(self.scores)), key=lambda i: (-self.scores[i], i))
        )
