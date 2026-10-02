"""Independent PANNs model ownership and explicit load/infer/release lifecycle.

A session serializes its operations and reuses its loaded model across calls.
Torch imports lazily; closing is terminal, while releasing permits reloading.
Reads: artifact resolver, private model/audio boundary and portable result contracts.
"""

import copy
import csv
import gc
import threading

from .checkpoints import CheckpointStore
from .results import CheckpointInfo, TaggingResult


def _resolve_device(request):
    import torch

    if not isinstance(request, str):
        raise ValueError("device must be auto, cpu, cuda or cuda:N")
    if request == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda" or (request.startswith("cuda:") and request[5:].isdigit()):
        index = 0 if request == "cuda" else int(request[5:])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device is unavailable: {request}")
        return torch.device(f"cuda:{index}")
    raise ValueError("device must be auto, cpu, cuda or cuda:N")


class PANNsSession:
    """Own one CNN14 model, with explicit readiness and no hidden cold start."""

    def __init__(
        self,
        *,
        device="auto",
        cache_dir=None,
        checkpoint_dir=None,
        offline=False,
        checkpoint_config=None,
    ):
        self._device_request = device
        self._store = CheckpointStore(cache_dir, checkpoint_dir, offline, checkpoint_config)
        self._lock = threading.RLock()
        self._status = "new"
        self._model = self._device = self._labels = self._label_ids = self._checkpoint = None
        self._loaded_provenance = None

    @property
    def status(self):
        return self._status

    @property
    def device(self):
        return str(self._device) if self._device is not None else None

    def cache_info(self):
        return self._store.cache_info()

    def loaded_checkpoint_info(self):
        """Return a portable snapshot of the successfully loaded model's provenance.

        Ready-only and serialized with inference; never touches checkpoint files.
        Returned nested values are independent of the session and contain no paths.
        """
        with self._lock:
            self._require_open()
            if self._status != "ready":
                raise RuntimeError("Call load() successfully before loaded_checkpoint_info()")
            return copy.deepcopy(self._loaded_provenance)

    def _require_open(self):
        if self._status == "closed":
            raise RuntimeError("This session is closed")

    def load(self):
        with self._lock:
            self._require_open()
            if self._status == "ready":
                return self
            self._status = "loading"
            try:
                device = _resolve_device(self._device_request)
                files = self._store.resolve()
                from ._impl.model import CLASSES, Cnn14

                with files["class_labels_indices.csv"].open() as source:
                    rows = list(csv.DictReader(source, skipinitialspace=True))
                    labels = {int(r["index"]): r["display_name"] for r in rows}
                    mids = {int(r["index"]): r["mid"] for r in rows}
                if len(rows) != CLASSES or len(set(mids.values())) != CLASSES:
                    raise ValueError("AudioSet mapping contains duplicate or missing classes")
                if set(labels) != set(range(CLASSES)):
                    raise ValueError("AudioSet mapping must cover exactly 527 classes")
                model = Cnn14()
                model.load_checkpoint(files["Cnn14.pth"])
                model.to(device).eval().requires_grad_(False)
                provenance = self._store.provenance()
                checkpoint = CheckpointInfo(
                    provenance["repo"],
                    provenance["revision"],
                    provenance["source_revision"],
                    tuple((name, meta["sha256"]) for name, meta in provenance["files"].items()),
                )
                for artifact in provenance["files"].values():
                    artifact.pop("path")
                provenance.update(
                    model_id="panns-cnn14",
                    checkpoint_id=f"{checkpoint.repo}@{checkpoint.revision}",
                    checkpoint_sha256=dict(checkpoint.artifacts)["Cnn14.pth"],
                    device=str(device),
                )
            except Exception:
                self._status = "failed"
                raise
            self._model, self._device = model, device
            self._labels = tuple(labels[i] for i in range(CLASSES))
            self._label_ids = tuple(mids[i] for i in range(CLASSES))
            self._checkpoint = checkpoint
            self._loaded_provenance = provenance
            self._status = "ready"
            return self

    def infer(self, audio, *, sample_rate=None, pad_short=False):
        with self._lock:
            self._require_open()
            if self._status != "ready":
                raise RuntimeError("Call load() successfully before infer()")
            if type(pad_short) is not bool:
                raise ValueError("pad_short must be a bool")
            import torch

            from ._impl.audio import normalize_audio

            waveform, info = normalize_audio(audio, sample_rate, self._device, pad_short=pad_short)
            self._status = "busy"
            try:
                with torch.inference_mode():
                    output = self._model(waveform)
                return TaggingResult(
                    tuple(output["clipwise_output"][0].cpu().tolist()),
                    self._labels,
                    tuple(output["embedding"][0].cpu().tolist()),
                    info,
                    self._checkpoint,
                    self._label_ids,
                )
            finally:
                self._status = "ready"

    def _release_model(self):
        device = self._device
        self._model = self._device = self._labels = self._label_ids = self._checkpoint = None
        self._loaded_provenance = None
        gc.collect()
        if device is not None:
            import torch

            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    def release(self):
        with self._lock:
            self._require_open()
            self._release_model()
            self._status = "released"

    def close(self):
        with self._lock:
            if self._status != "closed":
                self._release_model()
                self._status = "closed"

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()
