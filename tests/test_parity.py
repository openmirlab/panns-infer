"""All native scores and embedding components match unchanged-upstream fixtures.

Numeric tolerances account for documented Torch backend changes, with discrete
ranking assertions and an independent audio-normalization trace comparison.
Reads: committed original CPU/CUDA activations and optional verified model assets.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from panns_infer._impl.audio import normalize_audio

ROOT = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


@pytest.mark.parametrize("fixture", MANIFEST, ids=lambda f: f["id"])
def test_normalization_baseline(fixture):
    gold = np.load(ROOT / "baseline_cpu" / (fixture["id"] + ".npz"))
    wave, _ = normalize_audio(ROOT / fixture["path"], None, "cpu")
    np.testing.assert_allclose(wave.numpy(), gold["waveform"], rtol=1e-5, atol=1e-6)


@pytest.mark.model
@pytest.mark.parametrize("fixture", MANIFEST, ids=lambda f: f["id"])
def test_all_native_outputs(fixture, model_session, device):
    gold = np.load(ROOT / ("baseline_" + device.split(":")[0]) / (fixture["id"] + ".npz"))
    result = model_session.infer(ROOT / fixture["path"])
    np.testing.assert_allclose(result.scores, gold["clipwise_output"][0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(result.embedding, gold["embedding"][0], rtol=1e-4, atol=1e-4)
    assert len(result.labels) == 527 and len(result.embedding) == 2048
    np.testing.assert_array_equal(
        [tag.index for tag in result.ranked_tags[:10]], np.argsort(-gold["clipwise_output"][0])[:10]
    )
    assert result.embedding_shape == (2048,) and result.embedding_dtype == "float32"
    assert result.checkpoint.source_revision == "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"
    wave, _ = normalize_audio(ROOT / fixture["path"], None, device)
    with torch.inference_mode():
        model = model_session._model
        mel = model.logmel_extractor(model.spectrogram_extractor(wave)).cpu().numpy()
    np.testing.assert_allclose(mel, gold["logmel"], rtol=1e-4, atol=1e-4)
