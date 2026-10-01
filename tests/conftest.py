"""Explicit real-model tests use local verified assets, never implicit downloads.

Thread counts and CUDA precision settings are scoped to the test run.
Reads: PANNS_TEST_CHECKPOINT_DIR and PANNS_TEST_DEVICE environment variables.
"""

import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def deterministic_tests():
    previous = torch.get_num_threads()
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(previous)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


@pytest.fixture(scope="session")
def assets():
    path = os.environ.get("PANNS_TEST_CHECKPOINT_DIR")
    if not path:
        pytest.skip("set PANNS_TEST_CHECKPOINT_DIR for real model tests")
    return Path(path)


@pytest.fixture(scope="session")
def device():
    return os.environ.get("PANNS_TEST_DEVICE", "cpu")


@pytest.fixture(scope="session")
def model_session(assets, device):
    from panns_infer import PANNsSession

    with PANNsSession(device=device, checkpoint_dir=assets, offline=True) as session:
        session.load()
        yield session
