import os

import pytest
import torch

from colpali_engine.utils.maxsim import _dispatch_path, _torch_maxsim, maxsim_inbatch

EMBEDDING_DIM = 32


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_maxsim_inbatch_matches_einsum_baseline(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    query = torch.randn(4, 6, EMBEDDING_DIM, dtype=dtype)
    doc = torch.randn(7, 10, EMBEDDING_DIM, dtype=dtype)

    got = maxsim_inbatch(query, doc)
    want = _torch_maxsim(query, doc)

    assert got.shape == (4, 7)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    assert torch.allclose(got, want, atol=tol)


def test_dispatch_path_falls_back_on_cpu() -> None:
    query = torch.randn(2, 3, EMBEDDING_DIM)
    doc = torch.randn(2, 3, EMBEDDING_DIM)
    assert _dispatch_path(query, doc) is None


def test_dispatch_path_honors_lik_disable_env_var() -> None:
    query = torch.randn(2, 3, EMBEDDING_DIM)
    doc = torch.randn(2, 3, EMBEDDING_DIM)
    prev = os.environ.get("LIK_DISABLE")
    os.environ["LIK_DISABLE"] = "1"
    try:
        assert _dispatch_path(query, doc) is None
    finally:
        if prev is None:
            del os.environ["LIK_DISABLE"]
        else:
            os.environ["LIK_DISABLE"] = prev


def test_dispatch_path_rejects_tiny_embedding_dim() -> None:
    query = torch.randn(2, 3, 4)
    doc = torch.randn(2, 3, 4)
    assert _dispatch_path(query, doc) is None
