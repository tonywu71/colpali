"""CUDA-gated parity and training-smoke tests for the LIK MaxSim integration.

These tests verify that the fused kernel agrees with the pure-torch fallback on
both forward and backward, and that the three migrated in-batch losses train
without producing NaNs / collapsing. They are marked ``@pytest.mark.slow`` so
they are skipped by the default CPU CI; run them with ``pytest -m slow`` on a
host with a CUDA Ampere+ GPU and ``late-interaction-kernels`` installed.
"""

import pytest
import torch

from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
    ColbertPairwiseCELoss,
    ColbertSigmoidLoss,
)
from colpali_engine.utils.maxsim import _dispatch_path, _torch_maxsim, maxsim_inbatch

pytest.importorskip("late_interaction_kernels")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        reason="LIK kernel requires Ampere or newer",
    ),
]

EMBEDDING_DIM = 128
BATCH_SIZE = 4
QUERY_LEN = 16
DOC_LEN = 32

_DTYPES: list[torch.dtype] = [torch.float32, torch.float16, torch.bfloat16]


def _atol(dtype: torch.dtype) -> float:
    return 1e-4 if dtype == torch.float32 else 2e-2


def _random_inputs(dtype: torch.dtype, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    query = torch.randn(BATCH_SIZE, QUERY_LEN, EMBEDDING_DIM, dtype=dtype, device="cuda")
    doc = torch.randn(BATCH_SIZE, DOC_LEN, EMBEDDING_DIM, dtype=dtype, device="cuda")
    if requires_grad:
        query.requires_grad_(True)
        doc.requires_grad_(True)
    return query, doc


def test_dispatch_path_returns_cuda() -> None:
    query, doc = _random_inputs(torch.float32)
    assert _dispatch_path(query, doc) == "cuda"


@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward_parity_cuda(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    query, doc = _random_inputs(dtype)

    got = maxsim_inbatch(query, doc)

    monkeypatch.setenv("LIK_DISABLE", "1")
    want = maxsim_inbatch(query, doc)
    # Sanity: the disabled path should match the reference einsum exactly.
    assert torch.allclose(want, _torch_maxsim(query, doc))

    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=_atol(dtype))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_backward_parity_cuda(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    query_lik, doc_lik = _random_inputs(dtype, requires_grad=True)
    maxsim_inbatch(query_lik, doc_lik).sum().backward()

    monkeypatch.setenv("LIK_DISABLE", "1")
    query_ref, doc_ref = _random_inputs(dtype, requires_grad=True)
    maxsim_inbatch(query_ref, doc_ref).sum().backward()

    # The fused-kernel backward uses fp32 atomic adds; the torch path reduces
    # in the input dtype. We compare in fp32 with a tolerance matching the
    # forward parity check.
    tol = _atol(dtype)
    assert query_lik.grad is not None and query_ref.grad is not None
    assert doc_lik.grad is not None and doc_ref.grad is not None
    assert torch.allclose(query_lik.grad.float(), query_ref.grad.float(), atol=tol)
    assert torch.allclose(doc_lik.grad.float(), doc_ref.grad.float(), atol=tol)


@pytest.mark.parametrize(
    "loss_cls",
    [ColbertLoss, ColbertPairwiseCELoss, ColbertSigmoidLoss],
)
def test_loss_training_smoke_cuda(loss_cls: type) -> None:
    """5 SGD steps on random embeddings — loss must stay finite and trend down."""
    torch.manual_seed(0)
    query = torch.randn(BATCH_SIZE, QUERY_LEN, EMBEDDING_DIM, device="cuda", requires_grad=True)
    doc = torch.randn(BATCH_SIZE, DOC_LEN, EMBEDDING_DIM, device="cuda", requires_grad=True)
    loss_fn = loss_cls(normalize_scores=False).to("cuda")
    optimizer = torch.optim.SGD([query, doc], lr=1e-2)

    losses: list[float] = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(query, doc)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses))), f"non-finite loss: {losses}"
    # Loose check: final loss should be at least as low as the initial. SGD on
    # a 4×16/4×32 random batch with lr=1e-2 reliably drives the loss down; the
    # weak inequality leaves headroom for stochastic edge cases.
    assert losses[-1] <= losses[0], f"loss did not decrease: {losses}"
