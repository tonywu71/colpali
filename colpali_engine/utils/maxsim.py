"""Dispatch helper for late-interaction (MaxSim) scoring.

`maxsim_inbatch` is the single entry point used by both the inference processor
(``BaseVisualRetrieverProcessor.score_multi_vector``) and the in-batch losses
(``ColbertLoss`` / ``ColbertPairwiseCELoss`` / ``ColbertSigmoidLoss``). It routes
through the fused Triton/Metal kernels from ``late-interaction-kernels`` when
the dependency is installed and the runtime is supported, and falls through to
the pure-torch ``einsum + amax + sum`` otherwise.

The dispatch rules mirror what ``late_interaction_kernels.colpali_compat`` used
to do via monkey-patching. Now that colpali-engine owns the dispatch, the LIK
patch module can be retired.
"""

import importlib.util
import os

import torch

# Resolved once at import time: avoids a `find_spec` call on every loss step.
_LIK_AVAILABLE: bool = importlib.util.find_spec("late_interaction_kernels") is not None


def _dispatch_path(query: torch.Tensor, doc: torch.Tensor) -> str | None:
    """Pick the dispatch backend or return None to fall back to torch.

    Returns ``"cuda"`` for CUDA Ampere+ devices, ``"mps"`` for Apple Silicon,
    or ``None`` for every other case (no LIK installed, ``LIK_DISABLE=1``,
    mixed devices, ``d < 8``, sub-Ampere CUDA, CPU, ...).
    """
    if not _LIK_AVAILABLE:
        return None
    if os.environ.get("LIK_DISABLE", "0") == "1":
        return None
    if query.device != doc.device:
        return None
    if query.shape[-1] < 8:
        return None
    if query.is_cuda and doc.is_cuda:
        # Need Ampere or newer for bf16 + modern tensor cores.
        if torch.cuda.get_device_capability(query.device)[0] < 8:
            return None
        return "cuda"
    if query.device.type == "mps" and doc.device.type == "mps":
        return "mps"
    return None


def _torch_maxsim(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """Reference MaxSim: ``einsum("bnd,csd->bcns").amax(-1).sum(-1)``."""
    return torch.einsum("bnd,csd->bcns", query, doc).amax(dim=3).sum(dim=2)


def maxsim_inbatch(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """In-batch MaxSim scores for late-interaction retrieval.

    Args:
        query: ``[B_q, L_q, d]`` query token embeddings (padded with zeros).
        doc: ``[B_d, L_d, d]`` document token embeddings (padded with zeros).

    Returns:
        ``[B_q, B_d]`` similarity matrix — the sum over query tokens of each
        token's max similarity against ``doc``'s token dimension.

    Notes:
        Pad tokens must be exactly zero — both the LIK kernel and the reference
        path rely on zero-padding rather than an explicit mask.
    """
    path = _dispatch_path(query, doc)
    if path == "cuda":
        from late_interaction_kernels.autograd import maxsim as _lik_maxsim

        return _lik_maxsim(query, doc)
    if path == "mps":
        from late_interaction_kernels.mps import maxsim_mps as _lik_maxsim_mps

        return _lik_maxsim_mps(query, doc, normalize=False)
    return _torch_maxsim(query, doc)
