import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for repo in (ROOT / "unsloth", ROOT / "unsloth-zoo"):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_SKIP_MODEL_IMPORTS", "1")

from unsloth.kernels import is_kernel_backend_available
from unsloth.kernels._backend_registry import get_kernel_backend
from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss
from unsloth.kernels.geglu import (
    geglu_approx_backward_kernel,
    geglu_approx_forward_kernel,
    geglu_exact_backward_kernel,
    geglu_exact_forward_kernel,
)
from unsloth.kernels.layernorm import fast_layernorm
from unsloth.kernels.rms_layernorm import fast_rms_layernorm
from unsloth.kernels.rope_embedding import fast_rope_embedding
from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel, swiglu_fg_kernel


CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_DEVICE = torch.device("cuda") if CUDA_AVAILABLE else torch.device("cpu")
RUN_PERF = os.environ.get("UNSLOTH_RUN_KERNEL_BACKEND_PERF", "0") == "1"
RUN_STRESS = os.environ.get("UNSLOTH_RUN_KERNEL_BACKEND_STRESS", "0") == "1"
STRESS_WARMUP = int(os.environ.get("UNSLOTH_KERNEL_BACKEND_STRESS_WARMUP", "5"))
STRESS_ITERS = int(os.environ.get("UNSLOTH_KERNEL_BACKEND_STRESS_ITERS", "15"))
STRESS_REPORT_PATH = Path(
    os.environ.get(
        "UNSLOTH_KERNEL_BACKEND_REPORT_PATH",
        str(ROOT / "outputs" / "kernel_backend_stress_report.json"),
    )
)
SEED = 3407
BACKEND_ORDER = ("eager", "triton", "cutile")
FULL_PRECISION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _available_backends():
    backends = ["eager"]
    for backend in ("triton", "cutile"):
        available, _ = is_kernel_backend_available(backend)
        if available:
            backends.append(backend)
    return backends


AVAILABLE_BACKENDS = _available_backends()


def _require_multiple_backends():
    if len(AVAILABLE_BACKENDS) < 2:
        pytest.skip("Need at least 2 available backends for comparison.")


def _set_seed():
    torch.manual_seed(SEED)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(SEED)


def _clone_for_backend(tensor: torch.Tensor, *, requires_grad: bool | None = None):
    cloned = tensor.detach().clone()
    if requires_grad is None:
        requires_grad = tensor.requires_grad
    cloned.requires_grad_(requires_grad)
    return cloned


def _clone_grad(tensor: torch.Tensor):
    return tensor.detach().clone()


def _dtype_tolerances(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    if dtype in (torch.float16, torch.bfloat16):
        return 2e-2, 2e-2
    return 1e-4, 1e-4


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
):
    default_atol, default_rtol = _dtype_tolerances(expected.dtype)
    atol = default_atol if atol is None else atol
    rtol = default_rtol if rtol is None else rtol
    actual_f = actual.detach().float().cpu()
    expected_f = expected.detach().float().cpu()
    max_abs = (actual_f - expected_f).abs().max().item()
    denom = torch.maximum(expected_f.abs(), torch.full_like(expected_f, 1e-6))
    max_rel = ((actual_f - expected_f).abs() / denom).max().item()
    torch.testing.assert_close(
        actual_f,
        expected_f,
        atol = atol,
        rtol = rtol,
        msg = f"{name}: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}",
    )


def _max_abs_rel(actual: torch.Tensor, expected: torch.Tensor):
    actual_f = actual.detach().float().cpu()
    expected_f = expected.detach().float().cpu()
    abs_diff = (actual_f - expected_f).abs()
    denom = torch.maximum(expected_f.abs(), torch.full_like(expected_f, 1e-6))
    rel_diff = abs_diff / denom
    return {
        "max_abs": abs_diff.max().item(),
        "mean_abs": abs_diff.mean().item(),
        "max_rel": rel_diff.max().item(),
        "mean_rel": rel_diff.mean().item(),
    }


def _quantile(values: list[float], q: float):
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _benchmark_cuda(closure, *, warmup: int = 3, iters: int = 7):
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA benchmark requested without CUDA.")
    for _ in range(warmup):
        closure()
    torch.cuda.synchronize()
    latencies_ms = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing = True)
        end = torch.cuda.Event(enable_timing = True)
        start.record()
        closure()
        end.record()
        torch.cuda.synchronize()
        latencies_ms.append(start.elapsed_time(end))
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return {
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "peak_mem_mb": peak_mb,
    }


def _benchmark_cuda_detailed(closure, *, warmup: int, iters: int):
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA benchmark requested without CUDA.")
    for _ in range(warmup):
        closure()
    torch.cuda.synchronize()
    latencies_ms = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing = True)
        end = torch.cuda.Event(enable_timing = True)
        start.record()
        closure()
        end.record()
        torch.cuda.synchronize()
        latencies_ms.append(start.elapsed_time(end))
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    return {
        "iters": iters,
        "warmup": warmup,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies_ms),
        "p90_ms": _quantile(latencies_ms, 0.90),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "std_ms": std_ms,
        "cov": (std_ms / mean_ms) if mean_ms else 0.0,
        "peak_mem_mb": peak_mb,
    }


def _serialize_shape(shape):
    return [int(dim) for dim in shape]


def _serialize_dtype(dtype: torch.dtype):
    return str(dtype).replace("torch.", "")


def _write_report(report: dict):
    STRESS_REPORT_PATH.parent.mkdir(parents = True, exist_ok = True)
    STRESS_REPORT_PATH.write_text(json.dumps(report, indent = 2, sort_keys = True) + "\n")


def _resolve_requested_backend(kernel_name: str, requested_backend: str, *, rope_indices = False):
    return get_kernel_backend(kernel_name, backend = requested_backend)


def _cuda_runtime_metadata():
    if not CUDA_AVAILABLE:
        return {
            "cuda_available": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_device_count": 0,
            "current_device": None,
            "device_name": None,
        }
    current_device = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_count": torch.cuda.device_count(),
        "current_device": current_device,
        "device_name": torch.cuda.get_device_name(current_device),
    }


def _rope_reference_qk(Q, K, cos, sin):
    cos = cos.squeeze()
    sin = sin.squeeze()
    half = Q.shape[-1] // 2
    cos_exp = cos[..., :half].unsqueeze(0).unsqueeze(0)
    sin_exp = sin[..., :half].unsqueeze(0).unsqueeze(0)
    q0 = Q[..., :half]
    q1 = Q[..., half:]
    k0 = K[..., :half]
    k1 = K[..., half:]
    q_out = torch.cat((q0 * cos_exp - q1 * sin_exp, q1 * cos_exp + q0 * sin_exp), dim = -1)
    k_out = torch.cat((k0 * cos_exp - k1 * sin_exp, k1 * cos_exp + k0 * sin_exp), dim = -1)
    return q_out.to(Q.dtype), k_out.to(K.dtype)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
def test_layernorm_backend_numerics_hard_case():
    _require_multiple_backends()
    _set_seed()

    X_base = torch.randn(9, 5, 96, device = CUDA_DEVICE, dtype = torch.bfloat16).transpose(0, 1)
    layernorm = torch.nn.LayerNorm(96, device = CUDA_DEVICE, dtype = torch.bfloat16)
    grad_out = torch.randn_like(X_base)

    ref_x = _clone_for_backend(X_base, requires_grad = True)
    ref_out = fast_layernorm(layernorm, ref_x, backend = "eager")
    ref_out.backward(grad_out)
    ref_grad = ref_x.grad.detach().clone()
    ref_out = ref_out.detach().clone()

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        X = _clone_for_backend(X_base, requires_grad = True)
        out = fast_layernorm(layernorm, X, backend = backend)
        out.backward(grad_out)
        _assert_close(f"layernorm output [{backend}]", out, ref_out)
        _assert_close(f"layernorm grad [{backend}]", X.grad, ref_grad)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize("dtype", FULL_PRECISION_DTYPES)
def test_layernorm_supported_dtypes(dtype: torch.dtype):
    _require_multiple_backends()
    _set_seed()

    x_base = torch.randn(3, 7, 64, device = CUDA_DEVICE, dtype = dtype)
    layernorm = torch.nn.LayerNorm(64, device = CUDA_DEVICE, dtype = dtype)
    layernorm.weight.requires_grad_(False)
    layernorm.bias.requires_grad_(False)
    grad_out = torch.randn_like(x_base)

    ref_x = _clone_for_backend(x_base, requires_grad = True)
    ref_out = fast_layernorm(layernorm, ref_x, backend = "eager")
    ref_out.backward(_clone_grad(grad_out))
    ref_grad = ref_x.grad.detach().clone()
    ref_out = ref_out.detach().clone()

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        x = _clone_for_backend(x_base, requires_grad = True)
        out = fast_layernorm(layernorm, x, backend = backend)
        out.backward(_clone_grad(grad_out))
        _assert_close(f"layernorm dtype={dtype} output [{backend}]", out, ref_out)
        _assert_close(f"layernorm dtype={dtype} grad [{backend}]", x.grad, ref_grad)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize("gemma", [False, True])
def test_rmsnorm_backend_numerics_hard_case(gemma: bool):
    _require_multiple_backends()
    _set_seed()

    X_base = torch.randn(4, 13, 80, device = CUDA_DEVICE, dtype = torch.bfloat16).transpose(0, 1)
    layernorm = SimpleNamespace(
        weight = torch.randn(80, device = CUDA_DEVICE, dtype = torch.bfloat16),
        eps = 1e-6,
    )
    grad_out = torch.randn_like(X_base)

    ref_x = _clone_for_backend(X_base, requires_grad = True)
    ref_out = fast_rms_layernorm(layernorm, ref_x, gemma = gemma, backend = "eager")
    ref_out.backward(grad_out)
    ref_grad = ref_x.grad.detach().clone()
    ref_out = ref_out.detach().clone()

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        X = _clone_for_backend(X_base, requires_grad = True)
        out = fast_rms_layernorm(layernorm, X, gemma = gemma, backend = backend)
        out.backward(grad_out)
        _assert_close(f"rmsnorm output [{backend}, gemma={gemma}]", out, ref_out)
        _assert_close(f"rmsnorm grad [{backend}, gemma={gemma}]", X.grad, ref_grad)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize("dtype", FULL_PRECISION_DTYPES)
@pytest.mark.parametrize("gemma", [False, True])
def test_rmsnorm_supported_dtypes(dtype: torch.dtype, gemma: bool):
    _require_multiple_backends()
    _set_seed()

    x_base = torch.randn(2, 5, 96, device = CUDA_DEVICE, dtype = dtype)
    layernorm = SimpleNamespace(
        weight = torch.randn(96, device = CUDA_DEVICE, dtype = dtype),
        eps = 1e-6,
    )
    grad_out = torch.randn_like(x_base)

    ref_x = _clone_for_backend(x_base, requires_grad = True)
    ref_out = fast_rms_layernorm(layernorm, ref_x, gemma = gemma, backend = "eager")
    ref_out.backward(_clone_grad(grad_out))
    ref_grad = ref_x.grad.detach().clone()
    ref_out = ref_out.detach().clone()

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        x = _clone_for_backend(x_base, requires_grad = True)
        out = fast_rms_layernorm(layernorm, x, gemma = gemma, backend = backend)
        out.backward(_clone_grad(grad_out))
        _assert_close(f"rmsnorm dtype={dtype} gemma={gemma} output [{backend}]", out, ref_out)
        _assert_close(f"rmsnorm dtype={dtype} gemma={gemma} grad [{backend}]", x.grad, ref_grad)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
def test_swiglu_backend_numerics_hard_case():
    _require_multiple_backends()
    _set_seed()

    e_base = torch.randn(3, 11, 96, device = CUDA_DEVICE, dtype = torch.bfloat16)
    g_base = torch.randn(3, 11, 96, device = CUDA_DEVICE, dtype = torch.bfloat16)

    ref_out = swiglu_fg_kernel(e_base, g_base, backend = "eager")
    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        out = swiglu_fg_kernel(e_base, g_base, backend = backend)
        _assert_close(f"swiglu forward [{backend}]", out, ref_out)

    DW_base = torch.randn(33, 96, device = CUDA_DEVICE, dtype = torch.bfloat16)
    e2_base = torch.randn(33, 96, device = CUDA_DEVICE, dtype = torch.bfloat16)
    g2_base = torch.randn(33, 96, device = CUDA_DEVICE, dtype = torch.bfloat16)
    ref_h, ref_df, ref_de = swiglu_DWf_DW_dfg_kernel(
        _clone_for_backend(DW_base),
        _clone_for_backend(e2_base),
        _clone_for_backend(g2_base),
        backend = "eager",
    )
    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        h, df, de = swiglu_DWf_DW_dfg_kernel(
            _clone_for_backend(DW_base),
            _clone_for_backend(e2_base),
            _clone_for_backend(g2_base),
            backend = backend,
        )
        _assert_close(f"swiglu backward h [{backend}]", h, ref_h)
        _assert_close(f"swiglu backward df [{backend}]", df, ref_df)
        _assert_close(f"swiglu backward de [{backend}]", de, ref_de)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
def test_swiglu_supported_dtypes(dtype: torch.dtype):
    _require_multiple_backends()
    _set_seed()

    e_base = torch.randn(2, 9, 64, device = CUDA_DEVICE, dtype = dtype)
    g_base = torch.randn(2, 9, 64, device = CUDA_DEVICE, dtype = dtype)
    ref_out = swiglu_fg_kernel(e_base, g_base, backend = "eager")
    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        out = swiglu_fg_kernel(e_base, g_base, backend = backend)
        _assert_close(f"swiglu dtype={dtype} forward [{backend}]", out, ref_out)

    dw_base = torch.randn(18, 64, device = CUDA_DEVICE, dtype = dtype)
    e2_base = torch.randn(18, 64, device = CUDA_DEVICE, dtype = dtype)
    g2_base = torch.randn(18, 64, device = CUDA_DEVICE, dtype = dtype)
    ref_h, ref_df, ref_de = swiglu_DWf_DW_dfg_kernel(
        _clone_for_backend(dw_base),
        _clone_for_backend(e2_base),
        _clone_for_backend(g2_base),
        backend = "eager",
    )
    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        h, df, de = swiglu_DWf_DW_dfg_kernel(
            _clone_for_backend(dw_base),
            _clone_for_backend(e2_base),
            _clone_for_backend(g2_base),
            backend = backend,
        )
        _assert_close(f"swiglu dtype={dtype} backward h [{backend}]", h, ref_h)
        _assert_close(f"swiglu dtype={dtype} backward df [{backend}]", df, ref_df)
        _assert_close(f"swiglu dtype={dtype} backward de [{backend}]", de, ref_de)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize(
    "forward_fn,backward_fn,name",
    [
        (geglu_exact_forward_kernel, geglu_exact_backward_kernel, "exact"),
        (geglu_approx_forward_kernel, geglu_approx_backward_kernel, "approx"),
    ],
)
def test_geglu_backend_numerics_hard_case(forward_fn, backward_fn, name: str):
    _require_multiple_backends()
    _set_seed()

    gate = torch.randn(3, 13, 80, device = CUDA_DEVICE, dtype = torch.bfloat16)
    up = torch.randn(3, 13, 80, device = CUDA_DEVICE, dtype = torch.bfloat16)
    ref_out = forward_fn(gate, up, backend = "eager")
    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        out = forward_fn(gate, up, backend = backend)
        _assert_close(f"geglu {name} forward [{backend}]", out, ref_out)

    DW = torch.randn(39, 80, device = CUDA_DEVICE, dtype = torch.bfloat16)
    e = torch.randn(39, 80, device = CUDA_DEVICE, dtype = torch.bfloat16)
    g = torch.randn(39, 80, device = CUDA_DEVICE, dtype = torch.bfloat16)
    ref_h, ref_df, ref_de = backward_fn(
        _clone_for_backend(DW),
        _clone_for_backend(e),
        _clone_for_backend(g),
        backend = "eager",
    )
    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        h, df, de = backward_fn(
            _clone_for_backend(DW),
            _clone_for_backend(e),
            _clone_for_backend(g),
            backend = backend,
        )
        _assert_close(f"geglu {name} backward h [{backend}]", h, ref_h)
        _assert_close(f"geglu {name} backward df [{backend}]", df, ref_df)
        _assert_close(f"geglu {name} backward de [{backend}]", de, ref_de)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
@pytest.mark.parametrize(
    "forward_fn,backward_fn,name",
    [
        (geglu_exact_forward_kernel, geglu_exact_backward_kernel, "exact"),
        (geglu_approx_forward_kernel, geglu_approx_backward_kernel, "approx"),
    ],
)
def test_geglu_supported_dtypes(dtype: torch.dtype, forward_fn, backward_fn, name: str):
    _require_multiple_backends()
    _set_seed()

    gate = torch.randn(2, 7, 64, device = CUDA_DEVICE, dtype = dtype)
    up = torch.randn(2, 7, 64, device = CUDA_DEVICE, dtype = dtype)
    ref_out = forward_fn(gate, up, backend = "eager")
    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        out = forward_fn(gate, up, backend = backend)
        _assert_close(f"geglu {name} dtype={dtype} forward [{backend}]", out, ref_out)

    dw = torch.randn(14, 64, device = CUDA_DEVICE, dtype = dtype)
    e = torch.randn(14, 64, device = CUDA_DEVICE, dtype = dtype)
    g = torch.randn(14, 64, device = CUDA_DEVICE, dtype = dtype)
    ref_h, ref_df, ref_de = backward_fn(
        _clone_for_backend(dw),
        _clone_for_backend(e),
        _clone_for_backend(g),
        backend = "eager",
    )
    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        h, df, de = backward_fn(
            _clone_for_backend(dw),
            _clone_for_backend(e),
            _clone_for_backend(g),
            backend = backend,
        )
        _assert_close(f"geglu {name} dtype={dtype} backward h [{backend}]", h, ref_h)
        _assert_close(f"geglu {name} dtype={dtype} backward df [{backend}]", df, ref_df)
        _assert_close(f"geglu {name} dtype={dtype} backward de [{backend}]", de, ref_de)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize(
    "batch,seq_len,vocab_size,softcap,scale,dtype",
    [
        (2, 9, 257, 0.0, 0.0, torch.bfloat16),
        (1, 3, 70001, 3.0, 1.25, torch.float32),
    ],
)
def test_cross_entropy_backend_numerics_hard_cases(batch, seq_len, vocab_size, softcap, scale, dtype):
    _require_multiple_backends()
    _set_seed()

    logits_base = torch.randn(
        batch,
        seq_len,
        vocab_size,
        device = CUDA_DEVICE,
        dtype = dtype,
    )
    labels = torch.randint(0, vocab_size, (batch, seq_len), device = CUDA_DEVICE)
    labels[0, 0] = -100
    if batch * seq_len > 1:
        labels.view(-1)[-1] = -100

    ref_logits = _clone_for_backend(logits_base, requires_grad = True)
    ref_loss = fast_cross_entropy_loss(
        ref_logits,
        labels,
        logit_softcapping = softcap,
        logit_scaling = scale,
        backend = "eager",
    )
    ref_loss.backward()
    ref_grad = ref_logits.grad.detach().clone()
    ref_loss = ref_loss.detach().clone()
    loss_atol, loss_rtol = _dtype_tolerances(ref_loss.dtype)
    if dtype == torch.float32 and vocab_size >= 65536:
        loss_atol = max(loss_atol, 3e-2)
        loss_rtol = max(loss_rtol, 3e-3)

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        logits = _clone_for_backend(logits_base, requires_grad = True)
        loss = fast_cross_entropy_loss(
            logits,
            labels,
            logit_softcapping = softcap,
            logit_scaling = scale,
            backend = backend,
        )
        loss.backward()
        _assert_close(
            f"cross entropy loss [{backend}, vocab={vocab_size}]",
            loss,
            ref_loss,
            atol = loss_atol,
            rtol = loss_rtol,
        )
        _assert_close(f"cross entropy grad [{backend}, vocab={vocab_size}]", logits.grad, ref_grad)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize("dtype", FULL_PRECISION_DTYPES)
def test_cross_entropy_supported_dtypes(dtype: torch.dtype):
    _require_multiple_backends()
    _set_seed()

    logits_base = torch.randn(2, 11, 1024, device = CUDA_DEVICE, dtype = dtype)
    labels = torch.randint(0, 1024, (2, 11), device = CUDA_DEVICE)
    labels[0, 0] = -100
    labels[1, -1] = -100

    ref_logits = _clone_for_backend(logits_base, requires_grad = True)
    ref_loss = fast_cross_entropy_loss(ref_logits, labels, backend = "eager")
    ref_loss.backward()
    ref_grad = ref_logits.grad.detach().clone()
    ref_loss = ref_loss.detach().clone()

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        logits = _clone_for_backend(logits_base, requires_grad = True)
        loss = fast_cross_entropy_loss(logits, labels, backend = backend)
        loss.backward()
        _assert_close(f"cross entropy dtype={dtype} loss [{backend}]", loss, ref_loss)
        _assert_close(f"cross entropy dtype={dtype} grad [{backend}]", logits.grad, ref_grad)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
def test_rope_backend_numerics_hard_case():
    _require_multiple_backends()
    _set_seed()

    q_base = torch.randn(2, 4, 9, 32, device = CUDA_DEVICE, dtype = torch.bfloat16)
    k_base = torch.randn(2, 2, 9, 32, device = CUDA_DEVICE, dtype = torch.bfloat16)
    cos = torch.randn(32, 32, device = CUDA_DEVICE, dtype = torch.float32)
    sin = torch.randn(32, 32, device = CUDA_DEVICE, dtype = torch.float32)
    grad_q = torch.randn_like(q_base)
    grad_k = torch.randn_like(k_base)

    ref_q = _clone_for_backend(q_base, requires_grad = True)
    ref_k = _clone_for_backend(k_base, requires_grad = True)
    ref_q_out, ref_k_out = fast_rope_embedding(
        ref_q,
        ref_k,
        cos,
        sin,
        backend = "eager",
    )
    torch.autograd.backward((ref_q_out, ref_k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
    ref_q_grad = ref_q.grad.detach().clone()
    ref_k_grad = ref_k.grad.detach().clone()
    ref_q_out = ref_q_out.detach().clone()
    ref_k_out = ref_k_out.detach().clone()

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        q = _clone_for_backend(q_base, requires_grad = True)
        k = _clone_for_backend(k_base, requires_grad = True)
        q_out, k_out = fast_rope_embedding(
            q,
            k,
            cos,
            sin,
            backend = backend,
        )
        torch.autograd.backward((q_out, k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
        _assert_close(f"rope q output [{backend}]", q_out, ref_q_out)
        _assert_close(f"rope k output [{backend}]", k_out, ref_k_out)
        _assert_close(f"rope q grad [{backend}]", q.grad, ref_q_grad)
        _assert_close(f"rope k grad [{backend}]", k.grad, ref_k_grad)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize("dtype", FULL_PRECISION_DTYPES)
def test_rope_supported_dtypes(dtype: torch.dtype):
    _require_multiple_backends()
    _set_seed()

    q_base = torch.randn(2, 4, 13, 32, device = CUDA_DEVICE, dtype = dtype)
    k_base = torch.randn(2, 2, 13, 32, device = CUDA_DEVICE, dtype = dtype)
    cos = torch.randn(24, 32, device = CUDA_DEVICE, dtype = torch.float32)
    sin = torch.randn(24, 32, device = CUDA_DEVICE, dtype = torch.float32)
    grad_q = torch.randn_like(q_base)
    grad_k = torch.randn_like(k_base)

    ref_q = _clone_for_backend(q_base, requires_grad = True)
    ref_k = _clone_for_backend(k_base, requires_grad = True)
    ref_q_out, ref_k_out = fast_rope_embedding(ref_q, ref_k, cos, sin, backend = "eager")
    torch.autograd.backward((ref_q_out, ref_k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
    ref_q_grad = ref_q.grad.detach().clone()
    ref_k_grad = ref_k.grad.detach().clone()
    ref_q_out = ref_q_out.detach().clone()
    ref_k_out = ref_k_out.detach().clone()

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        q = _clone_for_backend(q_base, requires_grad = True)
        k = _clone_for_backend(k_base, requires_grad = True)
        q_out, k_out = fast_rope_embedding(q, k, cos, sin, backend = backend)
        torch.autograd.backward((q_out, k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
        _assert_close(f"rope dtype={dtype} q output [{backend}]", q_out, ref_q_out)
        _assert_close(f"rope dtype={dtype} k output [{backend}]", k_out, ref_k_out)
        _assert_close(f"rope dtype={dtype} q grad [{backend}]", q.grad, ref_q_grad)
        _assert_close(f"rope dtype={dtype} k grad [{backend}]", k.grad, ref_k_grad)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
def test_rope_nonstandard_cos_stride_hard_case(dtype: torch.dtype):
    _require_multiple_backends()
    _set_seed()

    q_base = torch.randn(2, 8, 64, 64, device = CUDA_DEVICE, dtype = dtype)
    k_base = torch.randn(2, 4, 64, 64, device = CUDA_DEVICE, dtype = dtype)
    half = q_base.shape[-1] // 2
    cos_half = torch.randn(64, half, device = CUDA_DEVICE, dtype = dtype)
    sin_half = torch.randn(64, half, device = CUDA_DEVICE, dtype = dtype)
    cos_full = torch.cat((cos_half, torch.randn_like(cos_half)), dim = -1)
    sin_full = torch.cat((sin_half, torch.randn_like(sin_half)), dim = -1)
    grad_q = torch.randn_like(q_base)
    grad_k = torch.randn_like(k_base)

    ref_q = _clone_for_backend(q_base, requires_grad = True)
    ref_k = _clone_for_backend(k_base, requires_grad = True)
    ref_q_out, ref_k_out = _rope_reference_qk(ref_q, ref_k, cos_half, sin_half)
    torch.autograd.backward((ref_q_out, ref_k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
    ref_q_grad = ref_q.grad.detach().clone()
    ref_k_grad = ref_k.grad.detach().clone()
    ref_q_out = ref_q_out.detach().clone()
    ref_k_out = ref_k_out.detach().clone()

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        q = _clone_for_backend(q_base, requires_grad = True)
        k = _clone_for_backend(k_base, requires_grad = True)
        q_out, k_out = fast_rope_embedding(q, k, cos_full, sin_full, backend = backend)
        torch.autograd.backward((q_out, k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
        _assert_close(f"rope nonstandard cos stride q output [{backend}, dtype={dtype}]", q_out, ref_q_out)
        _assert_close(f"rope nonstandard cos stride k output [{backend}, dtype={dtype}]", k_out, ref_k_out)
        _assert_close(f"rope nonstandard cos stride q grad [{backend}, dtype={dtype}]", q.grad, ref_q_grad)
        _assert_close(f"rope nonstandard cos stride k grad [{backend}, dtype={dtype}]", k.grad, ref_k_grad)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
@pytest.mark.parametrize("head_dim", [96, 160])
def test_rope_non_power_of_2_head_dim_hard_case(dtype: torch.dtype, head_dim: int):
    _require_multiple_backends()
    _set_seed()

    q_base = torch.randn(2, 8, 64, head_dim, device = CUDA_DEVICE, dtype = dtype)
    k_base = torch.randn(2, 4, 64, head_dim, device = CUDA_DEVICE, dtype = dtype)
    half = head_dim // 2
    cos = torch.randn(64, half, device = CUDA_DEVICE, dtype = dtype)
    sin = torch.randn(64, half, device = CUDA_DEVICE, dtype = dtype)
    grad_q = torch.randn_like(q_base)
    grad_k = torch.randn_like(k_base)

    ref_q = _clone_for_backend(q_base, requires_grad = True)
    ref_k = _clone_for_backend(k_base, requires_grad = True)
    ref_q_out, ref_k_out = _rope_reference_qk(ref_q, ref_k, cos, sin)
    torch.autograd.backward((ref_q_out, ref_k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
    ref_q_grad = ref_q.grad.detach().clone()
    ref_k_grad = ref_k.grad.detach().clone()
    ref_q_out = ref_q_out.detach().clone()
    ref_k_out = ref_k_out.detach().clone()

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        q = _clone_for_backend(q_base, requires_grad = True)
        k = _clone_for_backend(k_base, requires_grad = True)
        q_out, k_out = fast_rope_embedding(q, k, cos, sin, backend = backend)
        torch.autograd.backward((q_out, k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
        _assert_close(f"rope non-power2 q output [{backend}, dtype={dtype}, head_dim={head_dim}]", q_out, ref_q_out)
        _assert_close(f"rope non-power2 k output [{backend}, dtype={dtype}, head_dim={head_dim}]", k_out, ref_k_out)
        _assert_close(f"rope non-power2 q grad [{backend}, dtype={dtype}, head_dim={head_dim}]", q.grad, ref_q_grad)
        _assert_close(f"rope non-power2 k grad [{backend}, dtype={dtype}, head_dim={head_dim}]", k.grad, ref_k_grad)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
def test_rope_backend_indices_correctness_hard_case():
    _require_multiple_backends()
    _set_seed()

    q_base = torch.randn(2, 4, 9, 32, device = CUDA_DEVICE, dtype = torch.bfloat16)
    k_base = torch.randn(2, 2, 9, 32, device = CUDA_DEVICE, dtype = torch.bfloat16)
    cos = torch.randn(32, 32, device = CUDA_DEVICE, dtype = torch.float32)
    sin = torch.randn(32, 32, device = CUDA_DEVICE, dtype = torch.float32)
    rope_indices = torch.randint(0, 32, (2, 9), device = CUDA_DEVICE, dtype = torch.int32)
    grad_q = torch.randn_like(q_base)
    grad_k = torch.randn_like(k_base)

    ref_q = _clone_for_backend(q_base, requires_grad = True)
    ref_k = _clone_for_backend(k_base, requires_grad = True)
    ref_q_out, ref_k_out = fast_rope_embedding(
        ref_q,
        ref_k,
        cos,
        sin,
        rope_embedding_indices = rope_indices,
        backend = "eager",
    )
    torch.autograd.backward((ref_q_out, ref_k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
    ref_q_grad = ref_q.grad.detach().clone()
    ref_k_grad = ref_k.grad.detach().clone()
    ref_q_out = ref_q_out.detach().clone()
    ref_k_out = ref_k_out.detach().clone()

    for backend in AVAILABLE_BACKENDS:
        if backend == "eager":
            continue
        q = _clone_for_backend(q_base, requires_grad = True)
        k = _clone_for_backend(k_base, requires_grad = True)
        q_out, k_out = fast_rope_embedding(
            q,
            k,
            cos,
            sin,
            rope_embedding_indices = rope_indices,
            backend = backend,
        )
        torch.autograd.backward((q_out, k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
        _assert_close(f"rope indices q output [{backend}]", q_out, ref_q_out)
        _assert_close(f"rope indices k output [{backend}]", k_out, ref_k_out)
        _assert_close(f"rope indices q grad [{backend}]", q.grad, ref_q_grad)
        _assert_close(f"rope indices k grad [{backend}]", k.grad, ref_k_grad)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.skipif(not RUN_PERF, reason = "Set UNSLOTH_RUN_KERNEL_BACKEND_PERF=1 to run GPU perf smoke.")
def test_backend_performance_smoke_and_report():
    _require_multiple_backends()
    _set_seed()

    rows = []

    layernorm = torch.nn.LayerNorm(96, device = CUDA_DEVICE, dtype = torch.bfloat16)
    x_base = torch.randn(8, 16, 96, device = CUDA_DEVICE, dtype = torch.bfloat16)

    def layernorm_closure(backend):
        def _run():
            x = x_base.detach().clone()
            out = fast_layernorm(layernorm, x, backend = backend)
            torch.cuda.synchronize()
            return out

        return _run

    logits_base = torch.randn(2, 8, 70001, device = CUDA_DEVICE, dtype = torch.float32)
    labels = torch.randint(0, 70001, (2, 8), device = CUDA_DEVICE)
    labels[0, 0] = -100

    def cross_entropy_closure(backend):
        def _run():
            logits = logits_base.detach().clone().requires_grad_(True)
            loss = fast_cross_entropy_loss(
                logits,
                labels,
                logit_softcapping = 3.0,
                logit_scaling = 1.15,
                backend = backend,
            )
            loss.backward()
            torch.cuda.synchronize()
            return loss

        return _run

    q_base = torch.randn(2, 8, 17, 32, device = CUDA_DEVICE, dtype = torch.bfloat16)
    k_base = torch.randn(2, 4, 17, 32, device = CUDA_DEVICE, dtype = torch.bfloat16)
    cos = torch.randn(48, 32, device = CUDA_DEVICE, dtype = torch.float32)
    sin = torch.randn(48, 32, device = CUDA_DEVICE, dtype = torch.float32)
    def rope_closure(backend):
        def _run():
            q = q_base.detach().clone().requires_grad_(True)
            k = k_base.detach().clone().requires_grad_(True)
            q_out, k_out = fast_rope_embedding(
                q,
                k,
                cos,
                sin,
                backend = backend,
            )
            (q_out.float().sum() + k_out.float().sum()).backward()
            torch.cuda.synchronize()
            return q_out, k_out

        return _run

    benchmarks = [
        ("layernorm", layernorm_closure),
        ("cross_entropy_chunked", cross_entropy_closure),
        ("rope_qk", rope_closure),
    ]

    for op_name, closure_factory in benchmarks:
        for backend in AVAILABLE_BACKENDS:
            metrics = _benchmark_cuda(closure_factory(backend))
            rows.append({"op": op_name, "backend": backend, **metrics})

    print("\nKERNEL_BACKEND_BENCHMARKS=" + json.dumps(rows, sort_keys = True))
    assert rows
    assert all(row["median_ms"] > 0 for row in rows)
    assert all(row["peak_mem_mb"] >= 0 for row in rows)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason = "CUDA backend comparison requires CUDA.")
@pytest.mark.skipif(not RUN_STRESS, reason = "Set UNSLOTH_RUN_KERNEL_BACKEND_STRESS=1 to run GPU stress benchmarks.")
def test_backend_performance_stress_and_numerics_report():
    _require_multiple_backends()
    _set_seed()
    runtime_metadata = _cuda_runtime_metadata()
    assert runtime_metadata["visible_device_count"] >= 1
    assert runtime_metadata["current_device"] == 0

    requested_backends = list(AVAILABLE_BACKENDS)
    numerics_rows = []
    benchmark_rows = []

    layernorm_cases = [
        {
            "case_id": "layernorm_bf16_contig_8x128x1024",
            "shape": (8, 128, 1024),
            "dtype": torch.bfloat16,
            "eps": 1e-5,
            "noncontiguous": False,
        },
        {
            "case_id": "layernorm_bf16_noncontig_8x256x2048",
            "shape": (8, 256, 2048),
            "dtype": torch.bfloat16,
            "eps": 1e-5,
            "noncontiguous": True,
        },
        {
            "case_id": "layernorm_fp32_contig_2x1024x4096",
            "shape": (2, 1024, 4096),
            "dtype": torch.float32,
            "eps": 1e-5,
            "noncontiguous": False,
        },
    ]

    for case in layernorm_cases:
        batch, seq_len, hidden_dim = case["shape"]
        if case["noncontiguous"]:
            x_base = torch.randn(seq_len, batch, hidden_dim, device = CUDA_DEVICE, dtype = case["dtype"]).transpose(0, 1)
        else:
            x_base = torch.randn(batch, seq_len, hidden_dim, device = CUDA_DEVICE, dtype = case["dtype"])
        grad_out = torch.randn_like(x_base)
        layernorm = torch.nn.LayerNorm(hidden_dim, eps = case["eps"], device = CUDA_DEVICE, dtype = case["dtype"])
        layernorm.weight.requires_grad_(False)
        layernorm.bias.requires_grad_(False)

        def run_layernorm(backend):
            x = _clone_for_backend(x_base, requires_grad = True)
            out = fast_layernorm(layernorm, x, backend = backend)
            out.backward(_clone_grad(grad_out))
            return out.detach().clone(), x.grad.detach().clone()

        ref_out, ref_grad = run_layernorm("eager")
        eager_metrics = _benchmark_cuda_detailed(
            lambda: fast_layernorm(layernorm, x_base.detach().clone(), backend = "eager"),
            warmup = STRESS_WARMUP,
            iters = STRESS_ITERS,
        )
        benchmark_rows.append(
            {
                "op": "layernorm",
                "case_id": case["case_id"],
                "requested_backend": "eager",
                "effective_backend": "eager",
                "dtype": _serialize_dtype(case["dtype"]),
                "shape": _serialize_shape(case["shape"]),
                **eager_metrics,
                "speedup_vs_eager": 1.0,
            }
        )
        numerics_rows.append(
            {
                "op": "layernorm",
                "case_id": case["case_id"],
                "requested_backend": "eager",
                "effective_backend": "eager",
                "output": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0, "mean_rel": 0.0},
                "grad": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0, "mean_rel": 0.0},
            }
        )

        for backend in requested_backends:
            if backend == "eager":
                continue
            out, grad = run_layernorm(backend)
            _assert_close(f"{case['case_id']} output [{backend}]", out, ref_out)
            _assert_close(f"{case['case_id']} grad [{backend}]", grad, ref_grad)
            output_error = _max_abs_rel(out, ref_out)
            grad_error = _max_abs_rel(grad, ref_grad)
            effective_backend = _resolve_requested_backend("unsloth.layernorm", backend)
            metrics = _benchmark_cuda_detailed(
                lambda backend = backend: fast_layernorm(layernorm, x_base.detach().clone(), backend = backend),
                warmup = STRESS_WARMUP,
                iters = STRESS_ITERS,
            )
            benchmark_rows.append(
                {
                    "op": "layernorm",
                    "case_id": case["case_id"],
                    "requested_backend": backend,
                    "effective_backend": effective_backend,
                    "dtype": _serialize_dtype(case["dtype"]),
                    "shape": _serialize_shape(case["shape"]),
                    **metrics,
                    "speedup_vs_eager": eager_metrics["median_ms"] / metrics["median_ms"],
                }
            )
            numerics_rows.append(
                {
                    "op": "layernorm",
                    "case_id": case["case_id"],
                    "requested_backend": backend,
                    "effective_backend": effective_backend,
                    "output": output_error,
                    "grad": grad_error,
                }
            )

    cross_entropy_cases = [
        {
            "case_id": "cross_entropy_bf16_2x256x32000",
            "shape": (2, 256, 32000),
            "dtype": torch.bfloat16,
            "softcap": 0.0,
            "scale": 0.0,
        },
        {
            "case_id": "cross_entropy_fp32_2x128x70001_softcap",
            "shape": (2, 128, 70001),
            "dtype": torch.float32,
            "softcap": 3.0,
            "scale": 1.25,
        },
        {
            "case_id": "cross_entropy_bf16_2x128x128000",
            "shape": (2, 128, 128000),
            "dtype": torch.bfloat16,
            "softcap": 0.0,
            "scale": 0.0,
        },
    ]

    for case in cross_entropy_cases:
        batch, seq_len, vocab_size = case["shape"]
        logits_base = torch.randn(batch, seq_len, vocab_size, device = CUDA_DEVICE, dtype = case["dtype"])
        labels = torch.randint(0, vocab_size, (batch, seq_len), device = CUDA_DEVICE)
        labels[0, 0] = -100
        labels.view(-1)[-1] = -100

        def run_cross_entropy(backend):
            logits = _clone_for_backend(logits_base, requires_grad = True)
            loss = fast_cross_entropy_loss(
                logits,
                labels,
                logit_softcapping = case["softcap"],
                logit_scaling = case["scale"],
                backend = backend,
            )
            loss.backward()
            return loss.detach().clone(), logits.grad.detach().clone()

        ref_loss, ref_grad = run_cross_entropy("eager")
        eager_metrics = _benchmark_cuda_detailed(
            lambda: fast_cross_entropy_loss(
                logits_base.detach().clone().requires_grad_(True),
                labels,
                logit_softcapping = case["softcap"],
                logit_scaling = case["scale"],
                backend = "eager",
            ).backward(),
            warmup = STRESS_WARMUP,
            iters = STRESS_ITERS,
        )
        benchmark_rows.append(
            {
                "op": "cross_entropy",
                "case_id": case["case_id"],
                "requested_backend": "eager",
                "effective_backend": "eager",
                "dtype": _serialize_dtype(case["dtype"]),
                "shape": _serialize_shape(case["shape"]),
                **eager_metrics,
                "speedup_vs_eager": 1.0,
            }
        )
        numerics_rows.append(
            {
                "op": "cross_entropy",
                "case_id": case["case_id"],
                "requested_backend": "eager",
                "effective_backend": "eager",
                "loss": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0, "mean_rel": 0.0},
                "grad": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0, "mean_rel": 0.0},
            }
        )
        loss_atol, loss_rtol = _dtype_tolerances(ref_loss.dtype)
        if case["dtype"] == torch.float32 and vocab_size >= 65536:
            loss_atol = max(loss_atol, 3e-2)
            loss_rtol = max(loss_rtol, 3e-3)

        for backend in requested_backends:
            if backend == "eager":
                continue
            loss, grad = run_cross_entropy(backend)
            _assert_close(
                f"{case['case_id']} loss [{backend}]",
                loss,
                ref_loss,
                atol = loss_atol,
                rtol = loss_rtol,
            )
            _assert_close(f"{case['case_id']} grad [{backend}]", grad, ref_grad)
            loss_error = _max_abs_rel(loss, ref_loss)
            grad_error = _max_abs_rel(grad, ref_grad)
            effective_backend = _resolve_requested_backend("unsloth.cross_entropy_loss", backend)
            metrics = _benchmark_cuda_detailed(
                lambda backend = backend: fast_cross_entropy_loss(
                    logits_base.detach().clone().requires_grad_(True),
                    labels,
                    logit_softcapping = case["softcap"],
                    logit_scaling = case["scale"],
                    backend = backend,
                ).backward(),
                warmup = STRESS_WARMUP,
                iters = STRESS_ITERS,
            )
            benchmark_rows.append(
                {
                    "op": "cross_entropy",
                    "case_id": case["case_id"],
                    "requested_backend": backend,
                    "effective_backend": effective_backend,
                    "dtype": _serialize_dtype(case["dtype"]),
                    "shape": _serialize_shape(case["shape"]),
                    **metrics,
                    "speedup_vs_eager": eager_metrics["median_ms"] / metrics["median_ms"],
                }
            )
            numerics_rows.append(
                {
                    "op": "cross_entropy",
                    "case_id": case["case_id"],
                    "requested_backend": backend,
                    "effective_backend": effective_backend,
                    "loss": loss_error,
                    "grad": grad_error,
                }
            )

    rope_cases = [
        {
            "case_id": "rope_qk_bf16_4x8x4x256x64",
            "batch": 4,
            "q_heads": 8,
            "k_heads": 4,
            "seq_len": 256,
            "head_dim": 64,
            "dtype": torch.bfloat16,
            "use_indices": False,
        },
        {
            "case_id": "rope_qk_bf16_2x16x8x512x128",
            "batch": 2,
            "q_heads": 16,
            "k_heads": 8,
            "seq_len": 512,
            "head_dim": 128,
            "dtype": torch.bfloat16,
            "use_indices": False,
        },
        {
            "case_id": "rope_qk_indices_bf16_2x8x4x512x128",
            "batch": 2,
            "q_heads": 8,
            "k_heads": 4,
            "seq_len": 512,
            "head_dim": 128,
            "dtype": torch.bfloat16,
            "use_indices": True,
        },
    ]

    for case in rope_cases:
        q_base = torch.randn(
            case["batch"],
            case["q_heads"],
            case["seq_len"],
            case["head_dim"],
            device = CUDA_DEVICE,
            dtype = case["dtype"],
        )
        k_base = torch.randn(
            case["batch"],
            case["k_heads"],
            case["seq_len"],
            case["head_dim"],
            device = CUDA_DEVICE,
            dtype = case["dtype"],
        )
        cos = torch.randn(case["seq_len"] + 32, case["head_dim"], device = CUDA_DEVICE, dtype = torch.float32)
        sin = torch.randn(case["seq_len"] + 32, case["head_dim"], device = CUDA_DEVICE, dtype = torch.float32)
        rope_indices = None
        if case["use_indices"]:
            rope_indices = torch.randint(
                0,
                case["seq_len"] + 32,
                (case["batch"], case["seq_len"]),
                device = CUDA_DEVICE,
                dtype = torch.int32,
            )
        grad_q = torch.randn_like(q_base)
        grad_k = torch.randn_like(k_base)

        def run_rope(backend):
            q = _clone_for_backend(q_base, requires_grad = True)
            k = _clone_for_backend(k_base, requires_grad = True)
            q_out, k_out = fast_rope_embedding(
                q,
                k,
                cos,
                sin,
                rope_embedding_indices = rope_indices,
                backend = backend,
            )
            torch.autograd.backward((q_out, k_out), (_clone_grad(grad_q), _clone_grad(grad_k)))
            return q_out.detach().clone(), k_out.detach().clone(), q.grad.detach().clone(), k.grad.detach().clone()

        ref_q_out, ref_k_out, ref_q_grad, ref_k_grad = run_rope("eager")
        eager_metrics = _benchmark_cuda_detailed(
            lambda: (
                lambda q, k: torch.autograd.backward(
                    fast_rope_embedding(
                        q,
                        k,
                        cos,
                        sin,
                        rope_embedding_indices = rope_indices,
                        backend = "eager",
                    ),
                    (_clone_grad(grad_q), _clone_grad(grad_k)),
                )
            )(
                q_base.detach().clone().requires_grad_(True),
                k_base.detach().clone().requires_grad_(True),
            ),
            warmup = STRESS_WARMUP,
            iters = STRESS_ITERS,
        )
        benchmark_rows.append(
            {
                "op": "rope_qk",
                "case_id": case["case_id"],
                "requested_backend": "eager",
                "effective_backend": "eager",
                "dtype": _serialize_dtype(case["dtype"]),
                "shape": {
                    "q": _serialize_shape(q_base.shape),
                    "k": _serialize_shape(k_base.shape),
                    "indices": case["use_indices"],
                },
                **eager_metrics,
                "speedup_vs_eager": 1.0,
            }
        )
        numerics_rows.append(
            {
                "op": "rope_qk",
                "case_id": case["case_id"],
                "requested_backend": "eager",
                "effective_backend": "eager",
                "q_output": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0, "mean_rel": 0.0},
                "k_output": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0, "mean_rel": 0.0},
                "q_grad": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0, "mean_rel": 0.0},
                "k_grad": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0, "mean_rel": 0.0},
            }
        )

        for backend in requested_backends:
            if backend == "eager":
                continue
            q_out, k_out, q_grad, k_grad = run_rope(backend)
            _assert_close(f"{case['case_id']} q output [{backend}]", q_out, ref_q_out)
            _assert_close(f"{case['case_id']} k output [{backend}]", k_out, ref_k_out)
            _assert_close(f"{case['case_id']} q grad [{backend}]", q_grad, ref_q_grad)
            _assert_close(f"{case['case_id']} k grad [{backend}]", k_grad, ref_k_grad)
            effective_backend = _resolve_requested_backend(
                "unsloth.rope_embedding_qk",
                backend,
                rope_indices = case["use_indices"],
            )
            metrics = _benchmark_cuda_detailed(
                lambda backend = backend: (
                    lambda q, k: torch.autograd.backward(
                        fast_rope_embedding(
                            q,
                            k,
                            cos,
                            sin,
                            rope_embedding_indices = rope_indices,
                            backend = backend,
                        ),
                        (_clone_grad(grad_q), _clone_grad(grad_k)),
                    )
                )(
                    q_base.detach().clone().requires_grad_(True),
                    k_base.detach().clone().requires_grad_(True),
                ),
                warmup = STRESS_WARMUP,
                iters = STRESS_ITERS,
            )
            benchmark_rows.append(
                {
                    "op": "rope_qk",
                    "case_id": case["case_id"],
                    "requested_backend": backend,
                    "effective_backend": effective_backend,
                    "dtype": _serialize_dtype(case["dtype"]),
                    "shape": {
                        "q": _serialize_shape(q_base.shape),
                        "k": _serialize_shape(k_base.shape),
                        "indices": case["use_indices"],
                    },
                    **metrics,
                    "speedup_vs_eager": eager_metrics["median_ms"] / metrics["median_ms"],
                }
            )
            numerics_rows.append(
                {
                    "op": "rope_qk",
                    "case_id": case["case_id"],
                    "requested_backend": backend,
                    "effective_backend": effective_backend,
                    "q_output": _max_abs_rel(q_out, ref_q_out),
                    "k_output": _max_abs_rel(k_out, ref_k_out),
                    "q_grad": _max_abs_rel(q_grad, ref_q_grad),
                    "k_grad": _max_abs_rel(k_grad, ref_k_grad),
                }
            )

    report = {
        "timestamp": int(time.time()),
        "runtime": runtime_metadata,
        "available_backends": requested_backends,
        "stress_warmup": STRESS_WARMUP,
        "stress_iters": STRESS_ITERS,
        "benchmark_rows": benchmark_rows,
        "numerics_rows": numerics_rows,
    }
    _write_report(report)
    print("\nKERNEL_BACKEND_STRESS_REPORT=" + json.dumps(report, sort_keys = True))

    assert benchmark_rows
    assert numerics_rows
    assert all(row["median_ms"] > 0 for row in benchmark_rows)
    assert all(row["peak_mem_mb"] >= 0 for row in benchmark_rows)
    assert STRESS_REPORT_PATH.exists()
