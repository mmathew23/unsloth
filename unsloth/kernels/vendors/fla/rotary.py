# referenced from https://github.com/fla-org/flash-linear-attention.git
# Adapted for Unsloth vendor integration.

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fla_rotary_embedding_qk_kernel(
    q_ptr,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    k_ptr,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seqlen_offsets_ptr,
    seqlen,
    head_dim: tl.constexpr,
    n_heads_k: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_position = tl.program_id(0)
    head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    batch_id = row_position // seqlen
    seq_index = row_position - batch_id * seqlen
    rot_position = seq_index + tl.load(seqlen_offsets_ptr + batch_id).to(tl.int32)

    cos = tl.load(
        cos_ptr + rot_position * cos_row_stride + col_offsets,
        mask = mask,
        other = 1.0,
    )
    sin = tl.load(
        sin_ptr + rot_position * sin_row_stride + col_offsets,
        mask = mask,
        other = 0.0,
    )
    if CONJUGATE:
        sin = -sin

    q_base = (
        q_ptr
        + batch_id * q_batch_stride
        + head_position * q_head_stride
        + seq_index * q_seq_stride
    )
    q0 = tl.load(q_base + col_offsets, mask = mask, other = 0.0)
    q1 = tl.load(q_base + half_head_dim + col_offsets, mask = mask, other = 0.0)
    q_out0 = q0 * cos - q1 * sin
    q_out1 = q0 * sin + q1 * cos
    tl.store(q_base + col_offsets, q_out0, mask = mask)
    tl.store(q_base + half_head_dim + col_offsets, q_out1, mask = mask)

    if head_position < n_heads_k:
        k_base = (
            k_ptr
            + batch_id * k_batch_stride
            + head_position * k_head_stride
            + seq_index * k_seq_stride
        )
        k0 = tl.load(k_base + col_offsets, mask = mask, other = 0.0)
        k1 = tl.load(k_base + half_head_dim + col_offsets, mask = mask, other = 0.0)
        k_out0 = k0 * cos - k1 * sin
        k_out1 = k0 * sin + k1 * cos
        tl.store(k_base + col_offsets, k_out0, mask = mask)
        tl.store(k_base + half_head_dim + col_offsets, k_out1, mask = mask)


_fla_rotary_embedding_qk_kernel = triton.heuristics(
    {
        "CONJUGATE": lambda args: bool(args["CONJUGATE"]),
    }
)(_fla_rotary_embedding_qk_kernel)


@triton.jit
def _fla_rotary_embedding_qk_indices_kernel(
    q_ptr,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    k_ptr,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    position_ids_ptr,
    seqlen,
    head_dim: tl.constexpr,
    n_heads_k: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_position = tl.program_id(0)
    head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    if HAS_POSITION_IDS:
        rot_position = tl.load(
            position_ids_ptr + row_position,
            eviction_policy = "evict_first",
        ).to(tl.int32)
    else:
        rot_position = row_position % seqlen

    cos = tl.load(
        cos_ptr + rot_position * cos_row_stride + col_offsets,
        mask = mask,
        other = 1.0,
    )
    sin = tl.load(
        sin_ptr + rot_position * sin_row_stride + col_offsets,
        mask = mask,
        other = 0.0,
    )
    if CONJUGATE:
        sin = -sin

    batch_id = row_position // seqlen
    seq_index = row_position - batch_id * seqlen

    q_base = (
        q_ptr
        + batch_id * q_batch_stride
        + head_position * q_head_stride
        + seq_index * q_seq_stride
    )
    q0 = tl.load(q_base + col_offsets, mask = mask, other = 0.0)
    q1 = tl.load(q_base + half_head_dim + col_offsets, mask = mask, other = 0.0)
    tl.store(q_base + col_offsets, q0 * cos - q1 * sin, mask = mask)
    tl.store(q_base + half_head_dim + col_offsets, q1 * cos + q0 * sin, mask = mask)

    if head_position < n_heads_k:
        k_base = (
            k_ptr
            + batch_id * k_batch_stride
            + head_position * k_head_stride
            + seq_index * k_seq_stride
        )
        k0 = tl.load(k_base + col_offsets, mask = mask, other = 0.0)
        k1 = tl.load(k_base + half_head_dim + col_offsets, mask = mask, other = 0.0)
        tl.store(k_base + col_offsets, k0 * cos - k1 * sin, mask = mask)
        tl.store(k_base + half_head_dim + col_offsets, k1 * cos + k0 * sin, mask = mask)


_fla_rotary_embedding_qk_indices_kernel = triton.heuristics(
    {
        "CONJUGATE": lambda args: bool(args["CONJUGATE"]),
        "HAS_POSITION_IDS": lambda args: bool(args["HAS_POSITION_IDS"]),
    }
)(_fla_rotary_embedding_qk_indices_kernel)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps = num_warps, num_stages = num_stages)
        for num_warps in (2, 4, 8, 16)
        for num_stages in (2, 3, 4)
    ],
    key = ["H", "D", "R"],
)
@triton.jit
def _fla_rotary_embedding_kernel(
    x_ptr,
    y_ptr,
    cos_ptr,
    sin_ptr,
    seqlen_offsets_ptr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    TR: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    CONJUGATE: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_b = tl.program_id(1)
    i_h = tl.program_id(2)

    if i_t * BT >= T:
        return

    offsets = tl.load(seqlen_offsets_ptr + i_b).to(tl.int32)
    o_t = i_t * BT + tl.arange(0, BT)
    o_cs = o_t + offsets
    m_t = (o_t >= 0) & (o_t < T) & (o_cs >= 0) & (o_cs < TR)

    o_r = tl.arange(0, BD // 2)
    mask = m_t[:, None] & (o_r < R)[None, :]

    p_x = x_ptr + (((i_b * T + o_t[:, None]) * H + i_h) * D + o_r[None, :])
    p_cos = cos_ptr + (o_cs[:, None] * R + o_r[None, :])
    p_sin = sin_ptr + (o_cs[:, None] * R + o_r[None, :])

    b_cos = tl.load(p_cos, mask = mask, other = 1.0).to(tl.float32)
    b_sin = tl.load(p_sin, mask = mask, other = 0.0).to(tl.float32)
    if CONJUGATE:
        b_sin = -b_sin

    b_x0 = tl.load(p_x, mask = mask, other = 0.0).to(tl.float32)
    b_x1 = tl.load(p_x + R, mask = mask, other = 0.0).to(tl.float32)

    b_o0 = b_x0 * b_cos - b_x1 * b_sin
    b_o1 = b_x0 * b_sin + b_x1 * b_cos

    p_y = y_ptr + (((i_b * T + o_t[:, None]) * H + i_h) * D + o_r[None, :])
    tl.store(p_y, b_o0, mask = mask)
    tl.store(p_y + R, b_o1, mask = mask)


def _get_multiprocessor_count(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    if device.type == "hip":
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 1


def _normalize_seqlen_offsets(
    seqlen_offsets: int | torch.Tensor | None,
    x: torch.Tensor,
) -> torch.Tensor:
    batch = x.shape[0]
    if seqlen_offsets is None:
        return torch.zeros(batch, dtype = torch.int32, device = x.device)

    if isinstance(seqlen_offsets, int):
        return torch.full(
            (batch,),
            int(seqlen_offsets),
            dtype = torch.int32,
            device = x.device,
        )

    if seqlen_offsets.shape != (batch,):
        raise ValueError(
            f"seqlen_offsets must have shape {(batch,)}, got {tuple(seqlen_offsets.shape)}"
        )
    if seqlen_offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"seqlen_offsets must be int32/int64, got {seqlen_offsets.dtype}"
        )
    return seqlen_offsets.to(device = x.device, dtype = torch.int32, non_blocking = True)


def _normalize_position_ids(
    position_ids: torch.Tensor | None,
    q: torch.Tensor,
) -> torch.Tensor | None:
    if position_ids is None:
        return None

    batch = q.shape[0]
    seqlen = q.shape[2]
    if not isinstance(position_ids, torch.Tensor):
        raise ValueError("position_ids must be a torch.Tensor or None.")
    if position_ids.shape != (batch, seqlen):
        raise ValueError(
            f"position_ids must have shape {(batch, seqlen)}, got {tuple(position_ids.shape)}"
        )
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"position_ids must be int32/int64, got {position_ids.dtype}"
        )
    return position_ids.to(device = q.device, dtype = torch.int32, non_blocking = True).reshape(-1).contiguous()


def _calculate_launch_settings(head_dim: int) -> tuple[int, int]:
    block_size = triton.next_power_of_2(head_dim)
    num_warps = 4
    if block_size >= 32768:
        num_warps = 32
    elif block_size >= 8192:
        num_warps = 16
    elif block_size >= 2048:
        num_warps = 8
    return block_size, num_warps


def _fla_rotary_qk_fwdbwd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: torch.Tensor,
    conjugate: bool = False,
    validate_offsets: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must have 4 dimensions [B, H, T, D], got {q.shape}, {k.shape}")
    if q.device.type not in ("cuda", "hip"):
        raise ValueError(f"Unsupported device type {q.device.type}")
    if q.device != k.device:
        raise ValueError("q and k must be on the same device.")
    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("cos and sin must have shape [TR, R].")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have identical shapes, got {cos.shape} and {sin.shape}")

    bq, hq, tq, dq = q.shape
    bk, hk, tk, dk = k.shape
    if bq != bk or tq != tk or dq != dk:
        raise ValueError(f"Incompatible q/k shapes: {q.shape}, {k.shape}")
    if dq % 2 != 0:
        raise ValueError(f"head_dim must be even, got {dq}.")
    if dq > 256:
        raise ValueError(f"Unsupported head_dim={dq}. FLA rotary backend requires <= 256.")

    seq_len = tq
    rotary_half_dim = dq // 2
    tr_len, r = cos.shape
    if r != rotary_half_dim:
        raise ValueError(f"cos/sin second dim must equal D//2={rotary_half_dim}, got {r}.")

    if validate_offsets:
        min_offset = int(seqlen_offsets.min().item())
        max_offset = int(seqlen_offsets.max().item())
        if min_offset < 0 or (max_offset + seq_len > tr_len):
            raise ValueError(
                f"Invalid seqlen offsets range [{min_offset}, {max_offset}] for "
                f"seq_len={seq_len} and cache length={tr_len}."
            )

    q_out = q
    k_out = k
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
    if not seqlen_offsets.is_contiguous():
        seqlen_offsets = seqlen_offsets.contiguous()

    block_size, num_warps = _calculate_launch_settings(rotary_half_dim)
    grid = (bq * seq_len, hq)
    _fla_rotary_embedding_qk_kernel[grid](
        q_ptr = q_out,
        q_batch_stride = q_out.stride(0),
        q_head_stride = q_out.stride(1),
        q_seq_stride = q_out.stride(2),
        k_ptr = k_out,
        k_batch_stride = k_out.stride(0),
        k_head_stride = k_out.stride(1),
        k_seq_stride = k_out.stride(2),
        cos_ptr = cos,
        cos_row_stride = cos.stride(0),
        sin_ptr = sin,
        sin_row_stride = sin.stride(0),
        seqlen_offsets_ptr = seqlen_offsets,
        seqlen = seq_len,
        head_dim = dq,
        n_heads_k = hk,
        CONJUGATE = conjugate,
        BLOCK_SIZE = block_size,
        num_warps = num_warps,
    )
    return q_out, k_out


def _fla_rotary_qk_positions_fwdbwd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    conjugate: bool = False,
    validate_positions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must have 4 dimensions [B, H, T, D], got {q.shape}, {k.shape}")
    if q.device.type not in ("cuda", "hip"):
        raise ValueError(f"Unsupported device type {q.device.type}")
    if q.device != k.device:
        raise ValueError("q and k must be on the same device.")
    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("cos and sin must have shape [TR, R>=D//2].")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have identical shapes, got {cos.shape} and {sin.shape}")

    bq, hq, tq, dq = q.shape
    bk, hk, tk, dk = k.shape
    if bq != bk or tq != tk or dq != dk:
        raise ValueError(f"Incompatible q/k shapes: {q.shape}, {k.shape}")
    if dq % 2 != 0:
        raise ValueError(f"head_dim must be even, got {dq}.")
    if dq > 256:
        raise ValueError(f"Unsupported head_dim={dq}. FLA rotary backend requires <= 256.")

    rotary_half_dim = dq // 2
    tr_len, r = cos.shape
    if r < rotary_half_dim:
        raise ValueError(f"cos/sin second dim must be >= D//2={rotary_half_dim}, got {r}.")
    if cos.stride(-1) != 1 or sin.stride(-1) != 1:
        cos = cos.contiguous()
        sin = sin.contiguous()

    pos_flat = _normalize_position_ids(position_ids, q)
    if validate_positions:
        if pos_flat is None:
            if tq > tr_len:
                raise ValueError(
                    f"seqlen={tq} exceeds cache length={tr_len} for no-offset positions."
                )
        else:
            min_pos = int(pos_flat.min().item())
            max_pos = int(pos_flat.max().item())
            if min_pos < 0 or max_pos >= tr_len:
                raise ValueError(
                    f"Invalid position_ids range [{min_pos}, {max_pos}] for cache length={tr_len}."
                )

    if pos_flat is None:
        pos_ptr = q.new_empty(1, dtype = torch.int32)
    else:
        pos_ptr = pos_flat

    q_out = q
    k_out = k
    block_size, num_warps = _calculate_launch_settings(rotary_half_dim)
    grid = (bq * tq, hq)
    _fla_rotary_embedding_qk_indices_kernel[grid](
        q_ptr = q_out,
        q_batch_stride = q_out.stride(0),
        q_head_stride = q_out.stride(1),
        q_seq_stride = q_out.stride(2),
        k_ptr = k_out,
        k_batch_stride = k_out.stride(0),
        k_head_stride = k_out.stride(1),
        k_seq_stride = k_out.stride(2),
        cos_ptr = cos,
        cos_row_stride = cos.stride(0),
        sin_ptr = sin,
        sin_row_stride = sin.stride(0),
        position_ids_ptr = pos_ptr,
        seqlen = tq,
        head_dim = dq,
        n_heads_k = hk,
        HAS_POSITION_IDS = pos_flat is not None,
        CONJUGATE = conjugate,
        BLOCK_SIZE = block_size,
        num_warps = num_warps,
    )
    return q_out, k_out


def _fla_rotary_fwdbwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: torch.Tensor,
    conjugate: bool = False,
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"x must have 4 dimensions [B, T, H, D], got {x.shape}")
    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("cos and sin must have shape [TR, R]")
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos and sin must have identical shapes, got {cos.shape} and {sin.shape}"
        )
    if x.device.type not in ("cuda", "hip"):
        raise ValueError(f"Unsupported device type {x.device.type}")

    batch, seq_len, n_heads, head_dim = x.shape
    tr_len, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2

    if head_dim > 256:
        raise ValueError(f"Unsupported head_dim={head_dim}. FLA rotary backend requires <= 256.")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}.")
    if rotary_dim > head_dim:
        raise ValueError(
            f"Rotary dimension ({rotary_dim}) exceeds head_dim ({head_dim})."
        )

    min_offset = int(seqlen_offsets.min().item())
    max_offset = int(seqlen_offsets.max().item())
    if min_offset < 0 or (max_offset + seq_len > tr_len):
        raise ValueError(
            f"Invalid seqlen offsets range [{min_offset}, {max_offset}] for "
            f"seq_len={seq_len} and cache length={tr_len}."
        )

    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    seqlen_offsets = seqlen_offsets.contiguous()

    y = torch.empty_like(x)
    if rotary_dim < head_dim:
        y[..., rotary_dim:].copy_(x[..., rotary_dim:])

    sm_count = _get_multiprocessor_count(x.device)
    bt = min(128, triton.next_power_of_2(max(1, triton.cdiv(seq_len, sm_count))))
    bd = triton.next_power_of_2(rotary_dim)
    grid = (triton.cdiv(seq_len, bt), batch, n_heads)

    _fla_rotary_embedding_kernel[grid](
        x_ptr = x,
        y_ptr = y,
        cos_ptr = cos,
        sin_ptr = sin,
        seqlen_offsets_ptr = seqlen_offsets,
        T = seq_len,
        H = n_heads,
        D = head_dim,
        R = rotary_dim_half,
        TR = tr_len,
        BT = bt,
        BD = bd,
        CONJUGATE = conjugate,
    )
    return y


class _FlaRotaryEmbeddingFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seqlen_offsets: torch.Tensor,
    ) -> torch.Tensor:
        y = _fla_rotary_fwdbwd(
            x = x,
            cos = cos,
            sin = sin,
            seqlen_offsets = seqlen_offsets,
            conjugate = False,
        )
        ctx.save_for_backward(cos, sin, seqlen_offsets)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        cos, sin, seqlen_offsets = ctx.saved_tensors
        dx = _fla_rotary_fwdbwd(
            x = dy,
            cos = cos,
            sin = sin,
            seqlen_offsets = seqlen_offsets,
            conjugate = True,
        )
        return dx, None, None, None


def fla_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int | torch.Tensor | None = None,
) -> torch.Tensor:
    offsets = _normalize_seqlen_offsets(seqlen_offsets, x)
    return _FlaRotaryEmbeddingFunction.apply(
        x,
        cos,
        sin,
        offsets,
    )


class _FlaRotaryEmbeddingQKFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seqlen_offsets: torch.Tensor,
        validate_offsets: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_out, k_out = _fla_rotary_qk_fwdbwd(
            q = q,
            k = k,
            cos = cos,
            sin = sin,
            seqlen_offsets = seqlen_offsets,
            conjugate = False,
            validate_offsets = validate_offsets,
        )
        ctx.save_for_backward(cos, sin, seqlen_offsets)
        ctx.validate_offsets = validate_offsets
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor):
        cos, sin, seqlen_offsets = ctx.saved_tensors
        dq_out, dk_out = _fla_rotary_qk_fwdbwd(
            q = dq,
            k = dk,
            cos = cos,
            sin = sin,
            seqlen_offsets = seqlen_offsets,
            conjugate = True,
            validate_offsets = ctx.validate_offsets,
        )
        return dq_out, dk_out, None, None, None, None


class _FlaRotaryEmbeddingQKPositionsFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids_flat_or_empty: torch.Tensor,
        has_position_ids: bool,
        validate_positions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = position_ids_flat_or_empty if has_position_ids else None
        q_out, k_out = _fla_rotary_qk_positions_fwdbwd(
            q = q,
            k = k,
            cos = cos,
            sin = sin,
            position_ids = position_ids,
            conjugate = False,
            validate_positions = validate_positions,
        )
        ctx.save_for_backward(cos, sin, position_ids_flat_or_empty)
        ctx.has_position_ids = has_position_ids
        ctx.validate_positions = validate_positions
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor):
        cos, sin, position_ids_flat_or_empty = ctx.saved_tensors
        position_ids = position_ids_flat_or_empty if ctx.has_position_ids else None
        dq_out, dk_out = _fla_rotary_qk_positions_fwdbwd(
            q = dq,
            k = dk,
            cos = cos,
            sin = sin,
            position_ids = position_ids,
            conjugate = True,
            validate_positions = ctx.validate_positions,
        )
        return dq_out, dk_out, None, None, None, None, None


def fla_rotary_embedding_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int | torch.Tensor | None = None,
    validate_offsets: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = _normalize_seqlen_offsets(seqlen_offsets, q)
    return _FlaRotaryEmbeddingQKFunction.apply(
        q,
        k,
        cos,
        sin,
        offsets,
        validate_offsets,
    )


def fla_rotary_embedding_qk_positions(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    validate_positions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_ids is None:
        position_ids_flat_or_empty = q.new_empty(1, dtype = torch.int32)
        has_position_ids = False
    else:
        if not isinstance(position_ids, torch.Tensor):
            raise ValueError("position_ids must be a torch.Tensor or None.")
        position_ids_flat_or_empty = position_ids
        has_position_ids = True
    return _FlaRotaryEmbeddingQKPositionsFunction.apply(
        q,
        k,
        cos,
        sin,
        position_ids_flat_or_empty,
        has_position_ids,
        validate_positions,
    )
