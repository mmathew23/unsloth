from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
import trl

try:
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor import DeviceMesh
except (ImportError, AttributeError):
    context_parallel = None
    DeviceMesh = None

from .device_type import DEVICE_TYPE_TORCH
from .utils.packing import mask_packed_sequence_boundaries

_ACTIVE_MANAGER: contextvars.ContextVar[Optional["ContextParallelManager"]] = (
    contextvars.ContextVar("unsloth_active_cp_manager", default = None)
)

_BUFFER_NAMES = (
    "input_ids",
    "attention_mask",
    "labels",
    "position_ids",
    "shift_labels",
)


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def init_distributed(
    *,
    backend: str = "nccl",
    timeout: timedelta = timedelta(minutes=30),
) -> DistInfo:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timeout,
            device_id=device if device.type == "cuda" else None,
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def init_mesh(
    dp: int,
    *,
    tp: int = 1,
    cp: int = 1,
    device_type: str = "cuda",
) -> DeviceMesh:
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("torch.distributed must be initialized before init_mesh().")
    world_size = dist.get_world_size()
    if dp * tp * cp != world_size:
        raise ValueError(
            f"dp*tp*cp must equal world_size "
            f"(dp={dp}, tp={tp}, cp={cp}, world_size={world_size})."
        )
    shape = []
    names = []
    for size, name in ((dp, "dp"), (tp, "tp"), (cp, "cp")):
        if size > 1:
            shape.append(size)
            names.append(name)
    if not shape:
        shape, names = [1], ["dp"]

    from torch.distributed.device_mesh import init_device_mesh
    return init_device_mesh(device_type, tuple(shape), mesh_dim_names=tuple(names))


def get_cp_manager() -> Optional["ContextParallelManager"]:
    return _ACTIVE_MANAGER.get()


@dataclass
class ContextParallelSettings:
    size: int = field(
        default = 1,
        metadata = {
            "help": (
                "Number of ranks that should participate in context parallelism. "
                "Set to >1 only when running under torch.distributed / accelerate."
            )
        },
    )

    @classmethod
    def from_args(cls, args: Optional[object]) -> "ContextParallelSettings":
        if args is None:
            return cls()
        size = int(getattr(args, "context_parallel_size", 1))
        return cls(size = size)


def _attach_context_parallel_attention_hooks(model: torch.nn.Module) -> list:
    """
    Attach forward_pre_hooks to self_attn modules to ensure correct attention behavior
    during context parallelism with load balancing.

    Args:
        model: The model to attach hooks to

    Returns:
        List of hook handles that can be used to remove the hooks later
    """
    handles = []

    def _self_attn_pre_forward_hook(_module, module_args, module_kwargs):
        # Remove attention_mask and set is_causal=True
        # This ensures ring attention uses causal masking correctly
        if "attention_mask" in module_kwargs:
            module_kwargs["attention_mask"] = None
        if "is_causal" in module_kwargs or hasattr(_module, "is_causal"):
            module_kwargs["is_causal"] = True
        return module_args, module_kwargs

    # Find all self_attn modules
    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            handle = module.register_forward_pre_hook(
                _self_attn_pre_forward_hook, with_kwargs = True, prepend = True
            )
            handles.append(handle)

    return handles


class ContextParallelManager:
    """Toggles PyTorch context parallelism.

    Args:
        settings: CP configuration (size, etc.).
        device_mesh: Optional pre-built DeviceMesh (e.g. from Accelerate / FSDP2).
            Must contain a ``"cp"`` dimension whose size equals ``settings.size``.
            When *None*, a ``(dp, cp)`` mesh is created automatically with
            ``dp = world_size // cp_size``.
    """

    def __init__(
        self,
        settings: ContextParallelSettings,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        self.settings = settings
        self._world_size: int = dist.get_world_size()
        self._report_loss: Optional[torch.Tensor] = None
        self._attention_hook_handles: list = []

        if device_mesh is not None:
            # Validate the external mesh has a matching cp dimension.
            names = getattr(device_mesh, "mesh_dim_names", ())
            if "cp" not in names:
                raise ValueError(
                    "Provided device_mesh must contain a 'cp' dimension, "
                    f"got dimensions: {names}"
                )
            cp_dim_size = device_mesh.size(mesh_dim = "cp")
            if cp_dim_size != settings.size:
                raise ValueError(
                    f"device_mesh 'cp' dimension size ({cp_dim_size}) "
                    f"!= settings.size ({settings.size})."
                )
            self._full_mesh: DeviceMesh = device_mesh
        else:
            dp = self._world_size // settings.size
            self._full_mesh = init_mesh(dp, cp = settings.size, device_type = DEVICE_TYPE_TORCH)

        # 1D submesh for PyTorch's context_parallel() API (requires 1D).
        self._cp_mesh: DeviceMesh = self._full_mesh["cp"]
        self._cp_group: dist.ProcessGroup = self._cp_mesh.get_group()

    def attach_attention_hooks(self, model: torch.nn.Module) -> None:
        """
        Attach hooks to self_attn modules to ensure correct attention behavior during
        context parallelism with load balancing.
        """
        if self._attention_hook_handles:
            return
        self._attention_hook_handles = _attach_context_parallel_attention_hooks(model)

    @property
    def device_mesh(self) -> Optional[DeviceMesh]:
        return self._full_mesh

    @property
    def data_parallel_world_size(self) -> int:
        """Product of all non-CP mesh dimensions (handles dp, fsdp, hsdp, tp)."""
        total = 1
        for name in self._full_mesh.mesh_dim_names:
            if name != "cp":
                total *= self._full_mesh.size(mesh_dim = name)
        return total

    @property
    def cp_rank_index(self) -> int:
        return self._full_mesh.get_local_rank(mesh_dim = "cp")

    def data_parallel_rank(self) -> int:
        """Flat rank across all non-CP mesh dimensions.

        For (dp=4, cp=2): returns 0..3.
        For (dp_replicate=2, dp_shard=2, cp=2): returns 0..3
        (i.e. replicate_rank * dp_shard + shard_rank).
        """
        flat_rank = 0
        stride = 1
        for name in reversed(self._full_mesh.mesh_dim_names):
            if name == "cp":
                continue
            local_rank = self._full_mesh.get_local_rank(mesh_dim = name)
            flat_rank += local_rank * stride
            stride *= self._full_mesh.size(mesh_dim = name)
        return flat_rank

    def _collect_buffers(
        self, inputs: dict[str, torch.Tensor]
    ) -> Tuple[list[torch.Tensor], list[int], set[torch.Tensor]]:
        buffers: list[torch.Tensor] = []
        for name in _BUFFER_NAMES:
            tensor = inputs.get(name)
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim <= 1:
                continue
            buffers.append(tensor)
        return buffers, [1] * len(buffers), set(buffers)

    def _ensure_position_ids(self, inputs: dict[str, torch.Tensor]) -> None:
        if "position_ids" in inputs:
            return
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)
        inputs["position_ids"] = positions.unsqueeze(0).expand(input_ids.size(0), -1)

    def _ensure_shift_labels(self, inputs: dict[str, torch.Tensor]) -> None:
        """Pre-shift labels globally before sharding for correct next-token prediction."""
        if "shift_labels" in inputs:
            return
        labels = inputs.get("labels")
        if labels is None:
            return
        # Pad with -100, then take [1:] to get shifted labels
        shift_labels = F.pad(labels, (0, 1), value = -100)[:, 1:].contiguous()
        packed_seq_lengths = inputs.get("packed_seq_lengths")
        if packed_seq_lengths is not None:
            mask_packed_sequence_boundaries(shift_labels, packed_seq_lengths)
        inputs["shift_labels"] = shift_labels

    def _pad_inputs_for_context_parallel(self, inputs: dict[str, torch.Tensor]) -> None:
        """
        Right-pad sequence buffers so PyTorch context_parallel load balancing can shard.

        PyTorch's ring attention path currently expects sequence length to be divisible by
        ``2 * cp_size``. Variable-length packed batches can violate this constraint.
        We pad with ignore / masked values so training semantics stay unchanged.
        """
        multiple = max(int(self.settings.size) * 2, 1)
        if multiple <= 1:
            return

        seq_len = None
        for name in _BUFFER_NAMES:
            tensor = inputs.get(name)
            if isinstance(tensor, torch.Tensor) and tensor.ndim > 1:
                seq_len = int(tensor.size(1))
                break
        if seq_len is None:
            return

        pad_len = (-seq_len) % multiple
        if pad_len == 0:
            return

        def _pad_sequence_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim <= 1:
                return tensor
            if name in ("labels", "shift_labels"):
                return F.pad(tensor, (0, pad_len), value = -100)
            if name == "attention_mask":
                pad_value = False if tensor.dtype == torch.bool else 0
                return F.pad(tensor, (0, pad_len), value = pad_value)
            if name == "position_ids":
                if tensor.size(1) == 0:
                    padding = torch.arange(
                        pad_len, device = tensor.device, dtype = tensor.dtype
                    ).view(1, pad_len)
                else:
                    start = tensor[:, -1:] + 1
                    offsets = torch.arange(
                        pad_len, device = tensor.device, dtype = tensor.dtype
                    ).view(1, pad_len)
                    padding = start + offsets
                return torch.cat([tensor, padding.expand(tensor.size(0), -1)], dim = 1)
            return F.pad(tensor, (0, pad_len), value = 0)

        for name in _BUFFER_NAMES:
            tensor = inputs.get(name)
            if isinstance(tensor, torch.Tensor):
                inputs[name] = _pad_sequence_tensor(name, tensor)

    @contextlib.contextmanager
    def apply(self, inputs: dict[str, torch.Tensor]) -> Iterator[None]:
        """Wrap training step to shard buffers and patch SDPA for ring attention."""
        token = _ACTIVE_MANAGER.set(self)
        self._ensure_position_ids(inputs)
        self._ensure_shift_labels(inputs)
        self._pad_inputs_for_context_parallel(inputs)
        buffers, seq_dims, no_restore = self._collect_buffers(inputs)

        with context_parallel(
            self._cp_mesh,
            buffers = buffers,
            buffer_seq_dims = seq_dims,
            no_restore_buffers = no_restore,
        ):
            yield
        _ACTIVE_MANAGER.reset(token)

    def _set_report_loss(self, value: torch.Tensor) -> None:
        self._report_loss = value.detach() if torch.is_tensor(value) else None

    def consume_report_loss(self) -> Optional[torch.Tensor]:
        value = self._report_loss
        self._report_loss = None
        return value

    def get_global_tokens(self, inputs) -> torch.Tensor:
        """Compute global token count across CP group."""
        shift_labels = inputs.get("shift_labels")
        if shift_labels is None:
            return None

        local_tokens = shift_labels.ne(-100).sum().float()
        global_tokens = local_tokens.clone()
        dist.all_reduce(global_tokens, op = dist.ReduceOp.SUM, group = self._cp_group)
        return global_tokens

    def reduce_loss(self, loss):
        """Reduce loss across CP group for reporting."""
        if self._cp_group is None:
            return loss

        # Handle (loss, outputs) tuple from return_outputs=True
        is_tuple = isinstance(loss, tuple)
        if is_tuple:
            tensor, _ = loss[0], loss[1:]
        else:
            tensor = loss

        # Sum across ranks and divide by cp_size for correct mean.
        global_loss = tensor.detach().clone()
        dist.all_reduce(global_loss, op = dist.ReduceOp.SUM, group = self._cp_group)
        global_loss = global_loss / self.settings.size
        self._set_report_loss(global_loss)

        return loss


def patch_sft_config():
    """Patch SFTConfig to add context_parallel_size and shuffle_dataset fields."""
    base_cls = trl.SFTConfig
    if hasattr(base_cls, "context_parallel_size"):
        return

    @dataclass
    class PatchedSFTConfig(base_cls):  # type: ignore[misc, valid-type]
        context_parallel_size: int = field(
            default = 1,
            metadata = {
                "help": (
                    "Number of ranks participating in context parallelism. "
                    "Set to 1 to disable context parallelism."
                )
            },
        )
        shuffle_dataset: bool = field(
            default = True,
            metadata = {
                "help": (
                    "Whether to shuffle the training dataset before each epoch. "
                    "Exposed for CP = 1 vs. CP > 1 debugging purposes."
                )
            },
        )

    PatchedSFTConfig.__name__ = base_cls.__name__
    PatchedSFTConfig.__qualname__ = base_cls.__qualname__
    PatchedSFTConfig.__module__ = base_cls.__module__
    module = sys.modules.get(base_cls.__module__)
    if module is not None:
        setattr(module, base_cls.__name__, PatchedSFTConfig)
    trl.SFTConfig = PatchedSFTConfig
    if hasattr(trl, "trainer") and hasattr(trl.trainer, "sft_trainer"):
        trl.trainer.sft_trainer.SFTConfig = PatchedSFTConfig


def patch_sft_trainer() -> None:
    """Patch SFTTrainer to add context parallelism support."""
    trainer_cls = trl.SFTTrainer
    if hasattr(trainer_cls, "__unsloth_context_parallel__"):
        return

    original_init = trainer_cls.__init__
    original_compute_loss = trainer_cls.compute_loss
    original_prediction_step = trainer_cls.prediction_step
    original_training_step = trainer_cls.training_step
    original_get_train_sampler = getattr(trainer_cls, "_get_train_sampler", None)

    def _patch_train_sampler(original_fn):
        @functools.wraps(original_fn)
        def wrapper(self, *args, **kwargs):
            sampler = original_fn(self, *args, **kwargs)
            manager = getattr(self, "_context_parallel_manager", None)
            dataset = args[0] if args else None
            if dataset is None:
                dataset = getattr(self, "train_dataset", None)
            shuffle_dataset = getattr(self.args, "shuffle_dataset", True)
            if (
                manager
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and dataset is not None
            ):
                dp_world = manager.data_parallel_world_size
                world_size = torch.distributed.get_world_size()
                if dp_world != world_size:
                    try:
                        from torch.utils.data.distributed import DistributedSampler
                    except ImportError:
                        return sampler
                    dp_rank = manager.data_parallel_rank()
                    shuffle = shuffle_dataset and not getattr(
                        self.args, "group_by_length", False
                    )
                    return DistributedSampler(
                        dataset,
                        num_replicas = dp_world,
                        rank = dp_rank,
                        shuffle = shuffle,
                        drop_last = getattr(self.args, "dataloader_drop_last", False),
                    )
            if not shuffle_dataset and dataset is not None:
                try:
                    from torch.utils.data import SequentialSampler
                except ImportError:
                    return sampler
                return SequentialSampler(dataset)
            return sampler

        return wrapper

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        settings = ContextParallelSettings.from_args(getattr(self, "args", None))
        if settings.size > 1:
            if context_parallel is None or DeviceMesh is None:
                warnings.warn(
                    "Context parallelism requested but PyTorch >= 2.7 is required.",
                    stacklevel = 2,
                )
                self._context_parallel_manager = None
            elif not dist.is_initialized():
                raise RuntimeError(
                    f"Context parallelism requires torch.distributed to be initialized. "
                    f"Use torchrun or accelerate launch with {settings.size} processes."
                )
            elif dist.get_world_size() < settings.size:
                raise RuntimeError(
                    f"Context parallelism size ({settings.size}) exceeds world size "
                    f"({dist.get_world_size()}). Launch with at least {settings.size} processes."
                )
            else:
                # Check if Accelerate already provides a mesh (e.g. from FSDP2).
                # If it has a "cp" dimension, reuse it so all parallelism
                # strategies share the same topology.
                accelerator = getattr(self, "accelerator", None)
                existing_mesh = (
                    getattr(accelerator, "torch_device_mesh", None)
                    if accelerator is not None
                    else None
                )
                has_cp_dim = "cp" in getattr(existing_mesh, "mesh_dim_names", ())
                external_mesh = existing_mesh if has_cp_dim else None
                self._context_parallel_manager = ContextParallelManager(
                    settings, device_mesh = external_mesh,
                )
        else:
            self._context_parallel_manager = None

        accelerator = getattr(self, "accelerator", None)
        manager = self._context_parallel_manager
        if manager:
            print(
                f"Unsloth: Context parallelism enabled with size={manager.settings.size}"
            )

        # Publish our mesh to Accelerate so DDP / FSDP2 can see the
        # full topology.  Skip if Accelerate already has one with a cp dim
        # (we reused it above).
        if accelerator is not None and manager is not None:
            existing_mesh = getattr(accelerator, "torch_device_mesh", None)
            has_cp_dim = "cp" in getattr(existing_mesh, "mesh_dim_names", ())
            if not has_cp_dim:
                setattr(accelerator.state, "device_mesh", manager.device_mesh)

        # Enable sync_each_batch when using CP to ensure consistent computation graph
        # for DDP + static_graph mode. This is needed because ring attention changes
        # the graph structure, and sync_each_batch keeps it constant.
        if (
            manager
            and accelerator is not None
            and hasattr(accelerator, "gradient_state")
        ):
            accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True

        # Attach attention hooks for proper ring attention behavior with load balancing.
        # This ensures attention_mask is removed and is_causal=True for all self_attn calls.
        if manager:
            model = getattr(self, "model", None)
            if model is not None:
                manager.attach_attention_hooks(model)

    @functools.wraps(original_compute_loss)
    def patched_compute_loss(self, model, inputs, return_outputs = False, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)

        # Check if we're in CP mode with pre-shifted labels
        shift_labels = inputs.get("shift_labels")
        use_cp = manager and isinstance(shift_labels, torch.Tensor)

        if use_cp:
            global_tokens = manager.get_global_tokens(inputs)

            # For CP, pass both labels and shift_labels to the model.
            # The model's forward (e.g., llama.py) detects shift_labels and uses
            # unsloth_fused_ce_loss with the pre-shifted labels.
            # We keep labels so the fused CE loss path is triggered (requires labels is not None).

            # Scale by GA and cp_size to match gradient behavior of CP=1.
            ga_steps = getattr(self.args, "gradient_accumulation_steps", 1)
            cp_size = manager.settings.size
            inputs["num_items_in_batch"] = global_tokens * ga_steps / cp_size

            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            # Clean up temporary input
            inputs.pop("num_items_in_batch", None)

            if return_outputs:
                loss = (loss, outputs)
        else:
            # No CP - use original path
            loss = original_compute_loss(
                self,
                model,
                inputs,
                return_outputs = return_outputs,
                **kwargs,
            )

        if manager:
            loss = manager.reduce_loss(loss)
        return loss

    @functools.wraps(original_prediction_step)
    def patched_prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
        **kwargs,
    ):
        manager = getattr(self, "_context_parallel_manager", None)
        context = manager.apply(inputs) if manager else contextlib.nullcontext()
        with context:
            return original_prediction_step(
                self,
                model,
                inputs,
                prediction_loss_only,
                ignore_keys,
                **kwargs,
            )

    def _maybe_enable_sync_each_batch(trainer):
        """Enable sync_each_batch at runtime if gradient checkpointing is detected."""
        if getattr(trainer, "_sync_each_batch_checked", False):
            return
        setattr(trainer, "_sync_each_batch_checked", True)

        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is None or not hasattr(accelerator, "gradient_state"):
            return

        # Check if already enabled
        if accelerator.gradient_state.plugin_kwargs.get("sync_each_batch", False):
            return

        model = getattr(trainer, "model", None)
        is_checkpointing = getattr(model, "is_gradient_checkpointing", False)
        grad_accum_steps = getattr(trainer.args, "gradient_accumulation_steps", 1)

        if is_checkpointing and grad_accum_steps > 1:
            accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True

    @functools.wraps(original_training_step)
    def patched_training_step(self, model, inputs, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        original_n_gpu = getattr(self.args, "n_gpu", 1)
        if manager:
            setattr(self.args, "_n_gpu", manager.data_parallel_world_size)
            _maybe_enable_sync_each_batch(self)
            # Attach attention hooks if not already done (model may not be ready at init)
            if not manager._attention_hook_handles:
                m = getattr(self, "model", None)
                if m is not None:
                    manager.attach_attention_hooks(m)

        # Wrap entire training step (forward + backward) in context_parallel
        # This keeps SDPA patched and buffers sharded throughout, including
        # during gradient checkpoint recomputation in backward pass.
        cp_context = manager.apply(inputs) if manager else contextlib.nullcontext()
        with cp_context:
            loss = original_training_step(self, model, inputs, *args, **kwargs)

        if manager:
            setattr(self.args, "_n_gpu", original_n_gpu)

        report_loss = manager.consume_report_loss() if manager else None
        if report_loss is not None:
            return report_loss
        return loss

    trainer_cls.__init__ = patched_init
    trainer_cls.compute_loss = patched_compute_loss
    trainer_cls.prediction_step = patched_prediction_step
    trainer_cls.training_step = patched_training_step
    trainer_cls.__unsloth_context_parallel__ = True
    if original_get_train_sampler is not None:
        trainer_cls._get_train_sampler = _patch_train_sampler(
            original_get_train_sampler
        )
