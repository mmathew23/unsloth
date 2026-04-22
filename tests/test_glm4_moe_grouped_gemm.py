import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for repo in (ROOT / "unsloth", ROOT / "unsloth-zoo"):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_SKIP_MODEL_IMPORTS", "1")

from unsloth.kernels import clear_kernel_backend_overrides, kernel_backend_context
from unsloth.kernels.grouped_gemm import grouped_gemm as grouped_gemm_kernel
from unsloth.models import glm4_moe
from unsloth.models import loader as model_loader


class _FakeNaiveMoe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.act_fn = F.silu
        self.gate_up_proj = torch.nn.Parameter(
            torch.tensor(
                [
                    [
                        [0.50, -0.20, 0.30],
                        [-0.40, 0.10, 0.60],
                        [0.70, 0.20, -0.10],
                        [0.20, 0.30, 0.40],
                    ],
                    [
                        [-0.30, 0.40, 0.20],
                        [0.60, -0.50, 0.10],
                        [0.10, 0.80, -0.60],
                        [0.50, -0.10, 0.20],
                    ],
                ],
                dtype = torch.float32,
            )
        )
        self.down_proj = torch.nn.Parameter(
            torch.tensor(
                [
                    [
                        [0.30, -0.20],
                        [0.50, 0.10],
                        [-0.40, 0.60],
                    ],
                    [
                        [0.20, 0.70],
                        [-0.60, 0.30],
                        [0.10, -0.50],
                    ],
                ],
                dtype = torch.float32,
            )
        )


class Glm4MoeGroupedGemmTests(unittest.TestCase):
    def tearDown(self):
        clear_kernel_backend_overrides()

    def _manual_forward(self, moe, hidden_states, top_k_index, top_k_weights):
        outputs = []
        for token_idx in range(hidden_states.shape[0]):
            token_hidden = hidden_states[token_idx]
            token_output = torch.zeros_like(token_hidden)
            for route_idx in range(top_k_index.shape[1]):
                expert_idx = int(top_k_index[token_idx, route_idx])
                gate_up = F.linear(token_hidden, moe.gate_up_proj[expert_idx])
                gate, up = gate_up.chunk(2, dim = -1)
                intermediate = moe.act_fn(gate) * up
                expert_output = F.linear(intermediate, moe.down_proj[expert_idx])
                token_output = (
                    token_output
                    + expert_output * top_k_weights[token_idx, route_idx].to(token_hidden.dtype)
                )
            outputs.append(token_output)
        return torch.stack(outputs, dim = 0)

    def test_glm4_naive_moe_fast_forward_uses_pluggable_grouped_gemm(self):
        moe = _FakeNaiveMoe()
        hidden_states = torch.tensor(
            [[0.40, -0.70, 0.10], [0.20, 0.30, -0.50]],
            dtype = torch.float32,
            requires_grad = True,
        )
        top_k_index = torch.tensor([[0, 1], [1, 0]], dtype = torch.long)
        top_k_weights = torch.tensor([[0.75, 0.25], [0.60, 0.40]], dtype = torch.float32)
        expected = self._manual_forward(moe, hidden_states, top_k_index, top_k_weights)

        with patch.object(glm4_moe, "grouped_gemm", wraps = grouped_gemm_kernel) as grouped_gemm_spy:
            with kernel_backend_context(global_backend = "eager"):
                actual = glm4_moe.Glm4MoeLiteNaiveMoe_fast_forward(
                    moe,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(grouped_gemm_spy.call_count, 2)
        self.assertTrue(grouped_gemm_spy.call_args_list[0].kwargs["permute_x"])
        self.assertFalse(grouped_gemm_spy.call_args_list[0].kwargs["permute_y"])
        self.assertFalse(grouped_gemm_spy.call_args_list[1].kwargs["permute_x"])
        self.assertTrue(grouped_gemm_spy.call_args_list[1].kwargs["permute_y"])

        loss = actual.square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(hidden_states.grad).all())
        self.assertTrue(torch.isfinite(moe.gate_up_proj.grad).all())
        self.assertTrue(torch.isfinite(moe.down_proj.grad).all())

    def test_full_glm4_naive_moe_fast_forward_matches_reference(self):
        moe = _FakeNaiveMoe()
        hidden_states = torch.tensor(
            [[0.15, -0.10, 0.25], [-0.20, 0.45, 0.05]],
            dtype = torch.float32,
            requires_grad = True,
        )
        top_k_index = torch.tensor([[1, 0], [0, 1]], dtype = torch.long)
        top_k_weights = torch.tensor([[0.55, 0.45], [0.35, 0.65]], dtype = torch.float32)
        expected = self._manual_forward(moe, hidden_states, top_k_index, top_k_weights)

        with patch.object(glm4_moe, "grouped_gemm", wraps = grouped_gemm_kernel) as grouped_gemm_spy:
            with kernel_backend_context(global_backend = "eager"):
                actual = glm4_moe.Glm4MoeNaiveMoe_fast_forward(
                    moe,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(grouped_gemm_spy.call_count, 2)

    def test_full_glm4_moe_fast_forward_adds_shared_expert_and_restores_shape(self):
        class _FakeGate(torch.nn.Module):
            def forward(self, hidden_states):
                return torch.zeros(hidden_states.shape[0], 2, dtype = hidden_states.dtype)

        class _FakeSharedExperts(torch.nn.Module):
            def forward(self, hidden_states):
                return hidden_states * 0.25 + 0.1

        class _FakeFullMoe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = _FakeGate()
                self.top_k = 2
                self.n_routed_experts = 2
                self.experts = _FakeNaiveMoe()
                self.shared_experts = _FakeSharedExperts()

            def route_tokens_to_experts(self, router_logits):
                del router_logits
                topk_indices = torch.tensor(
                    [[0, 1], [1, 0], [0, 1], [1, 0]],
                    dtype = torch.long,
                )
                topk_weights = torch.tensor(
                    [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.55, 0.45]],
                    dtype = torch.float32,
                )
                return topk_indices, topk_weights

        moe = _FakeFullMoe()
        hidden_states = torch.tensor(
            [
                [[0.4, -0.7, 0.1], [0.2, 0.3, -0.5]],
                [[-0.1, 0.25, 0.35], [0.6, -0.2, 0.4]],
            ],
            dtype = torch.float32,
            requires_grad = True,
        )
        flat_hidden = hidden_states.detach().clone().view(-1, hidden_states.shape[-1])
        topk_indices, topk_weights = moe.route_tokens_to_experts(None)
        expected_routed = self._manual_forward(
            moe.experts,
            flat_hidden,
            topk_indices,
            topk_weights,
        )
        expected = expected_routed.view_as(hidden_states) + moe.shared_experts(
            hidden_states.detach().clone()
        )

        with patch.object(glm4_moe, "grouped_gemm", wraps = grouped_gemm_kernel) as grouped_gemm_spy:
            with kernel_backend_context(global_backend = "eager"):
                actual = glm4_moe.Glm4MoeMoE_fast_forward(moe, hidden_states)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, hidden_states.shape)
        self.assertEqual(grouped_gemm_spy.call_count, 2)

        actual.sum().backward()
        self.assertTrue(torch.isfinite(hidden_states.grad).all())

    def test_fast_language_model_dispatches_full_glm4_moe(self):
        model = SimpleNamespace(config = SimpleNamespace())
        tokenizer = SimpleNamespace()

        with patch.object(model_loader, "hf_login", side_effect = lambda token: token):
            with patch.object(
                model_loader.AutoConfig,
                "from_pretrained",
                return_value = SimpleNamespace(model_type = "glm4_moe"),
            ):
                with patch.object(
                    model_loader.PeftConfig,
                    "from_pretrained",
                    side_effect = OSError("no adapter config"),
                ):
                    with patch.object(
                        model_loader,
                        "get_transformers_model_type",
                        return_value = ["glm4_moe"],
                    ):
                        with patch.object(
                            model_loader,
                            "apply_unsloth_gradient_checkpointing",
                            side_effect = lambda setting, *_args: setting,
                        ):
                            with patch.object(
                                model_loader,
                                "_fix_rope_inv_freq",
                                side_effect = lambda loaded_model: loaded_model,
                            ):
                                with patch.object(
                                    model_loader.FastGLM4MoeModel,
                                    "from_pretrained",
                                    return_value = (model, tokenizer),
                                ) as dispatch_spy:
                                    actual_model, actual_tokenizer = (
                                        model_loader.FastLanguageModel.from_pretrained(
                                            model_name = "tiny-random/glm-4-moe",
                                            dtype = torch.bfloat16,
                                            load_in_4bit = False,
                                            load_in_16bit = True,
                                            use_exact_model_name = True,
                                        )
                                    )

        self.assertIs(actual_model, model)
        self.assertIs(actual_tokenizer, tokenizer)
        self.assertEqual(dispatch_spy.call_count, 1)
        self.assertEqual(
            dispatch_spy.call_args.kwargs["model_name"],
            "tiny-random/glm-4-moe",
        )
        self.assertEqual(
            dispatch_spy.call_args.kwargs["model_patcher"],
            model_loader.FastGLM4MoeModel,
        )
        self.assertEqual(dispatch_spy.call_args.kwargs["dtype"], torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
