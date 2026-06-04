# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pin Studio's behavior when a training event reports non-finite (NaN/Inf) loss.

Background: a NaN loss event would previously be filtered to None by the
sanitizer and never propagated as an error. The run would silently keep
"completing" with an unchanging stale loss, and the final API state would
report phase=completed with no error — pointing the user at a corrupt adapter.

This test pins the corrected behavior: the first NaN loss event marks the
run as failed (is_training=False, error set, _nonfinite_loss_reported=True,
_should_stop=True), and subsequent NaN events do not overwrite that state.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.training import TrainingBackend


def _make_backend() -> TrainingBackend:
    """Construct a bare TrainingBackend without launching any subprocess."""
    return TrainingBackend()


def _progress_event(step: int, loss: float, lr: float = 1e-4) -> dict:
    """Shape of an event the training subprocess emits via _event_queue."""
    return {
        "type": "progress",
        "step": step,
        "loss": loss,
        "learning_rate": lr,
        "epoch": 0.0,
        "total_steps": 100,
    }


class TestNonfiniteLossMarksRunFailed:
    def test_finite_loss_does_not_mark_failed(self):
        b = _make_backend()
        b._handle_event(_progress_event(step=1, loss=0.97))
        assert b._progress.loss == pytest.approx(0.97)
        assert b._progress.error is None
        assert b._should_stop is False
        assert getattr(b._progress, "_nonfinite_loss_reported", False) is False

    def test_nan_loss_sets_error_and_stops(self):
        b = _make_backend()
        b._handle_event(_progress_event(step=1, loss=0.97))
        b._handle_event(_progress_event(step=2, loss=float("nan")))
        assert b._progress.error is not None
        assert "non-finite" in b._progress.error.lower()
        assert "step 2" in b._progress.error
        assert b._progress.is_training is False
        assert b._should_stop is True
        assert b._progress._nonfinite_loss_reported is True

    def test_inf_loss_also_marks_failed(self):
        b = _make_backend()
        b._handle_event(_progress_event(step=1, loss=float("inf")))
        assert b._progress.error is not None
        assert "non-finite" in b._progress.error.lower()
        assert b._should_stop is True

    def test_negative_inf_loss_also_marks_failed(self):
        b = _make_backend()
        b._handle_event(_progress_event(step=1, loss=float("-inf")))
        assert b._progress.error is not None
        assert b._should_stop is True

    def test_repeated_nan_does_not_overwrite_first_step(self):
        """Once a run is marked failed at step N, subsequent NaN steps should not
        re-trigger error setup or change the recorded step."""
        b = _make_backend()
        b._handle_event(_progress_event(step=1, loss=0.97))
        b._handle_event(_progress_event(step=2, loss=float("nan")))
        first_error = b._progress.error
        b._handle_event(_progress_event(step=3, loss=float("nan")))
        b._handle_event(_progress_event(step=4, loss=float("nan")))
        assert b._progress.error == first_error
        assert "step 2" in b._progress.error
        assert b._progress._nonfinite_loss_reported is True

    def test_nan_loss_does_not_corrupt_progress_loss_field(self):
        """The progress.loss field stays at the last finite value, not NaN.
        Keeps API responses JSON-safe (JSON doesn't allow NaN literal)."""
        b = _make_backend()
        b._handle_event(_progress_event(step=1, loss=0.97))
        b._handle_event(_progress_event(step=2, loss=float("nan")))
        assert math.isfinite(b._progress.loss)
        assert b._progress.loss == pytest.approx(0.97)
