"""Tests for graceful best-model selection in communicators.

Communicators with ``best_model=True`` (the default) used to crash with a
ValueError when no model carried the ``elpd_waic`` diagnostic (e.g. the user
disabled the ``waic`` checker). They now warn and fall back to running over
all models, and silently use the only model when exactly one is fitted.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import arviz as az
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hibayes.analysis_state import AnalysisState, ModelAnalysisState
from hibayes.communicate.plots.plots import forest_plot
from hibayes.communicate.tables.tables import summary_table
from hibayes.communicate.utils import resolve_models_to_run
from hibayes.model import ModelConfig
from hibayes.ui import ModellingDisplay


def _dummy_model(features):
    pass


def _make_inference_data(seed: int = 0) -> az.InferenceData:
    rng = np.random.default_rng(seed)
    posterior = xr.Dataset({"theta": (["chain", "draw"], rng.normal(size=(2, 50)))})
    return az.InferenceData(posterior=posterior)


def _model_state(
    tag: str | None = None,
    diagnostics: dict | None = None,
    is_fitted: bool = True,
) -> ModelAnalysisState:
    return ModelAnalysisState(
        model=_dummy_model,
        model_config=ModelConfig(tag=tag),
        inference_data=_make_inference_data(),
        diagnostics=diagnostics,
        is_fitted=is_fitted,
    )


def _analysis_state(models: list[ModelAnalysisState]) -> AnalysisState:
    state = AnalysisState(data=pd.DataFrame({"score": [0.1, 0.2]}))
    for model in models:
        state.add_model(model)
    return state


@pytest.fixture
def mock_display() -> ModellingDisplay:
    display = MagicMock(spec=ModellingDisplay)
    display.logger = MagicMock()
    display.is_live = False
    return display


class TestResolveModelsToRun:
    def test_best_model_false_returns_all_models(self, mock_display):
        state = _analysis_state([_model_state(tag="a"), _model_state(tag="b")])
        models = resolve_models_to_run(state, best_model=False, display=mock_display)
        assert models == state.models

    def test_best_model_with_diagnostic_returns_best(self, mock_display):
        state = _analysis_state(
            [
                _model_state(tag="worse", diagnostics={"elpd_waic": -100.0}),
                _model_state(tag="better", diagnostics={"elpd_waic": -10.0}),
            ]
        )
        models = resolve_models_to_run(state, best_model=True, display=mock_display)
        assert len(models) == 1
        assert models[0].model_name == "_dummy_model_better"
        mock_display.logger.warning.assert_not_called()

    def test_single_fitted_model_used_without_warning(self, mock_display):
        """Best of one is unambiguous — no diagnostic needed, no warning."""
        state = _analysis_state([_model_state(tag="only")])
        models = resolve_models_to_run(state, best_model=True, display=mock_display)
        assert len(models) == 1
        assert models[0].model_name == "_dummy_model_only"
        mock_display.logger.warning.assert_not_called()

    def test_missing_diagnostic_warns_and_falls_back_to_all(self, mock_display):
        state = _analysis_state([_model_state(tag="a"), _model_state(tag="b")])
        models = resolve_models_to_run(state, best_model=True, display=mock_display)
        assert models == state.models
        mock_display.logger.warning.assert_called_once()
        warning = mock_display.logger.warning.call_args[0][0]
        assert "elpd_waic" in warning
        assert "best_model" in warning

    def test_missing_diagnostic_no_display_does_not_raise(self):
        state = _analysis_state([_model_state(tag="a"), _model_state(tag="b")])
        models = resolve_models_to_run(state, best_model=True, display=None)
        assert models == state.models


class TestCommunicatorsGracefulBestModel:
    def test_forest_plot_does_not_crash_without_waic(self, mock_display):
        """forest_plot with default best_model=True must not raise when no
        model has the elpd_waic diagnostic; it runs over all models."""
        state = _analysis_state([_model_state(tag="a"), _model_state(tag="b")])
        communicator = forest_plot()

        result_state, outcome = communicator(state, display=mock_display)

        assert outcome == "pass"
        plot_names = list(result_state.communicate.keys())
        assert any("_dummy_model_a" in name for name in plot_names)
        assert any("_dummy_model_b" in name for name in plot_names)
        mock_display.logger.warning.assert_called_once()

    def test_forest_plot_single_model_no_warning(self, mock_display):
        state = _analysis_state([_model_state(tag="only")])
        communicator = forest_plot()

        result_state, outcome = communicator(state, display=mock_display)

        assert outcome == "pass"
        assert any(
            "_dummy_model_only" in name for name in result_state.communicate.keys()
        )
        mock_display.logger.warning.assert_not_called()

    def test_summary_table_does_not_crash_without_waic(self, mock_display):
        state = _analysis_state([_model_state(tag="a"), _model_state(tag="b")])
        communicator = summary_table()

        result_state, outcome = communicator(state, display=mock_display)

        assert outcome == "pass"
        table_names = list(result_state.communicate.keys())
        assert "model__dummy_model_a_summary" in table_names
        assert "model__dummy_model_b_summary" in table_names
        mock_display.logger.warning.assert_called_once()

    def test_summary_table_picks_best_model_when_available(self, mock_display):
        state = _analysis_state(
            [
                _model_state(tag="worse", diagnostics={"elpd_waic": -100.0}),
                _model_state(tag="better", diagnostics={"elpd_waic": -10.0}),
            ]
        )
        communicator = summary_table()

        result_state, outcome = communicator(state, display=mock_display)

        assert outcome == "pass"
        table_names = list(result_state.communicate.keys())
        assert "model__dummy_model_better_summary" in table_names
        assert "model__dummy_model_worse_summary" not in table_names
