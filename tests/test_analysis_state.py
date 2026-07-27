"""Tests for ModelAnalysisState naming and AnalysisState.get_best_model."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hibayes.analysis_state import AnalysisState, ModelAnalysisState
from hibayes.model import ModelConfig


def ordered_logistic_model(features):
    """Picklable stub model — name mirrors the docs example."""
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
        model=ordered_logistic_model,
        model_config=ModelConfig(tag=tag),
        inference_data=_make_inference_data(),
        diagnostics=diagnostics,
        is_fitted=is_fitted,
    )


class TestModelName:
    def test_model_name_without_tag(self):
        state = _model_state()
        assert state.model_name == "ordered_logistic_model"

    def test_model_name_with_tag_uses_underscore_separator(self):
        """Regression: tags used to be concatenated with no separator,
        producing names like 'ordered_logistic_modelwith_interaction'."""
        state = _model_state(tag="with_interaction")
        assert state.model_name == "ordered_logistic_model_with_interaction"

    def test_model_state_save_load_round_trip_keeps_name(self, tmp_path: Path):
        state = _model_state(tag="v1")
        state.save(tmp_path / state.model_name)

        loaded = ModelAnalysisState.load(tmp_path / state.model_name)
        assert loaded.model_name == "ordered_logistic_model_v1"

    def test_analysis_state_save_load_round_trip(self, tmp_path: Path):
        data = pd.DataFrame({"score": [0.1, 0.2]})
        analysis_state = AnalysisState(data=data)
        analysis_state.add_model(_model_state(tag="v1"))

        analysis_state.save(tmp_path)
        loaded = AnalysisState.load(tmp_path)

        assert [m.model_name for m in loaded.models] == ["ordered_logistic_model_v1"]
        # get_model looks the model up by the model_name property
        assert loaded.get_model("ordered_logistic_model_v1") is not None


class TestGetBestModel:
    def _state_with_models(self) -> AnalysisState:
        data = pd.DataFrame({"score": [0.1, 0.2]})
        analysis_state = AnalysisState(data=data)
        analysis_state.add_model(
            _model_state(tag="low", diagnostics={"elpd_waic": -100.0})
        )
        analysis_state.add_model(
            _model_state(tag="high", diagnostics={"elpd_waic": -10.0})
        )
        return analysis_state

    def test_default_returns_maximum(self):
        """By default the model with the highest diagnostic wins (elpd_waic:
        higher is better)."""
        analysis_state = self._state_with_models()
        best = analysis_state.get_best_model()
        assert best.model_name == "ordered_logistic_model_high"

    def test_minimum_true_returns_minimum(self):
        analysis_state = self._state_with_models()
        best = analysis_state.get_best_model(minimum=True)
        assert best.model_name == "ordered_logistic_model_low"

    def test_raises_when_no_model_has_diagnostic(self):
        data = pd.DataFrame({"score": [0.1, 0.2]})
        analysis_state = AnalysisState(data=data)
        analysis_state.add_model(_model_state(tag="v1"))

        with pytest.raises(ValueError, match="elpd_waic"):
            analysis_state.get_best_model()
