"""Tests for interactive checker behaviour in unattended runs.

``prior_predictive_plot`` and ``posterior_predictive_plot`` default to
``interactive=True`` and prompt for user approval. When no interactive session
is available (``display.can_prompt()`` is False, e.g. no TTY), they must not
block: they warn, skip the prompt, and return "NA".
"""

from __future__ import annotations

from unittest.mock import MagicMock

import arviz as az
import matplotlib

matplotlib.use("Agg")

import numpy as np
import xarray as xr

from hibayes.analysis_state import ModelAnalysisState
from hibayes.check.checkers import posterior_predictive_plot, prior_predictive_plot
from hibayes.model import ModelConfig
from hibayes.ui import ModellingDisplay


def _dummy_model(features):
    pass


class FakeDisplay(ModellingDisplay):
    """Real display satisfying the Display protocol, with prompting stubbed.

    MagicMock(spec=ModellingDisplay) does not pass the runtime protocol
    isinstance check enforced by the @checker wrapper, so we subclass instead.
    """

    def __init__(self, can_prompt: bool = True, prompt_response: bool = True):
        super().__init__()
        self._can_prompt_value = can_prompt
        self._prompt_response = prompt_response
        self.prompt_calls: list[str] = []
        self.logger = MagicMock()

    def can_prompt(self) -> bool:
        return self._can_prompt_value

    def prompt_user(self, question="Would you like to proceed?", options=None):
        self.prompt_calls.append(question)
        return self._prompt_response

    def add_plot(self, *args, **kwargs):
        pass


def _make_display(can_prompt: bool) -> FakeDisplay:
    return FakeDisplay(can_prompt=can_prompt)


def _prior_state() -> ModelAnalysisState:
    """State with a prior_predictive group so the checker skips sampling."""
    rng = np.random.default_rng(0)
    prior_predictive = xr.Dataset(
        {"obs": (["chain", "draw", "obs_dim_0"], rng.normal(size=(1, 100, 20)))}
    )
    return ModelAnalysisState(
        model=_dummy_model,
        model_config=ModelConfig(),
        inference_data=az.InferenceData(prior_predictive=prior_predictive),
    )


def _posterior_state() -> ModelAnalysisState:
    """State with posterior + posterior_predictive so the checker skips sampling."""
    rng = np.random.default_rng(0)
    n_obs = 20
    idata = az.from_dict(
        posterior={"theta": rng.normal(size=(2, 100))},
        posterior_predictive={"obs": rng.normal(size=(2, 100, n_obs))},
        observed_data={"obs": rng.normal(size=n_obs)},
        coords={"obs_dim_0": np.arange(n_obs)},
        dims={"obs": ["obs_dim_0"]},
    )
    return ModelAnalysisState(
        model=_dummy_model,
        model_config=ModelConfig(),
        inference_data=idata,
        is_fitted=True,
    )


class TestPriorPredictivePlotInteractive:
    def test_no_interactive_session_returns_na_and_warns(self):
        display = _make_display(can_prompt=False)
        state = _prior_state()

        state, result = prior_predictive_plot()(state, display)

        assert result == "NA"
        assert display.prompt_calls == []
        display.logger.warning.assert_called()
        warning = display.logger.warning.call_args_list[0][0][0]
        assert "interactive" in warning.lower()

    def test_interactive_session_prompts_user(self):
        display = _make_display(can_prompt=True)
        state = _prior_state()

        state, result = prior_predictive_plot()(state, display)

        assert result == "pass"
        assert len(display.prompt_calls) > 0

    def test_interactive_false_unchanged(self):
        """interactive=False keeps its existing behaviour (no prompt, no
        interactivity warning)."""
        display = _make_display(can_prompt=False)
        state = _prior_state()

        state, result = prior_predictive_plot(interactive=False)(state, display)

        assert result == "pass"
        assert display.prompt_calls == []
        display.logger.warning.assert_not_called()

    def test_plots_still_saved_when_prompt_unavailable(self):
        """The diagnostic figures are still generated for offline review."""
        display = _make_display(can_prompt=False)
        state = _prior_state()

        state, result = prior_predictive_plot()(state, display)

        assert result == "NA"
        assert "obs_prior_predictive" in state.diagnostics


class TestPosteriorPredictivePlotInteractive:
    def test_no_interactive_session_returns_na_and_warns(self):
        display = _make_display(can_prompt=False)
        state = _posterior_state()

        state, result = posterior_predictive_plot()(state, display)

        assert result == "NA"
        assert display.prompt_calls == []
        display.logger.warning.assert_called()
        warning = display.logger.warning.call_args_list[0][0][0]
        assert "interactive" in warning.lower()

    def test_interactive_session_prompts_user(self):
        display = _make_display(can_prompt=True)
        state = _posterior_state()

        state, result = posterior_predictive_plot()(state, display)

        assert result == "pass"
        assert len(display.prompt_calls) == 1

    def test_interactive_false_unchanged(self):
        display = _make_display(can_prompt=False)
        state = _posterior_state()

        state, result = posterior_predictive_plot(interactive=False)(state, display)

        assert result == "NA"
        assert display.prompt_calls == []
        display.logger.warning.assert_not_called()


class TestDisplayCanPrompt:
    def test_modelling_display_can_prompt_reflects_stdin_tty(self, monkeypatch):
        import sys

        display = ModellingDisplay()

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        assert display.can_prompt() is False

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        assert display.can_prompt() is True
