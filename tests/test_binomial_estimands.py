import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import expit, logit

from hibayes.analysis_state import AnalysisState, ModelAnalysisState
from hibayes.communicate import binomial_estimands_table, group_binomial_estimands
from hibayes.communicate.communicate_config import CommunicateConfig
from hibayes.communicate.tables.binomial import _population_probability
from hibayes.model import ModelConfig
from hibayes.model.models import (
    simplified_group_binomial_exponential,
    two_level_group_binomial,
)


def fitted(builder=two_level_group_binomial, tag=None):
    return ModelAnalysisState(
        model=builder(),
        model_config=ModelConfig(tag=tag),
        is_fitted=True,
        features={
            "obs": np.array([0, 1, 0]),
            "group_index": np.array([0, 1, 1]),
            "n_total": np.array([10, 20, 70]),
            "num_group": 3,
        },
        inference_data=az.from_dict(
            posterior={
                "overall_mean": np.full((2, 10), -2.0),
                "sigma_group": np.full((2, 10), 3.0),
                "group_effects": np.broadcast_to(logit([0.1, 0.9, 0.99]), (2, 10, 3)),
            }
        ),
    )


@pytest.mark.parametrize(
    "builder", [simplified_group_binomial_exponential, two_level_group_binomial]
)
def test_distinguishes_evaluated_groups_trial_weights_and_population(builder):
    result = group_binomial_estimands(fitted(builder))
    # Repeated rows count once for equal group weights; the unobserved third
    # group must not enter either evaluated-data estimand.
    np.testing.assert_allclose(result.evaluated_group_mean_probability, 0.5)
    np.testing.assert_allclose(result.evaluated_trial_mean_probability, 0.82)
    np.testing.assert_allclose(result.median_group_probability, expit(-2))
    assert (result.population_mean_probability > 0.25).all()
    assert (result.population_mean_probability < 0.30).all()


@pytest.mark.parametrize(
    "mu,sigma",
    [
        (-10.0, 0.0),
        (-2.0, 0.1),
        (-2.0, 3.0),
        (2.0, 20.0),
        (-10.0, 100.0),
        (0.0, 1000.0),
    ],
)
def test_population_integral_matches_independent_probability_scale_integral(mu, sigma):
    # Independent integration over uniform quantiles of the normal, rather
    # than standardized normal density; stresses large-sigma behavior.
    from scipy.special import ndtri

    expected, _ = quad(lambda u: expit(mu + sigma * ndtri(u)), 0, 1, epsabs=1e-10)
    actual = _population_probability(np.array([[mu]]), np.array([[sigma]]))
    np.testing.assert_allclose(actual, expected, atol=2e-8)


def test_population_draws_preserve_alignment_across_batches():
    mu = np.linspace(-8, 8, 600).reshape(2, 300)
    actual = _population_probability(mu, np.zeros_like(mu))
    np.testing.assert_allclose(actual, expit(mu).ravel(), atol=1e-9)


def test_splitting_rows_does_not_change_estimands():
    split = fitted()
    merged = fitted()
    merged.features["group_index"] = np.array([0, 1])
    merged.features["n_total"] = np.array([10, 90])
    pd.testing.assert_frame_equal(
        group_binomial_estimands(split), group_binomial_estimands(merged)
    )


def test_scalar_totals_and_zero_trial_groups():
    analysis = fitted()
    analysis.features["n_total"] = 10
    np.testing.assert_allclose(
        group_binomial_estimands(analysis).evaluated_trial_mean_probability, 1.9 / 3
    )
    analysis.features["n_total"] = np.array([10, 0, 0])
    np.testing.assert_allclose(
        group_binomial_estimands(analysis).evaluated_group_mean_probability, 0.1
    )


def test_table_reports_probabilities_and_requested_quantiles_for_all_models():
    state = AnalysisState(data=pd.DataFrame())
    for builder in (simplified_group_binomial_exponential, two_level_group_binomial):
        analysis = fitted(builder)
        # Deliberately nonlinear link: probabilities must not be transformed again.
        analysis.model_config = ModelConfig(link_function=lambda x: x + 100)
        analysis.inference_data.posterior["overall_mean"].values[:] = np.linspace(
            -5, 0, 20
        ).reshape(2, 10)
        state.add_model(analysis)
    config = CommunicateConfig.from_dict(
        {
            "communicators": [
                {
                    "binomial_estimands_table": {"credible_interval": 0.8},
                }
            ]
        }
    )
    state, verdict = config.enabled_communicators[0](state)
    assert verdict == "pass"
    assert len(state.communicate) == 2
    for analysis in state.models:
        table = state.communicate[f"model_{analysis.model_name}_binomial_estimands"]
        expected = expit(np.linspace(-5, 0, 20))
        assert table.loc["median_group_probability", "mean"] == pytest.approx(
            expected.mean()
        )
        assert table.loc["median_group_probability", "ci_lower"] == pytest.approx(
            np.quantile(expected, 0.1)
        )
        assert table.loc["median_group_probability", "ci_upper"] == pytest.approx(
            np.quantile(expected, 0.9)
        )


def test_saved_fit_can_be_summarized_without_refitting(tmp_path):
    analysis = fitted()
    analysis.save(tmp_path / "fit")
    loaded = ModelAnalysisState.load(tmp_path / "fit")
    pd.testing.assert_frame_equal(
        group_binomial_estimands(loaded), group_binomial_estimands(analysis)
    )


def test_mixed_workflow_skips_unsupported_and_unfitted_models():
    from unittest.mock import patch
    from hibayes.model.models import linear_group_binomial

    state = AnalysisState(data=pd.DataFrame())
    unsupported = fitted()
    unsupported._model = linear_group_binomial(main_effects=["group"])
    state.add_model(unsupported)
    unfitted = fitted(tag="unfitted")
    unfitted.is_fitted = False
    state.add_model(unfitted)
    with patch("hibayes.communicate.tables.binomial.logger.warning") as warning:
        _, verdict = binomial_estimands_table()(state)
    assert verdict == "NA"
    warning.assert_called_once()
    assert not state.communicate


def test_missing_posterior_sites_or_features_fail_clearly():
    analysis = fitted()
    del analysis.features["n_total"]
    with pytest.raises(ValueError, match="saved features"):
        group_binomial_estimands(analysis)
    analysis = fitted()
    del analysis.inference_data.posterior["sigma_group"]
    with pytest.raises(ValueError, match="Posterior must contain"):
        group_binomial_estimands(analysis)


@pytest.mark.parametrize(
    "key,value",
    [
        ("n_total", [0, 0, 0]),
        ("n_total", [10, -1, 3]),
        ("n_total", [10, np.nan, 3]),
        ("n_total", [10, 3]),
        ("group_index", [0, 1, 3]),
        ("group_index", [-1, 1, 1]),
        ("group_index", [0.1, 1.0, 1.0]),
    ],
)
def test_invalid_group_weights_fail_clearly(key, value):
    analysis = fitted()
    analysis.features[key] = np.array(value)
    with pytest.raises(ValueError):
        group_binomial_estimands(analysis)


@pytest.mark.parametrize("interval", [0, 1, -1, np.nan])
def test_invalid_credible_intervals(interval):
    with pytest.raises(ValueError, match="credible_interval"):
        binomial_estimands_table(credible_interval=interval)
