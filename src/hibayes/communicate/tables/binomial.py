"""Probability-scale estimands for the two grouped binomial models."""

import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from scipy.special import expit, ndtr

from ...analysis_state import ModelAnalysisState
from ...utils import init_logger
from .._communicate import communicate
from ..utils import resolve_models_to_run
from .tables import _df_to_rich_table

logger = init_logger()
_SUPPORTED_MODELS = {
    "simplified_group_binomial_exponential",
    "two_level_group_binomial",
}


def _population_probability(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Integrate over a new N(0, 1) group effect for each posterior draw.

    For sigma >= 1, integrate Phi((mu + L) / sigma) over standard logistic
    noise L instead. This equivalent representation stays smooth at large
    sigma, where normal quadrature sees a near step function. Batching bounds
    memory; omitted tail mass beyond +/-40 is less than 9e-18.
    """
    result = np.empty(mu.size)
    mu, sigma = mu.ravel(), sigma.ravel()
    for start in range(0, mu.size, 256):
        m, s = mu[start : start + 256], sigma[start : start + 256]
        integral, _, info = quad_vec(
            lambda z: np.where(
                s < 1,
                expit(m + s * z) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
                ndtr((m + z) / np.maximum(s, 1)) * expit(z) * expit(-z),
            ),
            -40,
            40,
            epsabs=1e-9,
            epsrel=1e-9,
            norm="max",
            full_output=True,
        )
        if not info.success:
            raise ValueError("Population probability integration did not converge")
        result[start : start + 256] = integral
    return result


def group_binomial_estimands(model_analysis: ModelAnalysisState) -> pd.DataFrame:
    """Return one row per posterior draw, with explicitly named probabilities.

    Supports saved fits from simplified_group_binomial_exponential and
    two_level_group_binomial. Evaluated groups have positive total trials;
    repeated rows for a group are combined before computing weights. The
    population mean assumes exchangeable groups drawn from the fitted normal
    distribution of group log-odds. It is not the rate of one particular new
    group and does not include that group's predictive variation.
    """
    if model_analysis.model.__name__ not in _SUPPORTED_MODELS:
        raise ValueError(
            "Estimands require one of the two supported grouped binomial models"
        )
    posterior = model_analysis.inference_data.posterior
    required = {"overall_mean", "sigma_group", "group_effects"}
    if not required.issubset(posterior.data_vars):
        raise ValueError(f"Posterior must contain {sorted(required)}")
    mu = np.asarray(posterior.overall_mean.transpose("chain", "draw"), dtype=float)
    sigma = np.asarray(posterior.sigma_group.transpose("chain", "draw"), dtype=float)
    effects = np.asarray(
        posterior.group_effects.transpose("chain", "draw", ...), dtype=float
    )
    if effects.ndim != 3:
        raise ValueError("group_effects must have exactly one group dimension")
    if not all(np.isfinite(x).all() for x in (mu, sigma, effects)) or (sigma < 0).any():
        raise ValueError("Posterior draws must be finite and sigma_group nonnegative")

    features = model_analysis.features
    if not {"group_index", "n_total"}.issubset(features):
        raise ValueError(
            "Estimands require group_index and n_total in the saved features"
        )
    index = np.asarray(features["group_index"])
    totals = np.asarray(features["n_total"], dtype=float)
    if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
        raise ValueError("group_index must be a one-dimensional integer array")
    if totals.ndim == 0:
        totals = np.broadcast_to(totals, index.shape)
    if (
        totals.shape != index.shape
        or not np.isfinite(totals).all()
        or (totals < 0).any()
    ):
        raise ValueError(
            "n_total must contain finite nonnegative counts aligned with group_index"
        )
    if (index < 0).any() or (index >= effects.shape[-1]).any():
        raise ValueError("group_index contains an index outside group_effects")
    weights = np.bincount(index, weights=totals, minlength=effects.shape[-1])
    evaluated = weights > 0
    if not evaluated.any():
        raise ValueError("Estimands require at least one positive trial count")
    probabilities = expit(effects)
    return pd.DataFrame(
        {
            "median_group_probability": expit(mu).ravel(),
            "evaluated_group_mean_probability": probabilities[..., evaluated]
            .mean(-1)
            .ravel(),
            "evaluated_trial_mean_probability": (
                probabilities @ (weights / weights.sum())
            ).ravel(),
            "population_mean_probability": _population_probability(mu, sigma),
        }
    )


@communicate
def binomial_estimands_table(
    *, best_model: bool = False, credible_interval: float = 0.95
):
    """Summarize grouped binomial probabilities without applying a link again.

    Produces means, SDs and equal-tailed credible intervals for all fitted
    supported models by default, allowing comparison of pooling assumptions.
    Unsupported models in a mixed-model workflow are skipped with a warning.
    """
    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1")

    def communicate(state, display=None):
        produced_table = False
        for analysis in resolve_models_to_run(state, best_model, display):
            if not analysis.is_fitted:
                continue
            if analysis.model.__name__ not in _SUPPORTED_MODELS:
                logger.warning(
                    f"binomial_estimands_table: skipping unsupported model {analysis.model_name}"
                )
                continue
            draws = group_binomial_estimands(analysis)
            tail = (1 - credible_interval) / 2
            table = pd.DataFrame(
                {
                    "mean": draws.mean(),
                    "sd": draws.std(),
                    "ci_lower": draws.quantile(tail),
                    "ci_upper": draws.quantile(1 - tail),
                }
            )
            table.index.name = "estimand"
            state.add_table(
                table=table,
                table_name=f"model_{analysis.model_name}_binomial_estimands",
            )
            produced_table = True
            if display is not None and getattr(display, "is_live", False):
                display.update_body_content(_df_to_rich_table(table, 4))
        return state, "pass" if produced_table else "NA"

    return communicate
