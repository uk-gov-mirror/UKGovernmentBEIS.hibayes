"""Integrate boundary likelihoods using priors read from the real models.

These regressions distinguish more trials within a group from more groups.
They do not require agreement with a pooled-binomial reference interval.
"""

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from numpyro.handlers import seed, trace
from scipy.special import expit

from hibayes.model.models import simplified_group_binomial_exponential


def integrate_zero_data(groups, trials, rate=1.0):
    sites = trace(
        seed(simplified_group_binomial_exponential(prior_sigma_group_rate=rate), 0)
    ).get_trace(
        {
            "obs": jnp.zeros(groups),
            "num_group": groups,
            "group_index": jnp.arange(groups),
            "n_total": jnp.full(groups, trials),
        }
    )
    prior_mu, prior_sigma = sites["overall_mean"]["fn"], sites["sigma_group"]["fn"]
    mu = np.linspace(-9, 8, 501)
    prior = np.exp(-0.5 * ((mu - float(prior_mu.loc)) / float(prior_mu.scale)) ** 2)
    x, w = leggauss(100)
    sigma = (x + 1) * 10 / float(prior_sigma.rate)
    sw = w * 10 * np.exp(-sigma * float(prior_sigma.rate))
    x, w = hermgauss(160)
    z, zw = x * np.sqrt(2), w / np.sqrt(np.pi)
    location_weight = np.zeros(len(mu))
    observed_numerator = np.zeros(len(mu))
    population_numerator = np.zeros(len(mu))
    for start in range(0, len(mu), 20):
        logits = (
            mu[start : start + 20, None, None] + sigma[None, :, None] * z[None, None, :]
        )
        p = expit(logits)
        likelihood = np.exp(-trials * np.logaddexp(0, logits))
        group_likelihood = likelihood @ zw
        marginal_likelihood = group_likelihood**groups
        location_weight[start : start + 20] = marginal_likelihood @ sw
        observed_numerator[start : start + 20] = (
            (likelihood * p) @ zw * group_likelihood ** (groups - 1)
        ) @ sw
        population_numerator[start : start + 20] = (marginal_likelihood * (p @ zw)) @ sw
    normalization = prior @ location_weight
    return {
        "median_group": float((prior * location_weight) @ expit(mu) / normalization),
        "evaluated_group": float(prior @ observed_numerator / normalization),
        "population": float(prior @ population_numerator / normalization),
    }


def test_more_trials_identify_observed_rate_but_not_population_location():
    small = integrate_zero_data(1, 20)
    large = integrate_zero_data(1, 1000)
    assert large["evaluated_group"] < small["evaluated_group"] / 20
    assert large["median_group"] > 0.3
    assert large["population"] > 0.3


def test_more_independent_zero_groups_inform_population_and_prior_sensitivity():
    one = integrate_zero_data(1, 100)
    four = integrate_zero_data(4, 25)
    twenty = integrate_zero_data(20, 5)
    tighter = integrate_zero_data(4, 25, rate=3.0)
    assert one["population"] > four["population"] > twenty["population"]
    assert tighter["population"] < four["population"]
    assert 0.18 < four["median_group"] < 0.22
    assert four["evaluated_group"] < 0.03
