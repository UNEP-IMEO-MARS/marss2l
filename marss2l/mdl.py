"""Minimum detection limit (MDL) fitting utilities.

This module fits a power-law distribution multiplied by a
probability-of-detection (PoD) curve to **wind-normalized** flux rates
``q = flux / wind_speed``.  Observed (detected) wind-normalized rates are
expected to follow a power-law above a minimum value, modulated by the
instrument/algorithm detection probability which increases with signal
strength.  The PoD can be selected by name using ``PoDType.LOGNORMAL_CDF``
or ``PoDType.SIGMOID``, and the same model is supported for both
maximum-likelihood estimation (MLE) and Bayesian inference via MCMC.

Fitted parameters
-----------------
By default the model fits **three** parameters (``x_min`` is fixed):

- ``alpha``: power-law exponent controlling the tail decay of the detected
  wind-normalized flux distribution.
- ``mu``: PoD location parameter in log space — roughly the log of the
  wind-normalized flux where detection probability begins to rise.
- ``sigma``: PoD scale parameter in log space — controls how steeply
  detection probability increases.

The minimum wind-normalized flux ``x_min`` is fixed to
``X_MIN_DEFAULT = 1.0`` (kg/h per m/s) by default.  It can optionally be
fitted by setting ``fit_x_min=True`` in ``fit_mle`` / ``fit_mcmc``.

Model
-----
The fitting variable is ``q_i = flux_i / wind_speed_i``.

The combined density of detected wind-normalized flux rates is:

$$
f(q) = f_{\\text{PL}}(q) \\cdot \\text{PoD}(q),
$$

$$
f_{\\text{PL}}(q) = (\\alpha - 1) \\, x_{\\min}^{\\alpha-1} \\, q^{-\\alpha},
$$

with PoD defined as either:

- **Log-normal CDF**: $\\text{PoD}(q) = \\Phi((\\log q - \\mu) / \\sigma)$
- **Sigmoid**: $\\text{PoD}(q) = \\text{logistic}((\\log q - \\mu) / \\sigma)$

where $\\Phi$ is the standard normal CDF.

To display the PoD as a function of **flux** (instead of wind-normalized
flux), the module marginalizes over a reference wind-speed distribution:

$$
\\text{PoD}_{\\text{flux}}(F) = \\frac{1}{N} \\sum_{j=1}^{N} \\text{PoD}(F / w_j),
$$

where $\\{w_j\\}$ are wind-speed samples from the full dataset (~90 k rows).

Quick start
-----------
Expected outputs:

- ``fit_mle(...)`` returns an ``MLEFit`` with fitted parameters and optimizer
  diagnostics.  Access ``mle.params`` for ``[alpha, x_min, mu, sigma]``.
- ``fit_mcmc(...)`` returns an ``MCMCFit`` with posterior samples and
  acceptance statistics.  Access ``mcmc.samples`` with shape
  ``(n_samples, 4)``.
- ``posterior_pod_flux_ci(...)`` returns a ``CurveSummary`` with ``.median``,
  ``.lower``, ``.upper`` arrays — the PoD vs flux curve with CI bands.

```python
import matplotlib.pyplot as plt
import numpy as np
from marss2l.mdl import PoDType, fit_mle, fit_mcmc, posterior_pod_flux_ci

# Wind-normalized flux rates (the quantity that follows the power-law)
q = df_data["ch4_fluxrate"].to_numpy() / df_data["wind_speed"].to_numpy()

# Reference wind speeds for marginalization (full dataset, ~90k rows)
wind_speeds = dataframe_data_traintest["wind_speed"].to_numpy()

mle = fit_mle(q, pod=PoDType.LOGNORMAL_CDF)
mcmc = fit_mcmc(q, pod=PoDType.LOGNORMAL_CDF, nwalkers=120, nsteps=30000)

# PoD curve vs flux with 90% CI, marginalized over wind speeds
flux_grid = np.logspace(np.log10(100), np.log10(50000), 200)
pod_ci = posterior_pod_flux_ci(flux_grid, mcmc.samples, wind_speeds,
                               pod=PoDType.LOGNORMAL_CDF)

plt.plot(flux_grid, pod_ci.median, label="PoD median")
plt.fill_between(flux_grid, pod_ci.lower, pod_ci.upper, alpha=0.2,
                 label="90% CI")
plt.xscale("log")
plt.xlabel("Flowrate (Kg/h)")
plt.ylabel("PoD")
plt.legend()
plt.show()
```
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Mapping, Optional, Sequence, Tuple

import emcee
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
import matplotlib.pyplot as plt

ArrayLike = NDArray[np.float_]
PODFunction = Callable[[ArrayLike, float, float], ArrayLike]

# Default minimum wind-normalized flux q_min (kg/h per m/s).
X_MIN_DEFAULT: float = 1.0
# Default maximum wind-normalized flux q_max (kg/h per m/s).
X_MAX_DEFAULT: float = 80000.0

# Parameter names when x_min/x_max are fixed (default) vs fitted.
PARAM_NAMES_FIXED: list[str] = ["alpha", "mu", "sigma"]
PARAM_NAMES_FIT_XMIN: list[str] = ["alpha", "x_min", "mu", "sigma"]
PARAM_NAMES_FIT_BOTH: list[str] = ["alpha", "x_min", "x_max", "mu", "sigma"]


class PoDType(StrEnum):
    """Probability-of-detection functional form.

    Use a named option to select the PoD curve used by the likelihood.
    """

    LOGNORMAL_CDF = "lognormal_cdf"
    SIGMOID = "sigmoid"


@dataclass(frozen=True)
class ParameterBounds:
    """Uniform prior bounds for the model parameters.

    Holds (min, max) bounds used by the MCMC prior and MLE optimizer.

    Attributes
    ----------
    alpha : tuple[float, float]
        Bounds for the power-law exponent.
    mu : tuple[float, float]
        Bounds for the PoD location parameter in log space.
    sigma : tuple[float, float]
        Bounds for the PoD scale parameter in log space.
    x_min : tuple[float, float] | None
        Bounds for x_min. Only used when ``fit_x_min=True``.
    x_max : tuple[float, float] | None
        Bounds for x_max. Only used when ``fit_x_max=True``.
    """

    alpha: Tuple[float, float]
    mu: Tuple[float, float]
    sigma: Tuple[float, float]
    x_min: Optional[Tuple[float, float]] = None
    x_max: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class MLEFit:
    """MLE result for the power-law + PoD model.

    Contains optimizer metadata and the fitted parameters.

    Attributes
    ----------
    params : NDArray[np.float_]
        Fitted full parameters in the order [alpha, x_min, x_max, mu, sigma].
        When ``fit_x_min=False`` or ``fit_x_max=False``, the corresponding
        fixed values are used.
    success : bool
        Whether the optimizer reported success.
    fun : float
        Final negative log-likelihood value.
    message : str
        Optimizer termination message.
    nfev : int
        Number of function evaluations.
    nit : int | None
        Number of optimizer iterations, if provided by the solver.
    pod : PoDType
        PoD function used for the fit.
    fit_x_min : bool
        Whether x_min was fitted or fixed.
    fit_x_max : bool
        Whether x_max was fitted or fixed.
    """

    params: ArrayLike
    success: bool
    fun: float
    message: str
    nfev: int
    nit: Optional[int]
    pod: PoDType
    fit_x_min: bool
    fit_x_max: bool


@dataclass(frozen=True)
class MCMCFit:
    """MCMC result for the power-law + PoD model.

    Stores posterior samples and diagnostics for the model.

    Attributes
    ----------
    samples : NDArray[np.float_]
        Posterior samples with shape (n_samples, 5).  Each row is
        [alpha, x_min, x_max, mu, sigma].  When ``fit_x_min=False`` or
        ``fit_x_max=False``, the corresponding columns are constant
        across all samples.
    acceptance_fraction : NDArray[np.float_]
        Per-walker acceptance fractions from emcee.
    log_prob : NDArray[np.float_]
        Posterior log-probability for each sample.
    pod : PoDType
        PoD function used for the fit.
    fit_x_min : bool
        Whether x_min was fitted or fixed.
    fit_x_max : bool
        Whether x_max was fitted or fixed.
    """

    samples: ArrayLike
    acceptance_fraction: ArrayLike
    log_prob: ArrayLike
    pod: PoDType
    fit_x_min: bool
    fit_x_max: bool


@dataclass(frozen=True)
class CurveSummary:
    """Summary statistics for posterior curve bands.

    Provides median and confidence interval bands for a curve evaluated
    on a grid (e.g. PDF vs q, or PoD vs flux).

    Attributes
    ----------
    median : NDArray[np.float_]
        Median curve values evaluated on a grid.
    lower : NDArray[np.float_]
        Lower confidence band values.
    upper : NDArray[np.float_]
        Upper confidence band values.
    ci : tuple[float, float]
        Quantile levels used for the confidence interval.
    """

    median: ArrayLike
    lower: ArrayLike
    upper: ArrayLike
    ci: Tuple[float, float]


# ---------------------------------------------------------------------------
# PoD functions
# ---------------------------------------------------------------------------


def lognormal_cdf_pod(x: ArrayLike, mu: float, sigma: float) -> ArrayLike:
    """Log-normal CDF PoD for detection probability.

    Computes $\\Phi((\\log x - \\mu)/\\sigma)$ for each element of $x$.

    Args:
        x (NDArray[np.float_]): Wind-normalized flux rates.
        mu (float): Location parameter of the PoD curve in log space.
        sigma (float): Scale parameter of the PoD curve in log space.

    Returns:
        NDArray[np.float_]: PoD values in $[0, 1]$.
    """
    # Transform to log space.
    log_x = np.log(x)
    # Evaluate standard normal CDF at (log_x - mu) / sigma.
    #  expit(z)
    return norm.cdf(log_x, loc=mu, scale=sigma)


def sigmoid_pod(x: ArrayLike, mu: float, sigma: float) -> ArrayLike:
    """Sigmoid (logistic) PoD for detection probability.

    Computes $\\text{logistic}((\\log x - \\mu)/\\sigma)$ using the logistic function.

    Args:
        x (NDArray[np.float_]): Wind-normalized flux rates.
        mu (float): Location parameter of the PoD curve in log space.
        sigma (float): Scale parameter of the PoD curve in log space.

    Returns:
        NDArray[np.float_]: PoD values in $[0, 1]$.
    """
    # Transform to log space.
    log_x = np.log(x)
    # Evaluate logistic sigmoid at (log_x - mu) / sigma.
    return expit((log_x - mu) / sigma)


POD_REGISTRY: Mapping[PoDType, PODFunction] = {
    PoDType.LOGNORMAL_CDF: lognormal_cdf_pod,
    PoDType.SIGMOID: sigmoid_pod,
}


def resolve_pod_fn(pod: PoDType, pod_fn: Optional[PODFunction] = None) -> PODFunction:
    """Resolve the PoD function from a name or explicit callable.

    If a callable is provided, it takes precedence over the enum name.

    Args:
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.

    Returns:
        Callable[[NDArray, float, float], NDArray]: PoD function.
    """
    if pod_fn is not None:
        return pod_fn
    return POD_REGISTRY[pod]


# ---------------------------------------------------------------------------
# Density functions
# ---------------------------------------------------------------------------


def power_law_pdf(x: ArrayLike, alpha: float, x_min: float, x_max: float = np.inf) -> ArrayLike:
    r"""Power-law PDF for $x \in [x_{\min}, x_{\max}]$.

    For unbounded case ($x_{\max} = \infty$):

    .. math::

        f_{PL}(x) = (\alpha - 1) x_{\min}^{\alpha-1} x^{-\alpha}

    For truncated case ($x_{\max} < \infty$):

    .. math::

        f_{PL}(x) = \frac{(\alpha - 1) x^{-\alpha}}{x_{\min}^{1-\alpha} - x_{\max}^{1-\alpha}}

    Args:
        x (NDArray[np.float_]): Wind-normalized flux rates.
        alpha (float): Power-law exponent.
        x_min (float): Minimum wind-normalized flux (support lower bound).
        x_max (float): Maximum wind-normalized flux (support upper bound).
            Defaults to infinity (unbounded).

    Returns:
        NDArray[np.float_]: Power-law density values (0 outside [x_min, x_max]).
    """
    x = np.asarray(x, dtype=float)
    
    if np.isinf(x_max):
        # Unbounded case: standard Pareto
        norm_const = (alpha - 1.0) * (x_min ** (alpha - 1.0))
    else:
        # Truncated case: renormalize over finite interval
        norm_const = (alpha - 1.0) / (x_min ** (1.0 - alpha) - x_max ** (1.0 - alpha))
    
    # Power-law decay: x^(-alpha).
    pdf = norm_const * (x ** (-alpha))
    # Zero out values outside the support [x_min, x_max].
    in_support = (x >= x_min) & (x <= x_max)
    return np.where(in_support, pdf, 0.0)


def combined_pdf(
    x: ArrayLike,
    alpha: float,
    x_min: float,
    x_max: float,
    mu: float,
    sigma: float,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
) -> ArrayLike:
    r"""Combined PDF: power-law times PoD.

    .. math::

        f(x) = f_{PL}(x) \cdot PoD(x)

    Args:
        x (NDArray[np.float_]): Wind-normalized flux rates.
        alpha (float): Power-law exponent.
        x_min (float): Minimum wind-normalized flux.
        x_max (float): Maximum wind-normalized flux.
        mu (float): PoD location parameter in log space.
        sigma (float): PoD scale parameter in log space.
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.

    Returns:
        NDArray[np.float_]: Combined density values.
    """
    pod_fn = resolve_pod_fn(pod, pod_fn)
    # Compute power-law density term.
    pl = power_law_pdf(x, alpha, x_min, x_max)
    # Compute the PoD term (bounded in [0, 1]).
    pod_vals = pod_fn(np.asarray(x, dtype=float), mu, sigma)
    # Product gives the observed density of detected wind-normalized fluxes.
    return pl * pod_vals


# ---------------------------------------------------------------------------
# Likelihood helpers (always work with full 4-param vector internally)
# ---------------------------------------------------------------------------


def _full_params(
    params: Sequence[float],
    fit_x_min: bool,
    fit_x_max: bool,
    x_min_fixed: float,
    x_max_fixed: float,
) -> Tuple[float, float, float, float, float]:
    """Unpack params into (alpha, x_min, x_max, mu, sigma).

    Parameter vector options:
    - ``fit_x_min=False, fit_x_max=False``: [alpha, mu, sigma] (3 params)
    - ``fit_x_min=True, fit_x_max=False``: [alpha, x_min, mu, sigma] (4 params)
    - ``fit_x_min=False, fit_x_max=True``: [alpha, x_max, mu, sigma] (4 params)
    - ``fit_x_min=True, fit_x_max=True``: [alpha, x_min, x_max, mu, sigma] (5 params)

    Args:
        params (Sequence[float]): Parameter vector (length 3, 4, or 5).
        fit_x_min (bool): Whether x_min is part of the parameter vector.
        fit_x_max (bool): Whether x_max is part of the parameter vector.
        x_min_fixed (float): Fixed x_min value used when ``fit_x_min=False``.
        x_max_fixed (float): Fixed x_max value used when ``fit_x_max=False``.

    Returns:
        tuple[float, float, float, float, float]: (alpha, x_min, x_max, mu, sigma).
    """
    if fit_x_min and fit_x_max:
        alpha, x_min, x_max, mu, sigma = params
    elif fit_x_min:
        alpha, x_min, mu, sigma = params
        x_max = x_max_fixed
    elif fit_x_max:
        alpha, x_max, mu, sigma = params
        x_min = x_min_fixed
    else:
        alpha, mu, sigma = params
        x_min = x_min_fixed
        x_max = x_max_fixed
    return alpha, x_min, x_max, mu, sigma


def log_likelihood(
    params: Sequence[float],
    x: ArrayLike,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
    fit_x_min: bool = False,
    fit_x_max: bool = False,
    x_min_fixed: float = X_MIN_DEFAULT,
    x_max_fixed: float = X_MAX_DEFAULT,
    eps: float = 1e-300,
) -> float:
    r"""Log-likelihood for the combined PDF.

    .. math::

        \log L(\theta) = \sum_{i:\,x_{min} \le q_i \le x_{max}} \log f(q_i;\,\theta)

    Args:
        params (Sequence[float]): Parameters — see _full_params for length.
        x (NDArray[np.float_]): Wind-normalized flux rates q.
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.
        fit_x_min (bool): Whether x_min is being fitted.
        fit_x_max (bool): Whether x_max is being fitted.
        x_min_fixed (float): Fixed x_min value.
        x_max_fixed (float): Fixed x_max value.
        eps (float): Numerical floor for PoD values before taking log.

    Returns:
        float: Log-likelihood value.
    """
    alpha, x_min, x_max, mu, sigma = _full_params(
        params, fit_x_min, fit_x_max, x_min_fixed, x_max_fixed
    )

    # Guard against invalid parameter regions.
    if alpha <= 1.0 or x_min <= 0.0 or x_max <= x_min or sigma <= 0.0:
        return float("-inf")

    x = np.asarray(x, dtype=float)
    # Only use data within the power-law support [x_min, x_max].
    x_in_support = x[(x >= x_min) & (x <= x_max)]
    if x_in_support.size == 0:
        return float("-inf")

    n = x_in_support.size
    pod_fn = resolve_pod_fn(pod, pod_fn)

    # Log normalization constant depends on whether x_max is finite.
    if np.isinf(x_max):
        # Unbounded: log[(alpha - 1) * x_min^(alpha-1)]
        log_norm = np.log(alpha - 1.0) + (alpha - 1.0) * np.log(x_min)
    else:
        # Truncated: log[(alpha - 1) / (x_min^(1-alpha) - x_max^(1-alpha))]
        denom = x_min ** (1.0 - alpha) - x_max ** (1.0 - alpha)
        log_norm = np.log(alpha - 1.0) - np.log(denom)
    
    term1 = n * log_norm
    # Term 2: power-law decay: -alpha * sum(log(q_i)).
    term2 = -alpha * np.sum(np.log(x_in_support))
    # Term 3: PoD contribution: sum(log(PoD(q_i))).
    pod_vals = pod_fn(x_in_support, mu, sigma)
    term3 = np.sum(np.log(np.clip(pod_vals, eps, 1.0)))

    return float(term1 + term2 + term3)


def negative_log_likelihood(
    params: Sequence[float],
    x: ArrayLike,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
    fit_x_min: bool = False,
    fit_x_max: bool = False,
    x_min_fixed: float = X_MIN_DEFAULT,
    x_max_fixed: float = X_MAX_DEFAULT,
) -> float:
    """Negative log-likelihood for optimization.

    This is the objective minimized by the MLE optimizer.

    Args:
        params (Sequence[float]): Parameters (3, 4, or 5 elements).
        x (NDArray[np.float_]): Wind-normalized flux rates q.
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.
        fit_x_min (bool): Whether x_min is being fitted.
        fit_x_max (bool): Whether x_max is being fitted.
        x_min_fixed (float): Fixed x_min value.
        x_max_fixed (float): Fixed x_max value.

    Returns:
        float: Negative log-likelihood value.
    """
    return -log_likelihood(
        params, x, pod=pod, pod_fn=pod_fn,
        fit_x_min=fit_x_min, fit_x_max=fit_x_max,
        x_min_fixed=x_min_fixed, x_max_fixed=x_max_fixed,
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def default_initial_guess(
    x: ArrayLike,
) -> ArrayLike:
    """Heuristic initial guess based on data quantiles.

    Places the PoD midpoint (mu) at the 10th percentile of log(q), so that
    most observed data already have high detection probability.  Uses a
    moderate sigma (steep PoD transition) as starting point.

    Args:
        x (NDArray[np.float_]): Wind-normalized flux rates q.
        fit_x_min (bool): Ignored, kept for API compatibility.
        fit_x_max (bool): Ignored, kept for API compatibility.

    Returns:
        NDArray[np.float_]: Initial guess with 5 elements:
            [alpha, x_min, x_max, mu, sigma].
    """
    x = np.asarray(x, dtype=float)
    log_x = np.log(x)
    # Place PoD midpoint below most data — the 10th percentile.
    mu_guess = float(np.quantile(log_x, 0.10))
    # Start with a fairly steep PoD (small sigma).
    sigma_guess = float(np.std(log_x) * 0.3)
    sigma_guess = max(sigma_guess, 0.1)

    alpha_guess = 2.0
    x_min_guess = 1.0
    x_max_guess = float(np.quantile(x, 0.99))

    return np.array(
        [alpha_guess, x_min_guess, x_max_guess, mu_guess, sigma_guess], dtype=float
    )


def default_bounds(
    x: ArrayLike,
    fit_x_min: bool = False,
    fit_x_max: bool = False,
) -> ParameterBounds:
    """Heuristic bounds informed by the data distribution.

    Uses wider mu range extending well below the data minimum to allow
    the PoD midpoint to sit below all observations.

    Args:
        x (NDArray[np.float_]): Wind-normalized flux rates q.
        fit_x_min (bool): Whether x_min is being fitted.
        fit_x_max (bool): Whether x_max is being fitted.

    Returns:
        ParameterBounds: Bounds for the fitted parameters.
    """
    x = np.asarray(x, dtype=float)
    log_min = float(np.log(np.min(x)))
    log_max = float(np.log(np.max(x)))
    log_mean = float(np.mean(np.log(x)))
    log_range = log_max - log_min

    x_min_bounds = None
    x_max_bounds = None
    if fit_x_min:
        x_min_bounds = (0.1, float(np.quantile(x, 0.5)))
    if fit_x_max:
        x_max_bounds = (float(np.quantile(x, 0.5)), float(np.max(x) * 2.0))

    return ParameterBounds(
        alpha=(1.01, 10.0),
        # Allow mu to go well below the data (PoD midpoint can be far left).
        mu=(log_mean - 0.5, log_mean + log_range),
        # Allow tight (steep) PoD curves with small sigma.
        sigma=(0.01, 5.0),
        x_min=x_min_bounds,
        x_max=x_max_bounds,
    )


# ---------------------------------------------------------------------------
# MLE
# ---------------------------------------------------------------------------


def fit_mle(
    x: ArrayLike,
    initial_guess: Optional[Sequence[float]] = None,
    bounds: Optional[ParameterBounds] = None,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
    fit_x_min: bool = False,
    fit_x_max: bool = False,
    x_min_fixed: float = X_MIN_DEFAULT,
    x_max_fixed: float = X_MAX_DEFAULT,
    method: str = "L-BFGS-B",
) -> MLEFit:
    """Fit parameters by maximum likelihood.

    Optimizes the negative log-likelihood of the power-law x PoD model
    using box constraints.

    Args:
        x (NDArray[np.float_]): Wind-normalized flux rates q.
        initial_guess (Sequence[float] | None): Initial parameters.
        bounds (ParameterBounds | None): Parameter bounds.
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.
        fit_x_min (bool): If True, fit x_min as a free parameter.
            Defaults to False (x_min is fixed at ``x_min_fixed``).
        fit_x_max (bool): If True, fit x_max as a free parameter.
            Defaults to False (x_max is fixed at ``x_max_fixed``).
        x_min_fixed (float): Fixed x_min value, only used when
            ``fit_x_min=False``.
        x_max_fixed (float): Fixed x_max value, only used when
            ``fit_x_max=False``.
        method (str): SciPy optimizer method.

    Returns:
        MLEFit: Fitted parameters and optimizer diagnostics.
    """
    x = np.asarray(x, dtype=float)
    if initial_guess is None:
        initial_guess_full = default_initial_guess(x)
        # Extract only the parameters being fitted.
        if fit_x_min and fit_x_max:
            initial_guess = initial_guess_full  # [alpha, x_min, x_max, mu, sigma]
        elif fit_x_min:
            initial_guess = initial_guess_full[[0, 1, 3, 4]]  # [alpha, x_min, mu, sigma]
        elif fit_x_max:
            initial_guess = initial_guess_full[[0, 2, 3, 4]]  # [alpha, x_max, mu, sigma]
        else:
            initial_guess = initial_guess_full[[0, 3, 4]]  # [alpha, mu, sigma]
    initial_guess = np.asarray(initial_guess, dtype=float)
    if bounds is None:
        bounds = default_bounds(x, fit_x_min=fit_x_min, fit_x_max=fit_x_max)

    # Build scipy bounds list matching the parameter vector order.
    if fit_x_min and fit_x_max:
        scipy_bounds = [bounds.alpha, bounds.x_min, bounds.x_max, bounds.mu, bounds.sigma]
    elif fit_x_min:
        scipy_bounds = [bounds.alpha, bounds.x_min, bounds.mu, bounds.sigma]
    elif fit_x_max:
        scipy_bounds = [bounds.alpha, bounds.x_max, bounds.mu, bounds.sigma]
    else:
        scipy_bounds = [bounds.alpha, bounds.mu, bounds.sigma]

    result = minimize(
        negative_log_likelihood,
        x0=initial_guess,
        args=(x, pod, pod_fn, fit_x_min, fit_x_max, x_min_fixed, x_max_fixed),
        method=method,
        bounds=scipy_bounds,
    )

    # Reconstruct full 5-param vector for consistent output.
    alpha, x_min, x_max, mu, sigma = _full_params(
        result.x, fit_x_min, fit_x_max, x_min_fixed, x_max_fixed
    )
    full_params = np.array([alpha, x_min, x_max, mu, sigma], dtype=float)

    return MLEFit(
        params=full_params,
        success=bool(result.success),
        fun=float(result.fun),
        message=str(result.message),
        nfev=int(result.nfev),
        nit=int(result.nit) if hasattr(result, "nit") and result.nit is not None else None,
        pod=pod,
        fit_x_min=fit_x_min,
        fit_x_max=fit_x_max,
    )


# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------


def log_prior(
    params: Sequence[float],
    bounds: ParameterBounds,
    fit_x_min: bool = False,
) -> float:
    """Uniform log-prior within bounds.

    Returns 0 inside bounds and $-\\infty$ outside.

    Args:
        params (Sequence[float]): Parameters (3 or 4 elements).
        bounds (ParameterBounds): Parameter bounds.
        fit_x_min (bool): Whether x_min is being fitted.

    Returns:
        float: Log-prior value.
    """
    if fit_x_min:
        alpha, x_min, mu, sigma = params
        if bounds.x_min is not None and not (bounds.x_min[0] <= x_min <= bounds.x_min[1]):
            return float("-inf")
    else:
        alpha, mu, sigma = params

    if not (bounds.alpha[0] <= alpha <= bounds.alpha[1]):
        return float("-inf")
    if not (bounds.mu[0] <= mu <= bounds.mu[1]):
        return float("-inf")
    if not (bounds.sigma[0] <= sigma <= bounds.sigma[1]):
        return float("-inf")
    return 0.0


def log_posterior(
    params: Sequence[float],
    x: ArrayLike,
    bounds: ParameterBounds,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
    fit_x_min: bool = False,
    fit_x_max: bool = False,
    x_min_fixed: float = X_MIN_DEFAULT,
    x_max_fixed: float = X_MAX_DEFAULT,
) -> float:
    """Log-posterior: log-prior + log-likelihood.

    Combines the uniform prior with the data likelihood.

    Args:
        params (Sequence[float]): Parameters (3, 4, or 5 elements).
        x (NDArray[np.float_]): Wind-normalized flux rates q.
        bounds (ParameterBounds): Parameter bounds.
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.
        fit_x_min (bool): Whether x_min is being fitted.
        fit_x_max (bool): Whether x_max is being fitted.
        x_min_fixed (float): Fixed x_min value.
        x_max_fixed (float): Fixed x_max value.

    Returns:
        float: Log-posterior value.
    """
    lp = log_prior(params, bounds, fit_x_min=fit_x_min, fit_x_max=fit_x_max)
    if not np.isfinite(lp):
        return float("-inf")
    return lp + log_likelihood(
        params, x, pod=pod, pod_fn=pod_fn,
        fit_x_min=fit_x_min, fit_x_max=fit_x_max,
        x_min_fixed=x_min_fixed, x_max_fixed=x_max_fixed,
    )


def fit_mcmc(
    x: ArrayLike,
    initial_guess: Optional[Sequence[float]] = None,
    bounds: Optional[ParameterBounds] = None,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
    fit_x_min: bool = False,
    fit_x_max: bool = False,
    x_min_fixed: float = X_MIN_DEFAULT,
    x_max_fixed: float = X_MAX_DEFAULT,
    nwalkers: int = 80,
    nsteps: int = 20000,
    burn_in: int = 5000,
    thin: int = 10,
    random_seed: Optional[int] = None,
) -> MCMCFit:
    """Fit parameters by MCMC using emcee.

    Runs an ensemble sampler and returns posterior samples and diagnostics.

    Args:
        x (NDArray[np.float_]): Wind-normalized flux rates q.
        initial_guess (Sequence[float] | None): Initial parameters.
        bounds (ParameterBounds | None): Parameter bounds.
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.
        fit_x_min (bool): If True, fit x_min as a free parameter.
        fit_x_max (bool): If True, fit x_max as a free parameter.
        x_min_fixed (float): Fixed x_min value.
        x_max_fixed (float): Fixed x_max value.
        nwalkers (int): Number of MCMC walkers.
        nsteps (int): Number of sampling steps.
        burn_in (int): Number of burn-in steps to discard.
        thin (int): Thinning factor for samples.
        random_seed (int | None): Random seed for reproducibility.

    Returns:
        MCMCFit: Posterior samples and MCMC diagnostics.
    """
    x = np.asarray(x, dtype=float)
    if initial_guess is None:
        initial_guess_full = default_initial_guess(x)
        # Extract only the parameters being fitted.
        if fit_x_min and fit_x_max:
            initial_guess = initial_guess_full  # [alpha, x_min, x_max, mu, sigma]
        elif fit_x_min:
            initial_guess = initial_guess_full[[0, 1, 3, 4]]  # [alpha, x_min, mu, sigma]
        elif fit_x_max:
            initial_guess = initial_guess_full[[0, 2, 3, 4]]  # [alpha, x_max, mu, sigma]
        else:
            initial_guess = initial_guess_full[[0, 3, 4]]  # [alpha, mu, sigma]
    initial_guess = np.asarray(initial_guess, dtype=float)
    if bounds is None:
        bounds = default_bounds(x, fit_x_min=fit_x_min, fit_x_max=fit_x_max)

    rng = np.random.default_rng(random_seed)
    ndim = len(initial_guess)
    # Scatter walkers around the initial guess.
    pos = initial_guess + 1e-4 * rng.standard_normal(size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior,
        args=(x, bounds, pod, pod_fn, fit_x_min, fit_x_max, x_min_fixed, x_max_fixed),
    )
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Discard burn-in and thin the chain.
    raw_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    log_prob = sampler.get_log_prob(discard=burn_in, thin=thin, flat=True)

    # Expand to full 5-param rows [alpha, x_min, x_max, mu, sigma] for consistency.
    n_samples = raw_samples.shape[0]
    if fit_x_min and fit_x_max:
        samples_full = np.asarray(raw_samples, dtype=float)
    elif fit_x_min:
        # Insert fixed x_max as the third column.
        samples_full = np.column_stack([
            raw_samples[:, 0],                          # alpha
            raw_samples[:, 1],                          # x_min
            np.full(n_samples, x_max_fixed),            # x_max (fixed)
            raw_samples[:, 2],                          # mu
            raw_samples[:, 3],                          # sigma
        ])
    elif fit_x_max:
        # Insert fixed x_min as the second column.
        samples_full = np.column_stack([
            raw_samples[:, 0],                          # alpha
            np.full(n_samples, x_min_fixed),            # x_min (fixed)
            raw_samples[:, 1],                          # x_max
            raw_samples[:, 2],                          # mu
            raw_samples[:, 3],                          # sigma
        ])
    else:
        # Insert both x_min and x_max as fixed columns.
        samples_full = np.column_stack([
            raw_samples[:, 0],                          # alpha
            np.full(n_samples, x_min_fixed),            # x_min (fixed)
            np.full(n_samples, x_max_fixed),            # x_max (fixed)
            raw_samples[:, 1],                          # mu
            raw_samples[:, 2],                          # sigma
        ])

    return MCMCFit(
        samples=samples_full,
        acceptance_fraction=np.asarray(sampler.acceptance_fraction, dtype=float),
        log_prob=np.asarray(log_prob, dtype=float),
        pod=pod,
        fit_x_min=fit_x_min,
        fit_x_max=fit_x_max,
    )


# ---------------------------------------------------------------------------
# Grid evaluation helpers (q-space)
# ---------------------------------------------------------------------------


def pdf_on_grid(
    q_grid: ArrayLike,
    params: Sequence[float],
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
) -> ArrayLike:
    """Evaluate combined PDF on a grid of wind-normalized flux values.

    Computes f(q) for each grid point using the combined model.

    Args:
        q_grid (NDArray[np.float_]): Grid of q values.
        params (Sequence[float]): Full parameters [alpha, x_min, x_max, mu, sigma].
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.

    Returns:
        NDArray[np.float_]: PDF values on the grid.
    """
    alpha, x_min, x_max, mu, sigma = params
    return combined_pdf(q_grid, alpha, x_min, x_max, mu, sigma, pod=pod, pod_fn=pod_fn)


def pod_on_grid(
    q_grid: ArrayLike,
    params: Sequence[float],
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
) -> ArrayLike:
    """Evaluate PoD on a grid of wind-normalized flux values.

    Computes PoD(q) for each grid point using the selected PoD form.

    Args:
        q_grid (NDArray[np.float_]): Grid of q values.
        params (Sequence[float]): Full parameters [alpha, x_min, x_max, mu, sigma].
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.

    Returns:
        NDArray[np.float_]: PoD values on the grid.
    """
    _, _, _, mu, sigma = params
    pod_fn = resolve_pod_fn(pod, pod_fn)
    return pod_fn(np.asarray(q_grid, dtype=float), mu, sigma)


# ---------------------------------------------------------------------------
# Wind-speed marginalized PoD (flux space)
# ---------------------------------------------------------------------------


def pod_flux_on_grid(
    flux_grid: ArrayLike,
    params: Sequence[float],
    wind_speeds: ArrayLike,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
) -> ArrayLike:
    """Evaluate PoD vs flux, marginalized over a wind-speed distribution.

    For each flux value F, computes:
        PoD_flux(F) = (1/N) * sum_j PoD(F / w_j)

    Args:
        flux_grid (NDArray[np.float_]): Grid of flux values (Kg/h).
        params (Sequence[float]): Full parameters [alpha, x_min, x_max, mu, sigma].
        wind_speeds (NDArray[np.float_]): Reference wind-speed samples (m/s).
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.

    Returns:
        NDArray[np.float_]: Marginalized PoD values on the flux grid.
    """
    _, _, _, mu, sigma = params
    pod_fn = resolve_pod_fn(pod, pod_fn)
    flux_grid = np.asarray(flux_grid, dtype=float)
    wind_speeds = np.asarray(wind_speeds, dtype=float)
    # Ensure positive wind speeds.
    wind_speeds = wind_speeds[wind_speeds > 0]

    # q_matrix[i, j] = flux_grid[i] / wind_speeds[j]
    # Shape: (len(flux_grid), len(wind_speeds)).
    q_matrix = flux_grid[:, None] / wind_speeds[None, :]

    # Evaluate PoD for all (flux, wind) pairs.
    pod_matrix = pod_fn(q_matrix, mu, sigma)

    # Average over wind speeds (axis=1) to marginalize.
    return np.mean(pod_matrix, axis=1)


# ---------------------------------------------------------------------------
# PoD quantile helpers
# ---------------------------------------------------------------------------


def pod_quantile(
    p: float,
    mu: float,
    sigma: float,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
) -> float:
    """Wind-normalized flux q at which PoD reaches probability *p*.

    Inverts the PoD function analytically:

    - **Log-normal CDF**: ``q = exp(mu + sigma * Phi_inv(p))``
    - **Sigmoid**: ``q = exp(mu + sigma * logit(p))``

    Args:
        p (float): Target detection probability in (0, 1).
        mu (float): PoD location parameter in log space.
        sigma (float): PoD scale parameter in log space.
        pod (PoDType): PoD functional form.

    Returns:
        float: The q value where PoD(q) = p.
    """
    if pod == PoDType.LOGNORMAL_CDF:
        # Phi((log q - mu) / sigma) = p  =>  log q = mu + sigma * Phi_inv(p)
        return float(np.exp(mu + sigma * norm.ppf(p)))
    elif pod == PoDType.SIGMOID:
        # logistic((log q - mu) / sigma) = p  =>  log q = mu + sigma * logit(p)
        return float(np.exp(mu + sigma * logit(p)))
    else:
        raise ValueError(f"Unsupported PoDType: {pod}")


def pod_quantile_from_params(
    p: float,
    params: Sequence[float],
    pod: PoDType = PoDType.LOGNORMAL_CDF,
) -> float:
    """Convenience wrapper: extract mu, sigma from a full param vector.

    Args:
        p (float): Target detection probability in (0, 1).
        params (Sequence[float]): Full parameters [alpha, x_min, x_max, mu, sigma].
        pod (PoDType): PoD functional form.

    Returns:
        float: The q value where PoD(q) = p.
    """
    _, _, _, mu, sigma = params
    return pod_quantile(p, mu, sigma, pod=pod)


def pod_quantile_flux(
    p: float,
    params: Sequence[float],
    wind_speeds: ArrayLike,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    flux_range: Tuple[float, float] = (10.0, 100_000.0),
    n_grid: int = 2000,
) -> float:
    """Flux value at which the wind-marginalized PoD reaches probability *p*.

    Since the wind-marginalized PoD has no closed-form inverse, this function
    evaluates it on a dense log-spaced grid and interpolates.

    Args:
        p (float): Target detection probability in (0, 1).
        params (Sequence[float]): Full parameters [alpha, x_min, x_max, mu, sigma].
        wind_speeds (NDArray[np.float_]): Reference wind-speed samples (m/s).
        pod (PoDType): PoD functional form.
        flux_range (tuple[float, float]): Min/max flux for the search grid.
        n_grid (int): Number of grid points for interpolation.

    Returns:
        float: The flux (Kg/h) where marginalized PoD(flux) ≈ p.
            Returns ``nan`` if the PoD never reaches *p* on the grid.
    """
    flux_grid = np.logspace(
        np.log10(flux_range[0]), np.log10(flux_range[1]), n_grid
    )
    pod_vals = pod_flux_on_grid(flux_grid, params, wind_speeds, pod=pod)
    # Find first crossing above p.
    idx = np.searchsorted(pod_vals, p)
    if idx == 0 or idx >= len(flux_grid):
        return float("nan")
    # Linear interpolation in log-flux space between bracketing points.
    p_lo, p_hi = pod_vals[idx - 1], pod_vals[idx]
    f_lo, f_hi = np.log10(flux_grid[idx - 1]), np.log10(flux_grid[idx])
    frac = (p - p_lo) / (p_hi - p_lo)
    return float(10 ** (f_lo + frac * (f_hi - f_lo)))


# ---------------------------------------------------------------------------
# Posterior curve summaries
# ---------------------------------------------------------------------------


def posterior_pdf_ci(
    q_grid: ArrayLike,
    samples: ArrayLike,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
    ci: Tuple[float, float] = (0.05, 0.95),
) -> CurveSummary:
    """Posterior PDF median and CI bands on a q-grid.

    Aggregates PDF curves from posterior samples to compute quantiles.

    Args:
        q_grid (NDArray[np.float_]): Grid of wind-normalized flux values.
        samples (NDArray[np.float_]): Posterior samples (n_samples, 4).
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.
        ci (tuple[float, float]): Lower and upper quantiles.

    Returns:
        CurveSummary: Median and CI bands for the PDF.
    """
    pdf_matrix = np.array(
        [pdf_on_grid(q_grid, s, pod=pod, pod_fn=pod_fn) for s in samples]
    )
    median = np.quantile(pdf_matrix, 0.5, axis=0)
    lower = np.quantile(pdf_matrix, ci[0], axis=0)
    upper = np.quantile(pdf_matrix, ci[1], axis=0)
    return CurveSummary(median=median, lower=lower, upper=upper, ci=ci)


def posterior_pod_ci(
    q_grid: ArrayLike,
    samples: ArrayLike,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
    ci: Tuple[float, float] = (0.05, 0.95),
) -> CurveSummary:
    """Posterior PoD median and CI bands on a q-grid.

    Aggregates PoD(q) curves from posterior samples to compute quantiles.

    Args:
        q_grid (NDArray[np.float_]): Grid of wind-normalized flux values.
        samples (NDArray[np.float_]): Posterior samples (n_samples, 4).
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.
        ci (tuple[float, float]): Lower and upper quantiles.

    Returns:
        CurveSummary: Median and CI bands for the PoD curve in q-space.
    """
    pod_matrix = np.array(
        [pod_on_grid(q_grid, s, pod=pod, pod_fn=pod_fn) for s in samples]
    )
    median = np.quantile(pod_matrix, 0.5, axis=0)
    lower = np.quantile(pod_matrix, ci[0], axis=0)
    upper = np.quantile(pod_matrix, ci[1], axis=0)
    return CurveSummary(median=median, lower=lower, upper=upper, ci=ci)


def posterior_pod_flux_ci(
    flux_grid: ArrayLike,
    samples: ArrayLike,
    wind_speeds: ArrayLike,
    pod: PoDType = PoDType.LOGNORMAL_CDF,
    pod_fn: Optional[PODFunction] = None,
    ci: Tuple[float, float] = (0.05, 0.95),
) -> CurveSummary:
    """Posterior PoD-vs-flux median and CI bands, marginalized over wind speed.

    For each posterior sample, computes PoD_flux(F) by averaging PoD(F/w)
    over the reference wind-speed distribution, then takes quantiles.

    Args:
        flux_grid (NDArray[np.float_]): Grid of flux values (Kg/h).
        samples (NDArray[np.float_]): Posterior samples (n_samples, 4).
        wind_speeds (NDArray[np.float_]): Reference wind-speed samples (m/s).
        pod (PoDType): Named PoD selection.
        pod_fn (Callable | None): Custom PoD function override.
        ci (tuple[float, float]): Lower and upper quantiles.

    Returns:
        CurveSummary: Median and CI bands for PoD vs flux.
    """
    pod_matrix = np.array(
        [pod_flux_on_grid(flux_grid, s, wind_speeds, pod=pod, pod_fn=pod_fn)
         for s in samples]
    )
    median = np.quantile(pod_matrix, 0.5, axis=0)
    lower = np.quantile(pod_matrix, ci[0], axis=0)
    upper = np.quantile(pod_matrix, ci[1], axis=0)
    return CurveSummary(median=median, lower=lower, upper=upper, ci=ci)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_density_comparison(
    q_values: ArrayLike,
    mle_fit: MLEFit,
    mcmc_fit: MCMCFit,
    label: str,
    bins: ArrayLike,
    q_grid: ArrayLike,
    ci: Tuple[float, float] = (0.05, 0.95),
) -> None:
    """Plot side-by-side MLE and MCMC density fits against histogram.

    Creates a two-panel figure comparing the fitted density from MLE (left)
    and MCMC (right) overlaid on the histogram of observed wind-normalized
    flux rates q = flux / wind_speed.

    Args:
        q_values (NDArray[np.float_]): Observed q values for histogram.
        mle_fit (MLEFit): MLE fit result.
        mcmc_fit (MCMCFit): MCMC fit result.
        label (str): Title label for the plot (e.g., group name).
        bins (NDArray[np.float_]): Histogram bin edges (in q-space).
        q_grid (NDArray[np.float_]): Grid of q values for evaluating PDF.
        ci (tuple[float, float]): Lower and upper quantiles for MCMC CI.

    Returns:
        None: Displays the plot.
    """
    # Create a two-panel figure with shared y-axis for density comparison.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # --- Left panel: MLE fit ---
    ax = axes[0]
    # Plot histogram of observed q values as semi-transparent bars.
    ax.hist(q_values, bins=bins, density=True, alpha=0.3, label="Histogram")
    # Evaluate and overlay the fitted PDF using MLE parameters.
    mle_pdf = pdf_on_grid(q_grid, mle_fit.params, pod=mle_fit.pod)
    ax.plot(q_grid, mle_pdf, color="C1", label="MLE")
    # Log-log scale reveals the power-law behavior.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q = flux / wind_speed (Kg/h per m/s)")
    ax.set_ylabel("Density")
    ax.set_title(f"{label} — MLE")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    # --- Right panel: MCMC fit with confidence interval ---
    ax = axes[1]
    # Same histogram for comparison.
    ax.hist(q_values, bins=bins, density=True, alpha=0.3, label="Histogram")
    # Compute median and CI bands from posterior samples.
    pdf_ci = posterior_pdf_ci(q_grid, mcmc_fit.samples, pod=mcmc_fit.pod, ci=ci)
    # Overlay median PDF and shaded CI region.
    ax.plot(q_grid, pdf_ci.median, color="C2", label="MCMC median")
    ax.fill_between(
        q_grid, pdf_ci.lower, pdf_ci.upper,
        color="C2", alpha=0.2, label=f"MCMC CI {ci}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q = flux / wind_speed (Kg/h per m/s)")
    ax.set_title(f"{label} — MCMC")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    fig.suptitle("Power-law × PoD density fit (q-space)")
    plt.tight_layout()
    plt.show()


def plot_pod_curve(
    mcmc_fit: MCMCFit,
    label: str,
    flux_grid: ArrayLike,
    wind_speeds: ArrayLike,
    ci: Tuple[float, float] = (0.05, 0.95),
) -> None:
    """Plot the PoD curve vs flux with MCMC confidence interval bands.

    The PoD is marginalized over the reference wind-speed distribution so
    the x-axis shows flux (Kg/h) rather than wind-normalized flux q.

    Args:
        mcmc_fit (MCMCFit): MCMC fit result with posterior samples.
        label (str): Title label for the plot (e.g., group name).
        flux_grid (NDArray[np.float_]): Grid of flux values (Kg/h).
        wind_speeds (NDArray[np.float_]): Reference wind-speed samples (m/s).
        ci (tuple[float, float]): Lower and upper quantiles for CI bands.

    Returns:
        None: Displays the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    # Compute PoD vs flux marginalized over wind speeds.
    pod_ci = posterior_pod_flux_ci(
        flux_grid, mcmc_fit.samples, wind_speeds,
        pod=mcmc_fit.pod, ci=ci,
    )
    # Plot median PoD as a solid line.
    ax.plot(flux_grid, pod_ci.median, color="C0", label="PoD median")
    # Shade the CI region.
    ax.fill_between(
        flux_grid, pod_ci.lower, pod_ci.upper,
        color="C0", alpha=0.2, label=f"PoD CI {ci}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Flowrate (Kg/h)")
    ax.set_ylabel("PoD")
    ax.set_title(f"{label} — PoD vs Flux")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    plt.tight_layout()
    plt.show()


__all__ = [
    "X_MIN_DEFAULT",
    "X_MAX_DEFAULT",
    "PoDType",
    "ParameterBounds",
    "MLEFit",
    "MCMCFit",
    "CurveSummary",
    "lognormal_cdf_pod",
    "sigmoid_pod",
    "power_law_pdf",
    "combined_pdf",
    "log_likelihood",
    "negative_log_likelihood",
    "default_initial_guess",
    "default_bounds",
    "fit_mle",
    "log_prior",
    "log_posterior",
    "fit_mcmc",
    "pdf_on_grid",
    "pod_on_grid",
    "pod_flux_on_grid",
    "posterior_pdf_ci",
    "posterior_pod_ci",
    "posterior_pod_flux_ci",
    "pod_quantile",
    "pod_quantile_from_params",
    "pod_quantile_flux",
    "plot_density_comparison",
    "plot_pod_curve",
]
