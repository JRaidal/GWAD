"""
Post-simulation analysis: PDF estimation and likelihood.

compute_pdfs        — KDE + analytic tail stitching for each PTA mode.
compute_likelihood  — overlap integral L_k = ∫ p_model · p_data d(log10 x).
"""

from pathlib import Path
from time import time as _time

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d


def compute_pdfs(res, tail_norm, n_modes, f_modes,
                 kde_bw=0.1, n_eval=1000, n_hist=80,
                 min_counts=15, low_quantile=0.01,
                 n_kde_max=10_000, sim=None, gaussian=False,
                 verbose=True):
    """
    Build a three-region composite PDF for |δt_k| at each mode:
      x < x_lo  : analytic A·x²              (CLT low tail)
      x_lo–x_hi : KDE body
      x > x_hi  : GWAD integral (if sim given) or C_tail·x⁻³

    The high-x crossover x_hi is the position of the last histogram bin
    with >= min_counts samples — robust regardless of n_real.

    Parameters
    ----------
    res        : (n_real, n_modes) complex residuals
    tail_norm  : (n_modes,) asymptotic tail coefficient Λ_k (fallback only)
    n_modes    : number of PTA modes
    f_modes    : (n_modes,) mode frequencies [Hz]
    kde_bw     : KDE bandwidth
    n_eval     : evaluation points per mode
    n_hist     : histogram bins for crossover detection
    min_counts : minimum samples per bin before switching to tail
    low_quantile : quantile defining low-x crossover (default 0.01)
    n_kde_max  : max samples used to fit each KDE
    sim        : GlobalResidualsSimulator — if provided, uses compute_gwad_pdf()
                 for the tail region instead of the pure C_tail·x⁻³ power law
    gaussian   : if True, disable the high-x power-law tail entirely (pure KDE)

    Returns
    -------
    dt_grids : (n_modes, n_eval)  |δt| evaluation grid [s]
    pdf_out  : (n_modes, n_eval)  dP/d(ln|δt|)
    dt_cross : (n_modes,)         high-x crossover values [s]
    """
    dt_grids = np.zeros((n_modes, n_eval))
    pdf_out  = np.zeros((n_modes, n_eval))
    dt_cross = np.zeros(n_modes)

    _t_total = _time()

    for ki in range(n_modes):
        C_tail  = tail_norm[ki]
        samples = np.abs(res[:, ki]); samples = samples[samples > 0]

        kde_in  = (samples if len(samples) <= n_kde_max
                   else np.random.choice(samples, n_kde_max, replace=False))
        kde     = gaussian_kde(np.log10(kde_in), bw_method=kde_bw)

        dt_lo   = np.percentile(samples, 0.05)
        dt_hi   = np.percentile(samples, 99.9) * 500
        dt_grid = np.logspace(np.log10(dt_lo), np.log10(dt_hi), n_eval)
        pdf     = kde(np.log10(dt_grid)) / np.log(10)   # dP/d(ln x)

        # High-x crossover: last histogram bin with >= min_counts samples
        bins_h = np.logspace(np.log10(samples.min() / 2),
                             np.log10(samples.max() * 2), n_hist)
        cts, _ = np.histogram(samples, bins=bins_h)
        centers_h = np.sqrt(bins_h[:-1] * bins_h[1:])
        good = np.where(cts >= min_counts)[0]
        x_hi = float(centers_h[good[-1]]) if len(good) > 0 else dt_grid[-1]
        ic_hi = min(np.searchsorted(dt_grid, x_hi), len(dt_grid) - 1)
        x_hi  = dt_grid[ic_hi]

        if not gaussian:
            if sim is not None and ic_hi + 1 < len(dt_grid):
                pdf[ic_hi:] = sim.compute_gwad_pdf(dt_grid[ic_hi:], ki)
            elif C_tail > 0:
                pdf[ic_hi:] = C_tail * dt_grid[ic_hi:]**(-3)

        # Low-x crossover: analytic A·x² tail (A = 2q / x_lo²)
        x_lo  = float(np.quantile(samples, low_quantile))
        A     = 2 * low_quantile / x_lo**2
        ic_lo = max(np.searchsorted(dt_grid, x_lo) - 1, 0)
        pdf[:ic_lo] = A * dt_grid[:ic_lo]**2

        dt_grids[ki] = dt_grid
        pdf_out[ki]  = pdf
        dt_cross[ki] = x_hi

    if verbose:
        print(f"  Total KDE+tail: {_time() - _t_total:.2f}s")

    return dt_grids, pdf_out, dt_cross


def compute_likelihood(dt_grids, pdf_model, pta_data_dir, f_model, n_modes):
    """
    Compute the overlap likelihood between the model PDFs and PTA data.

    For each mode k:
        L_k = ∫ p_model(log10 x) · p_data(log10 x) d(log10 x)

    Total likelihood is the product over modes:
        L_total = ∏_k L_k

    Parameters
    ----------
    dt_grids     : (n_modes, n_eval)  |δt| grid [s]
    pdf_model    : (n_modes, n_eval)  dP/d(ln|δt|)
    pta_data_dir : path to directory containing
                     density.npy        — log-probability array, shape (1, n_ng, n_bins)
                     log10rhogrid.npy   — log10(|δt|/s) grid, shape (n_bins,)
                     freqs.npy          — mode frequencies [Hz], shape (n_ng,)
    f_model      : (n_modes,)  model mode frequencies [Hz]
    n_modes      : int

    Returns
    -------
    log_L_modes : (n_modes,) per-mode log-likelihoods (nan where skipped)
    log_L_total : scalar, sum of finite entries
    """
    data_dir = Path(pta_data_dir)
    prob_raw = np.load(data_dir / 'density.npy')[0]   # (n_ng, n_bins)
    L10rho   = np.load(data_dir / 'log10rhogrid.npy') # (n_bins,)
    fNG      = np.load(data_dir / 'freqs.npy')        # (n_ng,)

    # Match each NG frequency to the nearest model mode
    paired = [(int(np.argmin(np.abs(f_model - fj))), j)
              for j, fj in enumerate(fNG[:n_modes])]

    log_L_modes = np.full(n_modes, np.nan)
    for k, j in paired:
        p_raw = np.exp(prob_raw[j])
        norm  = np.trapz(p_raw, L10rho)
        if norm <= 0:
            continue
        p_data_L10 = p_raw / norm                        # dP/d(log10 x), normalised

        lx_mod    = np.log10(dt_grids[k])
        p_mod_L10 = pdf_model[k] * np.log(10)            # dP/d(ln x) -> dP/d(log10 x)

        x_lo = max(lx_mod.min(), L10rho.min())
        x_hi = min(lx_mod.max(), L10rho.max())
        if x_lo >= x_hi:
            continue

        x_comm  = np.linspace(x_lo, x_hi, 3000)
        p_mod_i = interp1d(lx_mod,  p_mod_L10,  kind='linear',
                           bounds_error=False, fill_value=0.0)(x_comm)
        p_dat_i = interp1d(L10rho, p_data_L10, kind='linear',
                           bounds_error=False, fill_value=0.0)(x_comm)

        L_k = np.trapz(p_mod_i * p_dat_i, x_comm)
        log_L_modes[k] = np.log(L_k) if L_k > 0 else -np.inf

    valid       = np.isfinite(log_L_modes)
    log_L_total = float(np.sum(log_L_modes[valid]))
    return log_L_modes, log_L_total
