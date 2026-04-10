# GWADpy

Monte Carlo simulator for pulsar timing array (PTA) GW timing residuals, with PDF estimation and likelihood evaluation against observational data.

## How to cite

If you use GWADpy, please cite [arXiv:2604.08506](https://arxiv.org/abs/2604.08506).

This code builds on analysis first developed in [arXiv:2306.17021](https://arxiv.org/abs/2306.17021).

## What it does

1. Computes the GW amplitude distribution (GWAD) from a chosen merger rate model.
2. Simulates realisations of the complex timing residual vector across all PTA Fourier modes.
3. Estimates the residual PDF at each mode via a three-region composite: CLT low tail, KDE bulk, and a full GWAD integral high tail.
4. Produces a validation plot (PDF breakdown for k=1,7,14).
5. Optionally (`--variance`): computes `dP/d ln σ₀²`, plots it alongside a σ₀² violin vs NANOGrav 15yr data, and evaluates the overlap likelihood.

## Package layout

| Module | Contents |
|---|---|
| `constants.py` | Physical and cosmological constants |
| `cosmology.py` | Luminosity distance, comoving volume, GW orbital mechanics |
| `merger_rates.py` | `ModelI` (numerical), `ModelII` (analytic) |
| `gwad.py` | `BrokenPowerLawGWAD`, cosmological GWAD integral |
| `windows.py` | PTA window functions, GW response sampler |
| `simulator.py` | `GlobalResidualsSimulator` |
| `analysis.py` | `compute_pdfs`, `compute_variance_likelihood` |
| `plotting.py` | `make_validation_plot` |
| `sigma0.py` | `sample_sigma2`, `compute_sigma0_tail`, `composite_sigma0_pdf`, `make_variance_plot` |
| `_nb_kernels.py` | Numba JIT kernels for residual and σ₀ MC sampling |
| `__main__.py` | CLI entry point |

## Usage

Run from the parent directory:

```bash
python -m gwadpy [global options] {modelI|modelII|bpl} [model options]
```

### Global options

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `.` | Output directory |
| `--prefix` | `gw_residuals` | Filename prefix |
| `--t-obs` | `16.0` | Observation time [years] |
| `--n-modes` | `14` | Number of PTA Fourier modes |
| `--n-real` | `10000` | Monte Carlo realisations |
| `--n-strong` | `10` | Strong-source threshold per frequency bin |
| `--f-start` | `1e-10` | Source frequency grid start [Hz] |
| `--f-end` | `100e-9` | Source frequency grid end [Hz] |
| `--n-bins` | `103` | Number of source frequency bins |
| `--window` | `tophat` | Window function: `sinc`, `tm`, `tophat`, `whitened` |
| `--env-f-ref` | *(omit)* | Environmental hardening reference frequency [Hz]; omit for GW-only |
| `--env-alpha` | `8/3` | Environmental hardening frequency power-law index |
| `--env-beta` | `5/8` | Environmental hardening mass-ratio power-law index |
| `--n-workers` | auto | Worker threads for bin precomputation and simulation |
| `--kde-bw` | `0.1` | KDE bandwidth (in log10 space) |
| `--kde-max-pts` | `10000` | Maximum realisations used to fit each per-mode KDE |
| `--gaussian` | off | Treat all sources as Gaussian; disables strong sources and the high-residual tail |
| `--variance` | off | Also produce a combined `dP/d ln σ₀²` + σ₀² violin plot |
| `--variance-n-real` | `100000` | MC realisations for the σ₀² histogram bulk |
| `--pta-data-dir` | *(omit)* | Path to PTA data directory (enables likelihood) |

### Model options

**`modelI`** — semi-numerical rate from halo merger tree convolved with a log-normal MBH–halo relation:

```bash
python -m gwadpy modelI [--a 8.95] [--b 1.4] [--sigma 0.47] [--pbh 0.06]
```

The rate grid is built once and cached to `.cache/` for reuse.

**`modelII`** — analytic phenomenological rate:

```bash
python -m gwadpy modelII [--R0 4e-14] [--M-star 2.5e9] [--c -0.2] [--d 6.0] [--z0 0.3]
```

**`bpl`** — broken power-law GWAD specified directly (bypasses the merger rate integral):

```bash
python -m gwadpy bpl --N-b <float> --A-b <float> --p <float> --q <float> [--s 2.0]
```

### Examples

```bash
# ModelI with environmental hardening, compared to NANOGrav data
python -m gwadpy --env-f-ref 3e-13 --pta-data-dir ./30f_fs{hd}_ceffyl \
    modelI --a 8.95 --b 1.4 --sigma 0.47 --pbh 0.06

# ModelII, custom output location
python -m gwadpy --output-dir results --prefix run_mII modelII

# BPL GWAD, sinc window, fewer bins for a quick test
python -m gwadpy --window sinc --n-bins 30 --n-real 2000 \
    bpl --N-b 1e10 --A-b 1e-15 --p 4.0 --q 1.5
```

## Outputs

| File | Description |
|---|---|
| `<prefix>_pdfs.npz` | Arrays: `f_modes`, `dt_grids`, `pdf`, `tail_norm`, `dt_cross` |
| `<prefix>_validation.pdf` | Three-panel PDF breakdown for k=1,7,14 |
| `<prefix>_variance.npz` | Arrays: `f_modes`, `sw`, `s2_draws`, `univ_grid`, `tail_all` (with `--variance`) |
| `<prefix>_variance.pdf` | `dP/d ln σ₀²` panels + σ₀² violin vs PTA data (with `--variance`) |
| stdout | Per-mode and total σ₀² log-likelihood (if `--variance` and `--pta-data-dir` both given) |

## PTA data format

The directory passed to `--pta-data-dir` must contain:

- `density.npy` — shape `(1, n_modes, n_bins)`, log-probability values
- `log10rhogrid.npy` — shape `(n_bins,)`, log10(σ_k/s) grid
- `freqs.npy` — shape `(n_modes,)`, mode frequencies [Hz]

This matches the output format of the Ceffyl free-spectrum analysis (NANOGrav 15yr).

## PDF construction

The residual PDF at each mode is a three-region composite:

- **Low tail** (`|δt| < x_lo`): analytic Rayleigh form `A · |δt|²`, matching the CLT limit for many weak sources.
- **Bulk** (`x_lo ≤ |δt| ≤ x_hi`): KDE fitted to the Monte Carlo realisations.
- **High tail** (`|δt| > x_hi`): full GWAD integral

  `dP/d ln|δt| ≈ ∫ d|R| p(|R|) ∫ d ln f  A* · dN/(dA d ln f)|_{A* = 4πfx/(|R||w_k^+(f)|)}`

  which asymptotes to `Λ_k |δt|^{-3}` at large |δt|. The crossover `x_hi` is set at the last histogram bin containing at least `min_counts` samples, ensuring a data-driven transition. The tail normalisation Λ_k is computed deterministically via the window-function integral

  `Λ_k = (1/256π³) ∫ df/f⁴ C(f) |w_k^+(f)|³`

  with no Monte Carlo sampling.
