# GWAD Precomputation: How It Became Fast

## What the code computes

For each source-frequency bin, `_gwad_density` evaluates the GW amplitude distribution:

$$\frac{dN}{dA\, d\ln f} = \int_0^\infty \int_0^{1/4} R\!\left(m_1(M_c, \eta),\, m_2(M_c, \eta),\, z\right) \cdot K(\eta) \cdot \frac{0.6\, \tau(A, z)}{A} \cdot \frac{dV_c}{d\ln z} \; d\eta\; d\ln z$$

where:
- $R(m_1, m_2, z)$ is the SMBHB merger rate (ModelI: looked up from a precomputed 3D grid)
- $M_c(A, z)$ is the chirp mass implied by strain amplitude $A$ at redshift $z$
- $m_1, m_2$ are the individual masses derived from $M_c$ and mass ratio $\eta$
- $K(\eta)$ is the Jacobian kernel $1/(\eta\sqrt{1-4\eta})$
- $\tau(A, z)$ is the GW residence time

This integral must be evaluated at every point in a discrete amplitude grid (150 values of $A$), for every source-frequency bin (~600 calls total during precomputation).

---

## The original bottleneck

The naive implementation discretises all three integrals simultaneously. With:
- $n_A = 150$ amplitude points
- $n_z = 150$ redshift points
- $n_\eta = 40$ mass-ratio points

each call to `_gwad_density` built a **$(150 \times 150 \times 40) = 900{,}000$-element tensor** and evaluated $R(m_1, m_2, z)$ at all 900k points via `scipy.RegularGridInterpolator`. This dominated the runtime.

### Profiling (before any optimisation)

| Step | Time/call | Share |
|------|-----------|-------|
| `rate_model` (scipy RGI on 900k pts) | 417 ms | 81.5% |
| `simpson` (nested integration) | 35 ms | 6.9% |
| `broadcast+m1m2` | 20 ms | 3.9% |
| `tau_2d` | 16 ms | 3.2% |
| **Total** | **511 ms** | |

With 16 parallel workers computing ~600 bins: **~20 s for precomputation**.

---

## Optimisation 1: Numba trilinear interpolator

`scipy.RegularGridInterpolator` on 900k points allocates a (900k, 3) intermediate array and uses binary search for each point. Replaced with a `@njit` scalar loop that computes the 3D grid index directly (O(1) arithmetic, no search, no allocation).

**Speedup on rate_model step: 30×** (417 ms → 14 ms steady-state).

---

## Optimisation 2: Precomputed m1/m2 coefficients

The `m1m2(Mc, η)` function converts chirp mass and mass ratio to individual masses. Expanding the formula:

$$m_1 = M_c \cdot f_1(\eta), \qquad m_2 = M_c \cdot f_2(\eta)$$

The functions $f_1(\eta)$ and $f_2(\eta)$ **depend only on $\eta$, not on $M_c$**. So instead of calling `m1m2` on the full (150 × 150 × 40) tensor every call (which involves six `np.power` operations on 900k elements), the 40 coefficients are computed once at module import and stored as `_M1C_GWAD`, `_M2C_GWAD`. Per-call cost reduces to two broadcasts.

---

## Optimisation 3: Precomputed integration weights + einsum

`scipy.integrate.simpson` carries Python overhead per call. The two nested calls on the (150 × 150 × 40) tensor were replaced with a single precomputed 2D trapezoid weight array `_W2D_GWAD` (shape 150 × 40) and one `np.einsum`:

```python
result = np.einsum('ijk,jk->i', integrand, _W2D_GWAD)
```

This maps to a single BLAS contraction. **Speedup on integration step: 5×**.

---

## Optimisation 4: Fast path for GW-only residence time

When running without environmental hardening (`f_ref = 1e-20`), the environmental term $t_\text{env} \gg t_\text{GW}$, so the harmonic mean reduces exactly to $\tfrac{2}{3} t_\text{GW}$. Two expensive power operations are skipped with an early return.

---

## The key insight: factoring out the η integral

After the above optimisations, profiling revealed a new structure:

| Step | Time/call |
|------|-----------|
| `rate_model` | 57 ms |
| `integrand` | 58 ms |
| `tau_2d` | 49 ms |

These were all slower than before—because fixing the `simpson` bottleneck unblocked all 16 workers to hit `rate_model` and `integrand` simultaneously, saturating memory bandwidth. The 900k-element tensor was still the root cause.

### The factorisation

Notice that the $\eta$ integration only depends on $M_c$ and $z$, not on the frequency $f$ or amplitude $A$ directly:

$$R_\text{eff}(M_c, z) \;=\; \int_0^{1/4} R\!\left(m_1(M_c, \eta),\, m_2(M_c, \eta),\, z\right) \cdot K(\eta)\; d\eta$$

This is a **pure function of $(M_c, z)$**. Once $R_\text{eff}$ is known, the full integral simplifies to:

$$\frac{dN}{dA\, d\ln f} = \int R_\text{eff}(M_c(A, z),\, z) \cdot \frac{0.6\, \tau(A, z)}{A} \cdot \frac{dV_c}{d\ln z} \; d\ln z$$

This is a **2D integral** over $(A, z)$ — the $\eta$ dimension is gone entirely.

### Building the R_eff table

`ModelI._ensure_R_eff()` builds a $(300 \times 150)$ table of $R_\text{eff}(M_c, z)$ **once** at construction time:

1. Lay out a grid of 300 log-spaced $M_c$ values over $[10^5, 10^{13}]\, M_\odot$
2. For each $(M_c, z)$ pair, evaluate $R(m_1(M_c, \eta_k), m_2(M_c, \eta_k), z)$ at all 40 $\eta$ points using the numba trilinear kernel — vectorised as a single $(300 \times 40 \times 150) = 1.8\text{M}$-point batch
3. Contract over $\eta$ with the precomputed weights: `np.einsum('iej,e->ij', rate, W)`

**Cost: ~200 ms, paid once per ModelI instance.**

### Using R_eff in _gwad_density

Each call now:
1. Computes `Mc_2d` (shape 150 × 150) from $A$ and $z$
2. Looks up `R_eff_2d = R_eff_eval(Mc_2d, z)` — 1D linear interpolation in $\log M_c$, pure numpy, ~0.1 ms
3. Multiplies by $\tau$, $dV/d\ln z$, $0.6/A$ and integrates over $\ln z$ with a matrix-vector product

**Total: ~0.7 ms per call**, vs 511 ms originally.

---

## Final profiling

| Step | Before | After |
|------|--------|-------|
| `rate_model` | 417 ms | 0 ms |
| `tau_2d` | 49 ms | 0.21 ms |
| `R_eff_eval` + integrate | — | 0.32 ms |
| `f_b + Mc_2d` | 15 ms | 0.18 ms |
| **Total/call** | **511 ms** | **0.71 ms** |
| **R_eff build (once)** | — | **200 ms** |

**Overall speedup: ~700×.** Precomputation wall time: ~20 s → well under 1 s (dominated by multiprocessing overhead, not GWAD computation).

---

## Why this works

The η integral could be factored out because:

- $m_1$ and $m_2$ are functions of $(M_c, \eta)$ only — they do not depend on $A$, $f$, or $z$ independently
- $M_c$ depends on $(A, z, f)$ but only enters the rate through $(m_1, m_2) = M_c \cdot (f_1(\eta), f_2(\eta))$
- Therefore $R(m_1, m_2, z) = R(M_c \cdot f_1(\eta), M_c \cdot f_2(\eta), z)$ factors into a lookup in $(M_c, z)$ space after integrating over $\eta$

The 40-point η dimension (which multiplied the tensor size by 40×) can be collapsed into a precomputed table at the cost of one extra interpolation dimension.
