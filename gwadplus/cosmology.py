"""Cosmological distances and GW orbital mechanics."""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .constants import (Omega_M, Omega_L, c_km_s, H0_km_s_Mpc,
                         KPC_TO_SECONDS, TSun, YEAR_IN_SEC)

# ── Comoving distance look-up table (built at import time) ───────────────────
_z_grid = np.concatenate([np.linspace(0, 0.1, 100), np.linspace(0.1, 10, 200)[1:]])

def _E(z):
    return np.sqrt(Omega_M * (1 + z)**3 + Omega_L)

_dc_vals   = [0] + [quad(lambda z: c_km_s / (H0_km_s_Mpc * _E(z)), 0, zi)[0]
                    for zi in _z_grid[1:]]
_dc_interp = interp1d(_z_grid, _dc_vals, kind='cubic', fill_value='extrapolate')


def DLz(z):
    """Luminosity distance [Mpc]."""
    return (1 + z) * _dc_interp(z) * 1000.0


def DVc(z):
    """Comoving volume element dV_c/dz [Mpc^3]."""
    dz   = 1e-4
    dcdz = (_dc_interp(z + dz) - _dc_interp(z - dz)) / (2 * dz)
    return 4 * np.pi * (_dc_interp(z) * 1000)**2 * (dcdz * 1000)


def m1m2(Mc, eta):
    """Individual masses from chirp mass and symmetric mass ratio."""
    s      = np.sqrt(np.maximum(1.0 - 4.0 * eta, 0.0))
    A_term = 1.0 + s + eta * (-5.0 - 3.0*s + (5.0 + s) * eta)
    A5     = np.power(A_term, 1.0/5.0)
    denom  = np.power(2.0, 1.0/5.0) * np.power(eta, 3.0/5.0)
    m1 = Mc * A5 / denom
    m2 = -Mc * (-1.0 + s + 2.0*eta) * A5 / (
         2.0 * np.power(2.0, 1.0/5.0) * np.power(eta, 8.0/5.0))
    return m1, m2


def residence_time(f_b, Mc_Msun, z, f_ref=1e-20, alpha=8/3, beta=5/8):
    """GW + environmental hardening residence time [s]."""
    Mc_sec = Mc_Msun * TSun
    t_GW   = (5.0/64.0) * (1+z) * Mc_sec**(-5.0/3.0) * (2*np.pi*f_b)**(-8.0/3.0)
    if f_ref < 1e-15:   # GW-only: t_env >> t_GW, harmonic mean → (2/3)*t_GW
        return (2.0/3.0) * t_GW
    f_env  = f_ref * (Mc_Msun / 1e9)**(-beta)
    t_env  = t_GW * (2 * f_b / f_env)**alpha
    return (2.0/3.0) / (1.0/t_GW + 1.0/t_env)
