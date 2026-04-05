"""Physical and cosmological constants."""

# SI mechanics
G_SI        = 6.674e-11          # gravitational constant [m^3 kg^-1 s^-2]
c_SI        = 2.998e8            # speed of light [m/s]
Msun_SI     = 1.989e30           # solar mass [kg]
YEAR_IN_SEC = 365.25 * 24 * 3600 # Julian year [s]

# GW units
TSun = G_SI * Msun_SI / c_SI**3  # time of 1 solar mass [s]

# Distance conversions
KPC_TO_METERS  = 3.086e19
KPC_TO_SECONDS = KPC_TO_METERS / c_SI
MPC_TO_KPC     = 1000.0

# Planck 2018 cosmology
h_cosmo      = 0.674
H0_SI        = 100 * h_cosmo * 1000 / (1e6 * 3.086e16)  # [s^-1]
Omega_M      = 0.315
Omega_L      = 1 - Omega_M
c_km_s       = 2.998e5
H0_km_s_Mpc  = 100 * h_cosmo
