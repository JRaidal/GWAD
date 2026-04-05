"""GWAD+ — GW timing residual analysis package."""

from .merger_rates import ModelI, ModelII
from .gwad import BrokenPowerLawGWAD, calculate_gwad
from .windows import WINDOWS, sample_R, sample_absR, R_MEAN_SQ, _R_CDF_AXIS, _R_VALS_AXIS
from .simulator import GlobalResidualsSimulator
from .analysis import compute_pdfs, compute_likelihood
from .plotting import make_validation_plot
