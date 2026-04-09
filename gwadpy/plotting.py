"""
Validation plot (section 8): violin plot of simulated residuals vs NANOGrav
data, plus a single-mode PDF breakdown (weak / strong / analytic tail / total).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

# Plotting Style — Publication Quality
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


# ── Internal helpers ──────────────────────────────────────────────────────────

def _lhist(data, n_bins=80):
    x = np.abs(data); x = x[x > 0]
    if len(x) < 5: return None, None
    bins = np.logspace(np.log10(x.min()/2), np.log10(x.max()*2), n_bins)
    dlnb = np.log(bins[1]/bins[0])
    cts, _ = np.histogram(x, bins=bins)
    return np.sqrt(bins[:-1]*bins[1:]), cts/(len(x)*dlnb)


def _stairs(ctr, pdf):
    dlnb  = np.log(ctr[1]/ctr[0])
    edges = np.exp(np.concatenate([np.log(ctr)-dlnb/2, [np.log(ctr[-1])+dlnb/2]]))
    return np.repeat(edges, 2), np.concatenate([[0], np.repeat(pdf, 2), [0]])


def _composite_pdf(mag_1d, C_tail, n_hist=80,
                   low_quantile=0.01, min_counts=15,
                   sim=None, ki=None, gaussian=False):
    """
    Build a three-region composite PDF:
      x < x_lo  : analytic A·x²  (Rayleigh/CLT low tail)
      x_lo–x_hi : empirical histogram (smoothed in log-space)
      x > x_hi  : GWAD integral (if sim given) or C_tail·x⁻³

    x_lo = low_quantile percentile of |δt|
    x_hi = last histogram bin with >= min_counts samples
    A    = 2·q / x_lo²  (exact for Rayleigh, model-free otherwise)
    """
    pos = np.abs(mag_1d); pos = pos[pos > 0]
    if len(pos) < 10:
        return None, None, None, None

    bins    = np.logspace(np.log10(pos.min()/2), np.log10(pos.max()*2), n_hist)
    dlnb    = np.log(bins[1]/bins[0])
    cts, _  = np.histogram(pos, bins=bins)
    pdf_mc  = cts / (len(pos) * dlnb)
    centers = np.sqrt(bins[:-1] * bins[1:])

    # High-x crossover: last bin with >= min_counts samples
    good  = np.where(cts >= min_counts)[0]
    ci_hi = int(good[-1]) if len(good) > 0 else len(centers) - 1

    # Low-x crossover: analytic A·x² tail
    x_lo  = float(np.quantile(pos, low_quantile))
    A     = 2 * low_quantile / x_lo**2
    ci_lo = max(np.searchsorted(centers, x_lo) - 1, 0)

    pdf_comp = pdf_mc.copy()
    if not gaussian:
        if sim is not None and ki is not None and ci_hi + 1 < len(centers):
            pdf_comp[ci_hi + 1:] = sim.compute_gwad_pdf(centers[ci_hi + 1:], ki)
        elif C_tail > 0:
            pdf_comp[ci_hi + 1:] = C_tail * centers[ci_hi + 1:]**(-3)
    pdf_comp[:ci_lo] = A * centers[:ci_lo]**2

    # Smooth only the bulk region in log-space
    bulk     = pdf_comp[ci_lo:ci_hi + 1]
    pos_mask = bulk > 0
    if pos_mask.sum() > 4:
        log_bulk_smooth = gaussian_filter1d(
            np.log(np.where(pos_mask, bulk, 1.0)), sigma=0.1)
        pdf_comp[ci_lo:ci_hi + 1] = np.where(pos_mask,
                                              np.exp(log_bulk_smooth), bulk)

    return centers, pdf_comp, pdf_mc, centers[ci_hi]


def _composite_rms(res_ki, C_tail, n_hist=80, low_quantile=0.01, min_counts=15,
                   gaussian=False):
    """RMS of the composite PDF: analytic low/high tails + empirical bulk."""
    pos  = np.abs(res_ki); pos = pos[pos > 0]
    x_lo = float(np.quantile(pos, low_quantile))
    A    = 2 * low_quantile / x_lo**2

    bins  = np.logspace(np.log10(pos.min()/2), np.log10(pos.max()*2), n_hist)
    cts, _ = np.histogram(pos, bins=bins)
    centers = np.sqrt(bins[:-1] * bins[1:])
    good  = np.where(cts >= min_counts)[0]
    x_hi  = float(centers[good[-1]]) if len(good) > 0 else float(pos.max())

    bulk_mask = (pos >= x_lo) & (pos <= x_hi)
    rms2_bulk = np.mean(pos[bulk_mask]**2) * bulk_mask.mean()
    rms2_low  = A * x_lo**4 / 4
    rms2_high = (C_tail / x_hi) if (C_tail > 0 and not gaussian) else 0.0

    return float(np.sqrt(rms2_low + rms2_bulk + rms2_high))


# ── Public function ───────────────────────────────────────────────────────────

def make_validation_plot(res, res_strong, res_weak, tail_norm,
                         ng_data, sim, out_path, n_real, n_strong, model_label,
                         gaussian=False):
    """
    Four-panel figure: PDF breakdown for k=1,7,14 (left×3) + violin across modes (right).
    """
    K_DEMOS = (1, 7, 14)
    x_pos_v = np.log10(sim.f_obs)

    fig = plt.figure(figsize=(14, 4.5))
    gs  = gridspec.GridSpec(1, 4, wspace=0.35, width_ratios=[1, 1, 1, 1.6])
    axes_pdf = [fig.add_subplot(gs[i]) for i in range(3)]
    ax_vln   = fig.add_subplot(gs[3])

    # ── Helper: compute visual slope in display coordinates ───────────────────
    def _line_angle(ax, x0, y0, x1, y1):
        d0 = ax.transData.transform((x0, y0))
        d1 = ax.transData.transform((x1, y1))
        return np.degrees(np.arctan2(d1[1] - d0[1], d1[0] - d0[0]))

    # ── PDF panels ────────────────────────────────────────────────────────────
    for col, K_DEMO in enumerate(K_DEMOS):
        ax_pdf = axes_pdf[col]
        ki     = K_DEMO - 1
        C_tail = tail_norm[ki]
        f_k    = K_DEMO / sim.T_obs
        show_ylabel = (col == 0)

        for data, color, label in [(res_weak[:, ki],   'C0', 'Weak'),
                                    (res_strong[:, ki], 'C3', 'Strong')]:
            ctr, pdf = _lhist(data)
            if ctr is not None:
                x, y = _stairs(ctr, pdf)
                ax_pdf.fill_between(x, y, alpha=0.30, color=color)
                ax_pdf.plot(x, y, '-', color=color, lw=1.2, alpha=0.85, label=label)

        ctr_t, pdf_comp, _, _ = _composite_pdf(res[:, ki], C_tail,
                                                sim=sim, ki=ki,
                                                gaussian=gaussian)
        if ctr_t is not None:
            ax_pdf.plot(ctr_t, pdf_comp, '-', color='k', lw=2.5, label='Total')

        # x² normalisation from low quantile
        _q   = 0.01
        _x_q = float(np.quantile(np.abs(res[:, ki]), _q))
        _A   = 2 * _q / _x_q**2

        ax_pdf.set(xscale='log', yscale='log',
                   xlabel=r'$|\tilde{\delta t}_k|$ [s]',
                   ylabel=(r'$d\rm{P}/d\ln|\tilde{\delta t}_k|$' if show_ylabel else ''),
                   title=rf'$k={K_DEMO}$,  $f={f_k*1e9:.1f}$ nHz')
        ax_pdf.autoscale(enable=True, axis='both', tight=False)
        _xlim = ax_pdf.get_xlim()
        _ylim = ax_pdf.get_ylim()
        ax_pdf.set_xlim(_xlim)
        ax_pdf.set_ylim(_ylim)

        # Reference lines extending across the full plot
        _ref_color = '0.45'
        _x_wide = np.logspace(np.log10(_xlim[0]) - 1, np.log10(_xlim[1]) + 1, 600)
        ax_pdf.plot(_x_wide, _A * _x_wide**2, '--', color=_ref_color,
                    lw=1.6, alpha=0.85, zorder=0)
        if not gaussian:
            _gwad_ref = sim.compute_gwad_pdf(_x_wide, ki)
            _gwad_pos = _gwad_ref > 0
            if _gwad_pos.any():
                ax_pdf.plot(_x_wide[_gwad_pos], _gwad_ref[_gwad_pos], '--',
                            color=_ref_color, lw=1.6, alpha=0.85, zorder=0)

        ax_pdf.tick_params(which='both', direction='in', top=True, right=True)
        if show_ylabel:
            ax_pdf.legend(loc='upper left', frameon=True, fancybox=False,
                          edgecolor='black')
        ax_pdf.grid(True, which='major', alpha=0.3)
        ax_pdf.grid(True, which='minor', alpha=0.15, linestyle=':')

        # Annotations
        _lx0, _lx1 = np.log10(_xlim[0]), np.log10(_xlim[1])
        _log_offset = 0.8
        _x2_slide   = 0.25
        _label_kw = dict(color='black', fontsize=9, ha='left', va='bottom',
                         rotation_mode='anchor',
                         bbox=dict(boxstyle='round,pad=0.0', fc='white',
                                   ec='none', alpha=0.9))

        _xa = 10**(_lx0 + _x2_slide * (_lx1 - _lx0))
        _ya_line = _A * _xa**2
        _ya = _ya_line * 10**0.4
        if _xlim[0] <= _xa <= _xlim[1] and _ylim[0] < _ya < _ylim[1]:
            _ang2 = _line_angle(ax_pdf, _xa, _ya_line, _xa * 2, _A * (_xa * 2)**2)
            ax_pdf.text(_xa, _ya, r'$\propto|\tilde{\delta t}|^{2}$',
                        rotation=_ang2, **_label_kw)

        if not gaussian and _gwad_pos.any():
            _xb = 10**(_lx0 + 0.85 * (_lx1 - _lx0))
            _xb2 = _xb * 2
            _yb_line = float(sim.compute_gwad_pdf(np.array([_xb]),  ki)[0])
            _yb2     = float(sim.compute_gwad_pdf(np.array([_xb2]), ki)[0])
            _yb = _yb_line * 10**_log_offset
            if _yb_line > 0 and _ylim[0] < _yb < _ylim[1]:
                _ang3 = _line_angle(ax_pdf, _xb, _yb_line, _xb2, _yb2)
                ax_pdf.text(_xb, _yb, r'$\propto|\tilde{\delta t}|^{-3}$',
                            rotation=_ang3, **_label_kw)

    # ── Right panel: violin plot ─────────────────────────────────────────────
    data_log = np.log10(np.abs(res[:20_000]) + 1e-30)
    parts = ax_vln.violinplot([data_log[:, i] for i in range(sim.n_modes)],
                               positions=x_pos_v, widths=0.03,
                               showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('C2'); pc.set_edgecolor('black')
        pc.set_alpha(0.55); pc.set_linewidth(1.2)

    if ng_data is not None:
        y_grid = np.linspace(-10, -4, 4000)
        for (fng, pdf_fn) in ng_data[:sim.n_modes]:
            xc = np.log10(fng)
            if xc < x_pos_v.min()-0.2 or xc > x_pos_v.max()+0.2:
                continue
            xd = pdf_fn(y_grid)
            ax_vln.fill_betweenx(y_grid, xc-xd*0.027, xc+xd*0.027,
                                  facecolor='#FF9900', edgecolor='black',
                                  linewidth=0.8, alpha=0.40, zorder=3)
            ax_vln.scatter(xc, y_grid[np.argmax(xd)], s=12, color='white', zorder=5)

    ref_x  = np.log10(ng_data[0][0]) if ng_data else x_pos_v[0]
    x_line = np.linspace(x_pos_v.min()-0.05, x_pos_v.max()+0.05, 100)
    ax_vln.plot(x_line, -13/6*(x_line-ref_x)-6.2, 'k--', lw=2.0, alpha=0.7)

    rms_log = np.array([np.log10(_composite_rms(res[:, i], tail_norm[i],
                                                gaussian=gaussian))
                        for i in range(sim.n_modes)])
    ax_vln.plot(x_pos_v, rms_log, '-.', color='C2', lw=1.8, label='RMS')

    ax_vln.set(xlabel=r'$\log_{10}(f\,[\mathrm{Hz}])$',
               ylabel=r'$\log_{10}(|\tilde{\delta t}|\,[\mathrm{s}])$',
               xlim=(x_pos_v.min()-0.10, x_pos_v.max()+0.10),
               ylim=(-8.5, -5.5))
    ax_vln.tick_params(which='both', direction='in', top=True, right=True)

    legend_handles = [
        Patch(facecolor='C2', edgecolor='black', alpha=0.6, label='Simulation'),
        Line2D([0],[0], color='black', lw=2, ls='--', label=r'Power law $-13/6$'),
        Line2D([0],[0], color='C2', lw=1.8, ls='-.', label='RMS mean'),
    ]
    if ng_data:
        legend_handles.insert(1, Patch(facecolor='#FF9900', edgecolor='black',
                                        alpha=0.5, label='NANOGrav 15yr'))
    ax_vln.legend(handles=legend_handles, loc='upper right', frameon=True,
                  fancybox=False, edgecolor='black', framealpha=0.95)

    _env = sim.env_params
    if _env:
        _f_ref = _env.get('f_ref', 1e-20)
        _alpha = _env.get('alpha', 8/3)
        _beta  = _env.get('beta',  5/8)
        _env_str = (rf',  $f_{{\rm ref}}={_f_ref*1e9:.1f}$ nHz,'
                    rf'  $\alpha={_alpha:.3g}$,  $\beta={_beta:.3g}$')
    else:
        _env_str = ',  GW only'
    plt.subplots_adjust(top=0.85)
    _gauss_str = r',  \textbf{Gaussian}' if gaussian else rf',  $n_{{\rm strong}}={n_strong}$'
    fig.suptitle(rf'{model_label}{_gauss_str},  '
                 rf'$N_{{\rm real}}={n_real:,}${_env_str}',
                 fontsize=15)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved validation plot -> {out_path}")
