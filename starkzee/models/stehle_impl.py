"""
Stehle (MMM) Stark-Zeeman-Doppler lineshape.

Based on pystark package (https://github.com/jsallcock/pystark).
"""

import os
import numpy as np
from scipy.constants import c as C, e as E, k as K, m_e as M_E
try:
    from numpy import trapezoid as trapz
except ImportError:
    from numpy import trapz
from scipy.io import netcdf_file
from scipy.signal import fftconvolve

_NC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'stehle_tables.nc'))
_NC = None
try:
    _NC = netcdf_file(_NC_PATH, 'r', mmap=True)
except Exception:
    pass


def _available_transitions():
    """Return the upper/lower pairs actually present in the bundled table."""
    if _NC is None:
        return ()
    transitions = []
    for variable_name in _NC.variables:
        if not variable_name.endswith('_tempe'):
            continue
        fields = variable_name.split('_')
        if len(fields) == 4 and fields[0] == 'n':
            transitions.append((int(fields[1]), int(fields[2])))
    return tuple(sorted(transitions, key=lambda pair: (pair[1], pair[0])))


_AVAILABLE_TRANSITIONS = _available_transitions()


def _axis_weights(nodes, value, name):
    """Return one exact-node weight or two linear interpolation weights."""
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or len(nodes) < 2 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError(f'Stehle: invalid {name} coordinate in table')
    exact = np.flatnonzero(np.isclose(nodes, value, rtol=1e-12, atol=0.0))
    if exact.size:
        return ((int(exact[0]), 1.0),)
    if value < nodes[0] or value > nodes[-1]:
        raise ValueError(
            f'Stehle: {name} {value:.6g} outside table range '
            f'[{nodes[0]:.6g}, {nodes[-1]:.6g}]'
        )

    upper = int(np.searchsorted(nodes, value, side='right'))
    lower = upper - 1
    fraction = (value - nodes[lower]) / (nodes[upper] - nodes[lower])
    return ((lower, 1.0 - fraction), (upper, fraction))


# ── Fortran FINTRP: three-point hyperbolic/quadratic interpolation ─────────────
def _fintrp(x1, x2, x3, y1, y2, y3, x):
    if x == x2:
        return y2
    a12 = x1 - x2;  a22 = x1 - x3
    v1  = y1 - y2;  v2  = y1 - y3
    if ((y1 < y2 < y3) or (y1 > y2 > y3)):
        deter = v1 * a22 - v2 * a12
        if abs(deter) < 1e-40:
            return y1 + (x - x1) * (y3 - y1) / (x3 - x1)
        a21 = x1 * y1
        a11 = a21 - x2 * y2
        a21 = a21 - x3 * y3
        c   = (a22 * a11 - a12 * a21) / deter
        a   = (-v2 * a11 + v1 * a21) / deter
        b   = (y1 - a) * (x1 - c)
        return a + b / (x - c)
    else:
        x1c = x1 * x1
        a11 = x1c - x2 * x2
        a21 = x1c - x3 * x3
        deter = a11 * a22 - a12 * a21
        if abs(deter) < 1e-40:
            raise ValueError('FINTRP: degenerate inputs')
        a = (a22 * v1 - a12 * v2) / deter
        b = (-a21 * v1 + a11 * v2) / deter
        return (a * x + b) * x + (y1 - a * x1c - b * x1)


def _fintrp_grid(x_values, y_values, targets, wing_constant=None):
    """Evaluate a positive-detuning profile with the CDS ``FINTRP`` rule.

    The Stehle reader emits a deliberately sparse common detuning grid.  A
    subsequent linear interpolation across that grid creates broad artificial
    shelves in the line wings.  Evaluate each requested detuning with the same
    three-point hyperbolic/quadratic interpolator used to construct the common
    grid instead.  Beyond the final point, use the published Holtsmark wing
    when its frequency-space constant is supplied.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    result = np.empty_like(targets)

    if len(x_values) < 3 or np.any(np.diff(x_values) <= 0.0):
        raise ValueError("FINTRP grid requires at least three increasing detunings")

    for output_index, target in np.ndenumerate(targets):
        if target <= x_values[0]:
            result[output_index] = y_values[0]
            continue
        if target > x_values[-1]:
            result[output_index] = (
                0.0
                if wing_constant is None
                else wing_constant / target**2.5
            )
            continue

        upper = int(np.searchsorted(x_values, target, side='left'))
        if target == x_values[upper]:
            result[output_index] = y_values[upper]
            continue
        if upper >= len(x_values) - 1:
            indices = (-3, -2, -1)
        else:
            indices = (upper - 1, upper, upper + 1)
        x1, x2, x3 = (x_values[index] for index in indices)
        y1, y2, y3 = (y_values[index] for index in indices)
        value = _fintrp(x1, x2, x3, y1, y2, y3, target)

        # FINTRP can become singular or overshoot below zero for nearly
        # degenerate triples.  Preserve a physical positive profile with a
        # local log-log fallback; this is inactive for the published H-alpha
        # grid but makes the table interface robust for all transitions.
        if not np.isfinite(value) or value <= 0.0:
            lower = upper - 1
            fraction = ((np.log(target) - np.log(x_values[lower]))
                        / (np.log(x_values[upper]) - np.log(x_values[lower])))
            value = np.exp(np.log(y_values[lower])
                           + fraction * (np.log(y_values[upper])
                                         - np.log(y_values[lower])))
        result[output_index] = value
    return result


def _log_log_grid(x_values, y_values, targets, wing_constant=None):
    """Interpolate a positive profile in log-log space.

    The center node has zero detuning, so its first interval cannot be
    represented logarithmically and is interpolated linearly.  Outside the
    tabulated detuning range, use the published Holtsmark wing when its
    frequency-space constant is supplied.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if len(x_values) < 2 or np.any(np.diff(x_values) <= 0.0):
        raise ValueError("log-log grid requires at least two increasing detunings")
    if np.any(y_values <= 0.0):
        raise ValueError("log-log interpolation requires positive profile values")

    result = np.empty_like(targets)
    center = targets <= x_values[0]
    wing = targets > x_values[-1]
    interior = ~(center | wing)
    result[center] = y_values[0]
    result[wing] = (
        0.0
        if wing_constant is None
        else wing_constant / targets[wing]**2.5
    )

    if np.any(interior):
        interior_targets = targets[interior]
        upper = np.searchsorted(x_values, interior_targets, side='left')
        first_interval = upper == 1
        interpolated = np.empty_like(interior_targets)
        interpolated[first_interval] = np.interp(
            interior_targets[first_interval], x_values[:2], y_values[:2]
        )

        logarithmic = ~first_interval
        if np.any(logarithmic):
            logarithmic_upper = upper[logarithmic]
            logarithmic_lower = logarithmic_upper - 1
            log_target = np.log(interior_targets[logarithmic])
            log_x_lower = np.log(x_values[logarithmic_lower])
            fraction = (
                (log_target - log_x_lower)
                / (np.log(x_values[logarithmic_upper]) - log_x_lower)
            )
            log_y_lower = np.log(y_values[logarithmic_lower])
            interpolated[logarithmic] = np.exp(
                log_y_lower
                + fraction
                * (np.log(y_values[logarithmic_upper]) - log_y_lower)
            )
        result[interior] = interpolated
    return result


def _interpolate_grid(x_values, y_values, targets, wing_constant, method):
    """Evaluate one Stehle table node with the selected interpolation rule."""
    if method == 'log-log':
        return _log_log_grid(x_values, y_values, targets, wing_constant)
    if method == 'fintrp':
        return _fintrp_grid(x_values, y_values, targets, wing_constant)
    raise ValueError("interpolation must be 'log-log' or 'fintrp'")


def _compute_stehle_stark(n_u, n_l, Ne_m3, Te_ev, wl_center_m, freq_axis,
                           interpolation='log-log'):
    """Pure Stark profile on *freq_axis* [Hz], area-normalized to 1.

    Read the local Stehle tables and return a profile in 1/Hz.

    ``interpolation`` selects direct log-log interpolation (the default) or
    the CDS reader's three-point ``'fintrp'`` rule between detuning nodes.
    """
    if _NC is None:
        raise FileNotFoundError(f"stehle_tables.nc not found at {_NC_PATH}")

    transition = (n_u, n_l)
    if transition not in _AVAILABLE_TRANSITIONS:
        raise ValueError(
            f'Stehle: transition {n_u}→{n_l} is not in the bundled table; '
            f'supported lower levels are 1, 2, and 3 with n_u <= 30'
        )

    temp_k  = Te_ev * E / K
    dens_cm = Ne_m3 * 1e-6
    prefix  = f'n_{n_u}_{n_l}_'

    tab_temp_k   = np.array(_NC.variables[prefix + 'tempe'].data)
    num_tab_dens = int(np.asarray(_NC.variables[prefix + 'id_max'].data).item())
    # ``id_maxi`` is the allocated first-dimension length, whereas ``id_max``
    # is the final active density index.  Some transitions pad the remainder
    # of ``dense`` with zeros, so only the active prefix is a coordinate axis.
    density_allocation = int(np.asarray(_NC.variables[prefix + 'id_maxi'].data).item())
    fainom       = float(np.asarray(_NC.variables[prefix + 'fainom'].data).item())
    stored_densities = _NC.variables[prefix + 'dense'].data
    tab_dens_cm  = np.array(stored_densities[:num_tab_dens + 1])
    pr0          = np.array(_NC.variables[prefix + 'pr0'].data)
    jtot         = np.array(_NC.variables[prefix + 'jtot'].data, dtype=int)
    dom          = np.array(_NC.variables[prefix + 'dom'].data)
    o1lines      = np.array(_NC.variables[prefix + 'o1lines'].data)

    if density_allocation != len(stored_densities):
        raise ValueError('Stehle: inconsistent density metadata in table')
    density_weights = _axis_weights(tab_dens_cm, dens_cm, 'density [cm⁻³]')
    temperature_weights = _axis_weights(tab_temp_k, temp_k, 'temperature [K]')

    normal_hf = 1.25e-9 * (dens_cm ** (2. / 3.))          # normal Holtsmark field [ues]

    PR0_exp = 0.0898 * (dens_cm ** (1. / 6.)) / np.sqrt(temp_k)
    if PR0_exp > 1.:
        raise ValueError('Stehle: plasma too strongly correlated (r₀/λ_D > 1)')

    wl_center_angst = wl_center_m * 1e10
    c_angst         = C * 1e10
    angular_freq_0  = 2 * np.pi * c_angst / wl_center_angst
    otrans          = -2 * np.pi * c_angst / wl_center_angst ** 2
    olines          = o1lines / abs(otrans)

    freq_center = C / wl_center_m
    requested_dom = (
        2.0 * np.pi * np.abs(freq_axis - freq_center) / normal_hf
    )

    # Evaluate the requested grid directly from the required source profiles.
    # Exact table nodes use that node alone; interpolated requests use two or
    # four corners.  The original reader first projected every source onto
    # a heavily thinned common grid; resampling that sparse grid introduced
    # the broad shelves that motivated this implementation.  Direct evaluation
    # retains the source-table information while preserving the reader's
    # temperature and density interpolation order.
    normalized_profile = np.zeros_like(requested_dom)
    for density_index, density_weight in density_weights:
        for temperature_index, temperature_weight in temperature_weights:
            point_count = jtot[density_index, temperature_index]
            if point_count < 3 or pr0[density_index, temperature_index] > 1.0:
                raise ValueError(
                    'Stehle: requested density-temperature cell is not tabulated'
                )
            normalized_profile += density_weight * temperature_weight * (
                _interpolate_grid(
                    dom[density_index, temperature_index, :point_count],
                    olines[density_index, temperature_index, :point_count],
                    requested_dom,
                    fainom,
                    interpolation,
                )
            )
    return normalized_profile * (2.0 * np.pi / normal_hf)


def _zeeman_split_freq(freq_axis, ls, B, view_angle_deg):
    if B == 0.:
        return ls
    theta      = np.deg2rad(view_angle_deg)
    rel_pi     = np.sin(theta)**2 / 2.
    rel_sigma  = (1. + np.cos(theta)**2) / 4.
    freq_shift = E / (4. * np.pi * M_E) * B
    ls_sm = rel_sigma * np.interp(freq_axis + freq_shift, freq_axis, ls, left=0., right=0.)
    ls_sp = rel_sigma * np.interp(freq_axis - freq_shift, freq_axis, ls, left=0., right=0.)
    return rel_pi * ls + ls_sm + ls_sp


def stehle(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
           view_angle_deg=90.0, species='H', interpolation='log-log'):
    """Stehle (MMM) Stark-Zeeman-Doppler profile using local tabulated data.

    Parameters
    ----------
    interpolation : {'log-log', 'fintrp'}, optional
        Detuning interpolation within each source profile.  ``'log-log'`` is
        the default; ``'fintrp'`` reproduces the three-point rule distributed
        with the CDS tables.
    """
    from starkzee.utils import species_to_ZA
    from starkzee.models.analytical import _fwhm_doppler_nm, _SIGMA2FWHM, _nist_center_air_nm

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    if wavelengths_nm.ndim != 1 or wavelengths_nm.size < 2:
        raise ValueError('wavelengths_nm must be a one-dimensional grid with at least two points')
    if np.any(~np.isfinite(wavelengths_nm)) or np.any(wavelengths_nm <= 0.0):
        raise ValueError('wavelengths_nm must contain finite positive values')
    differences = np.diff(wavelengths_nm)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError('wavelengths_nm must be strictly monotonic')
    reversed_grid = differences[0] < 0.0
    work_wavelengths_nm = wavelengths_nm[::-1] if reversed_grid else wavelengths_nm

    scalar_inputs = (B, Ne_m3, Te_ev, Ti_ev, view_angle_deg)
    if any(not np.isscalar(value) or not np.isfinite(value)
           for value in scalar_inputs):
        raise ValueError('Stehle plasma parameters must be finite scalars')
    if B < 0.0 or Ne_m3 <= 0.0 or Te_ev <= 0.0 or Ti_ev < 0.0:
        raise ValueError('Stehle requires B >= 0, Ne_m3 > 0, Te_ev > 0, and Ti_ev >= 0')
    if not 0.0 <= view_angle_deg <= 180.0:
        raise ValueError('view_angle_deg must lie in [0, 180]')
    if isinstance(n_u, (bool, np.bool_)) or isinstance(n_l, (bool, np.bool_)):
        raise ValueError('n_u and n_l must identify a tabulated transition')
    try:
        transition = (int(n_u), int(n_l))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('n_u and n_l must identify a tabulated transition') from error
    if transition != (n_u, n_l) or transition not in _AVAILABLE_TRANSITIONS:
        raise ValueError(
            f'Stehle: transition {n_u}→{n_l} is not in the bundled table; '
            f'supported lower levels are 1, 2, and 3 with n_u <= 30'
        )
    n_u, n_l = transition

    Z, A = species_to_ZA(species)

    lambda0_nm  = _nist_center_air_nm(n_u, n_l, species, work_wavelengths_nm)
    wl_center_m = lambda0_nm * 1e-9
    freq_center = C / wl_center_m

    # freq_axis: just wide enough to cover the wavelength grid. pystark builds it from
    # required_no_fwhm = (max_detuning / fwhm) * 1.05, i.e. a 5 % margin past the grid,
    # with its default npts of 2001 — mirror both exactly so the convolution grid matches.
    max_dfreq = max(abs(C / (work_wavelengths_nm.min() * 1e-9) - freq_center),
                    abs(C / (work_wavelengths_nm.max() * 1e-9) - freq_center))
    half_hz   = max_dfreq * 1.05
    npts      = 2001
    freq_axis = np.linspace(freq_center - half_hz, freq_center + half_hz, npts)

    # Pure Stark profile in frequency space
    ls_s = _compute_stehle_stark(
        n_u, n_l, Ne_m3, Te_ev, wl_center_m, freq_axis,
        interpolation=interpolation,
    )

    # Doppler kernel on freq_axis_conv (500 extra points each side, like pystark).
    # fftconvolve(ls_s, ls_d, 'same') returns len(ls_s) points; the extra width in
    # ls_d prevents the convolution from picking up zero-padding artifacts at the edges.
    extra     = 1000
    dfreq     = (freq_axis[-1] - freq_axis[0]) / (len(freq_axis) - 1)
    freq_conv = np.linspace(freq_axis[0]  - extra // 2 * dfreq,
                             freq_axis[-1] + extra // 2 * dfreq,
                             len(freq_axis) + extra)
    if Ti_ev == 0.0:
        ls_sd = ls_s
    else:
        fwhm_d_hz = (_fwhm_doppler_nm(lambda0_nm, Ti_ev, A) * 1e-9) * C / wl_center_m**2
        sigma_hz  = fwhm_d_hz / _SIGMA2FWHM
        ls_d      = np.exp(-0.5 * ((freq_conv - freq_center) / sigma_hz)**2)
        ls_d     /= ls_d.sum()
        ls_sd = fftconvolve(ls_s, ls_d, 'same')  # len(ls_s) = npts

    # Zeeman splitting in frequency space
    ls_szd = _zeeman_split_freq(freq_axis, ls_sd, B, view_angle_deg)

    # Convert frequency → wavelength.
    # I(ν) [1/Hz] → I(λ) [1/m]: I(λ) = I(ν) · |dν/dλ| = I(ν) · c/λ²
    wl_from_freq_nm = C / freq_axis * 1e9            # nm, non-uniform & reversed
    wl_from_freq_m  = C / freq_axis                  # m
    ls_wl = ls_szd * C / wl_from_freq_m**2           # [1/m]

    # Sort by ascending wavelength for np.interp
    order     = np.argsort(wl_from_freq_nm)
    wl_sorted = wl_from_freq_nm[order]
    ls_sorted = ls_wl[order]

    ls_out = np.interp(work_wavelengths_nm, wl_sorted, ls_sorted, left=0., right=0.)

    # Area-normalize
    area = trapz(ls_out, work_wavelengths_nm * 1e-9)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError('Stehle profile has no positive finite area on the requested grid')
    ls_out /= area
    return ls_out[::-1] if reversed_grid else ls_out
