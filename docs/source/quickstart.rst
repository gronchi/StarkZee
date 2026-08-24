Quick start
===========

The recommended entry point is the :class:`~starkzee.line_profile.LineProfile`
class.  It wraps all solver calls, stores results as attributes, and provides
convenience properties for common observation geometries.

Choosing static, FFM, or discrete output
-----------------------------------------

The Python API does not select a model automatically:
:meth:`~starkzee.line_profile.LineProfile.compute_profile` and
:meth:`~starkzee.line_profile.LineProfile.compute_static_profile` always use
the quasi-static-ion solver, whereas
:meth:`~starkzee.line_profile.LineProfile.compute_ffm_profile` explicitly uses
the Frequency Fluctuation Model (FFM).  Use
:meth:`~starkzee.line_profile.LineProfile.compute_discrete` when a stick
spectrum at one specified electric-field configuration is required instead of
a plasma-averaged profile.

.. mermaid::

   flowchart TD
       A["What output is needed?"] --> B{"One specified electric-field<br/>configuration?"}
       B -- Yes --> C["compute_discrete(Fz, Fx)"]
       C --> D["Diagonalize the upper and lower<br/>Stark-Zeeman Hamiltonians once"]
       D --> E["Return transition energies,<br/>polarizations, and strengths"]

       B -- No --> F["Continuous plasma line profile"]
       F --> G{"Are the StarkZee assumptions valid?<br/>Hydrogen-like radiator and<br/>within-shell Stark mixing"}
       G -- No --> H["Use a model with multi-electron<br/>or inter-shell coupling"]
       G -- Yes --> I{"Is the ion microfield effectively frozen?<br/>Ion fluctuation energy νᵢ much smaller<br/>than the characteristic Stark/SDT spread"}

       I -- Yes --> J["Static solver<br/>compute_static_profile()"]
       I -- No --> K["Dynamic-ion solver<br/>compute_ffm_profile()"]
       I -- Unsure --> L["Run both with matched settings<br/>and compare the line core"]
       L --> J
       L --> K

       J --> M["π, σ+, and σ− profiles"]
       K --> M
       M --> N["Combine for the observation angle"]
       N --> O["Optional instrumental convolution"]
       O --> P["Observable spectrum"]

Here ``ν_i`` denotes the ion jumping rate in energy units (the code multiplies
the rate in s\ :sup:`-1` by :math:`\hbar`).  FFM becomes more relevant as the
ions move faster (larger ``Ti_ev`` and smaller ion mass), and for line cores
whose Stark-dressed structure is not well separated, often at lower magnetic
field or higher principal quantum number.  Density changes both the
microfield distribution and its fluctuation rate, so density alone is not a
universal decision boundary.

When the two regimes are not clearly separated, compare both solvers using
the same ``num_f``, ``num_mu``, ``max_beta``, microfield distribution,
electron model, and Doppler setting.  For the closest implementation-level
comparison, set ``frequency_dependent_width=False`` on the static solver,
because the FFM currently uses one resonance electron-impact width for all
Stark-dressed transitions.  In the theoretical limit ``ν_i -> 0``, the FFM
reduces to the static profile.

Basic profile
-------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from starkzee.line_profile import LineProfile

    # Hα (n=3→2), DIII-D edge conditions
    lp = LineProfile(n_u=3, n_l=2, B=5.0, Ne_m3=1e20, Te_ev=5.0)

    # Build an energy grid ±3 nm around the line center
    lam0 = lp.E0_wavelength_nm        # ≈ 656.1 nm
    HC   = 1239.84193                  # hc/e  [eV·nm]
    energies = np.linspace(HC/(lam0+3), HC/(lam0-3), 1000)

    lp.compute_profile(energies, num_f=30, num_mu=8,
                       use_screening=True, quadratic_zeeman=False)

    # Observation at 90° to B (transverse): π + ½(σ+ + σ-)
    plt.plot(lp.wavelengths_nm - lam0, lp.profile_transverse, label="90°")

    # Along B (0°): σ+ + σ-
    plt.plot(lp.wavelengths_nm - lam0, lp.profile_parallel,
             ls="--", label="0°")

    plt.xlabel("Δλ  (nm)")
    plt.legend()
    plt.show()

The profile is stored as three polarization components
(``profile_pi``, ``profile_sig_plus``, ``profile_sig_minus``).
Use :meth:`~starkzee.line_profile.LineProfile.profile_at_angle` for
arbitrary observation angle θ.

Discrete transition stick spectrum
-----------------------------------

.. code-block:: python

    lp.compute_discrete(Fz=0.0, Fx=0.0)

    disc = lp.discrete
    print(disc)          # DiscreteTransitions(89 transitions, ...)

    # Sorted by energy; q = 0 (π), -1 (σ+, blue-shifted), +1 (σ-, red-shifted)
    for E, q, S in zip(disc.energy_ev, disc.q, disc.strength):
        print(f"  E={E:.4f} eV  q={q:+d}  |d|²={S:.4e} a₀²")

Both ``compute_profile`` and ``compute_discrete`` return ``self``, so they can
be chained:

.. code-block:: python

    lp.compute_profile(energies).compute_discrete()

Balmer series at once
----------------------

.. code-block:: python

    from starkzee.line_profile import LineProfile
    import numpy as np

    lines = [(3,2,"Hα"), (4,2,"Hβ"), (5,2,"Hγ"), (6,2,"Hδ")]
    B, Ne, Te = 5.0, 1e20, 5.0

    profiles = {}
    for n_u, n_l, label in lines:
        lp = LineProfile(n_u=n_u, n_l=n_l, B=B, Ne_m3=Ne, Te_ev=Te)
        HC = 1239.84193
        lam0 = lp.E0_wavelength_nm
        energies = np.linspace(HC/(lam0+2), HC/(lam0-2), 800)
        lp.compute_profile(energies, num_f=30, num_mu=8)
        profiles[label] = lp

Lower-level API
---------------

The class delegates to :func:`~starkzee.static_profile.calculate_static_profile`
and :func:`~starkzee.static_profile.discrete_transitions`.  These can be
called directly for more control:

.. code-block:: python

    from starkzee.static_profile import calculate_static_profile
    import numpy as np
    from starkzee.utils import RYDBERG_EV

    E0 = RYDBERG_EV * (1/4 - 1/9)   # Hα
    energies = E0 + np.linspace(-0.005, 0.005, 1000)

    pi, sig_plus, sig_minus = calculate_static_profile(
        n_u=3, n_l=2, Z=1, B=5.0,
        Ne_m3=1e20, Te_ev=5.0,
        energies_ev=energies,
        num_f=30, num_mu=8,
    )

FFM (dynamic ion) profile
-------------------------

Use the FFM when the ion microfield changes appreciably on the line-formation
timescale.  From ``LineProfile``, set ``Ti_ev`` on the constructor and call
:meth:`~starkzee.line_profile.LineProfile.compute_ffm_profile` instead of
``compute_profile``.  It populates the same result attributes
(``profile_pi``, ``profile``, ``profile_transverse``, ...) and applies Doppler
broadening by default:

.. code-block:: python

    lp_ffm = LineProfile(n_u=3, n_l=2, B=5.0,
                         Ne_m3=1e23, Te_ev=5.0, Ti_ev=0.1)
    lp_ffm.compute_ffm_profile(energies, num_f=30, num_mu=8,
                               sdt_bin_tol=1e-5)
    plt.plot(lp_ffm.wavelengths_nm, lp_ffm.profile_transverse)

``sdt_bin_tol`` (eV) merges Stark-dressed transitions closer together than
this tolerance before the Markov solve — often a 10-100x speedup with
negligible accuracy loss; omit it (default ``None``) for the exact,
unbinned calculation.

The lower-level :func:`~starkzee.ffm.calculate_ffm_profile` can also be
called directly for more control:

.. code-block:: python

    from starkzee.ffm import calculate_ffm_profile

    pi, sp, sm = calculate_ffm_profile(
        n_u=3, n_l=2, Z=1, B=5.0,
        Ne_m3=1e23, Te_ev=5.0, Ti_ev=0.1,
        A_ion=1, energies_ev=energies,
        num_f=30, num_mu=8, sdt_bin_tol=1e-5,
    )

Doppler and instrumental broadening
------------------------------------

Doppler broadening can be applied inside either solver:

- The static path includes it when ``Ti_ev`` is set on ``LineProfile`` (or
  passed directly to ``calculate_static_profile``).
- The FFM path requires ``Ti_ev`` for the ion fluctuation rate and applies
  Doppler broadening by default; pass ``apply_doppler=False`` to disable it.

Instrumental broadening is never automatic.  The standalone
:func:`~starkzee.convolutions.apply_doppler_broadening` and
:func:`~starkzee.convolutions.apply_instrument_broadening` helpers are useful
when post-processing explicitly.  Do not apply the Doppler helper to a profile
that already included Doppler inside its solver.  Both helpers require a
**uniform wavelength grid**:

.. code-block:: python

    from starkzee.convolutions import (
        apply_doppler_broadening,
        apply_instrument_broadening,
    )

    # Compute directly on a uniform *wavelength* grid so lam below is uniform
    # (an energy grid mapped through E=hc/lambda is not uniform in wavelength).
    lam = np.linspace(lp.E0_wavelength_nm - 3, lp.E0_wavelength_nm + 3, 1000)
    # This lp has no Ti_ev, so the static result below is Doppler-free.
    lp.compute_profile(lam, grid_type='wavelength_nm', num_f=30, num_mu=8)
    total = lp.profile_transverse

    # Doppler: Ti=0.5 eV, hydrogen emitter (species='H')
    broadened = apply_doppler_broadening(lam, total, Ti_ev=0.5, species='H')

    # Instrument: FWHM = 0.05 nm
    broadened = apply_instrument_broadening(lam, broadened, fwhm_nm=0.05)

The complete pipeline is :math:`I_\text{obs} = (I_\text{SZ} * G_D) * G_\text{inst}`.
