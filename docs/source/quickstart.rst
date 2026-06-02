Quick start
===========

The recommended entry point is the :class:`~starkzee.line_profile.LineProfile`
class.  It wraps all solver calls, stores results as attributes, and provides
convenience properties for common observation geometries.

Basic profile
-------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from starkzee.line_profile import LineProfile

    # Hα (n=3→2), DIII-D edge conditions
    lp = LineProfile(n_u=3, n_l=2, Z=1, B=5.0, Ne_m3=1e20, Te_ev=5.0)

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

    # Sorted by energy; q = 0 (π), +1 (σ+), -1 (σ-)
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
        lp = LineProfile(n_u=n_u, n_l=n_l, Z=1, B=B, Ne_m3=Ne, Te_ev=Te)
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
--------------------------

Use :func:`~starkzee.ffm.calculate_ffm_profile` when ion dynamics matter
(high density, low B, or high-n transitions):

.. code-block:: python

    from starkzee.ffm import calculate_ffm_profile

    pi, sp, sm = calculate_ffm_profile(
        n_u=3, n_l=2, Z=1, B=5.0,
        Ne_m3=1e23, Te_ev=5.0, Ti_ev=0.1,
        A_ion=1, energies_ev=energies,
        num_f=30, num_mu=8,
    )

Post-processing: Doppler and instrumental broadening
----------------------------------------------------

Doppler and instrumental broadening are **not** applied automatically.
Apply them as a post-processing step using :func:`~starkzee.convolutions.apply_doppler_broadening`
and :func:`~starkzee.convolutions.apply_instrument_broadening`.
Both require a **uniform wavelength grid**:

.. code-block:: python

    from starkzee.convolutions import (
        apply_doppler_broadening,
        apply_instrument_broadening,
    )

    lp.compute_profile(energies, num_f=30, num_mu=8)
    total = lp.profile_transverse

    # Convert energy grid → uniform wavelength grid
    lam = lp.wavelengths_nm

    # Doppler: Ti=0.5 eV, hydrogen emitter (A=1)
    broadened = apply_doppler_broadening(lam, total, Ti_ev=0.5,
                                         A_emitter=1, E0_ev=lp.E0)

    # Instrument: FWHM = 0.05 nm
    broadened = apply_instrument_broadening(lam, broadened, fwhm_nm=0.05)

The complete pipeline is :math:`I_\text{obs} = (I_\text{SZ} * G_D) * G_\text{inst}`.
