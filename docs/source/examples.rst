Examples
========

All example scripts are in the ``examples/`` directory of the repository.
Run them from the project root:

.. code-block:: bash

    python examples/example_balmer_lineprofile.py

Balmer series — LineProfile class
-----------------------------------

**Script:** ``examples/example_balmer_lineprofile.py``

Computes Hα through Hε at DIII-D edge conditions (B = 12 T,
:math:`N_e = 10^{20}` m\ :sup:`-3`, :math:`T_e = 5` eV) using the
:class:`~starkzee.line_profile.LineProfile` class.  Each panel shows:

- Transverse (90°) and parallel (0°) broadened profiles
- Discrete stick spectrum at zero microfield (:math:`F = 0`)

The wavelength window is set by converting ±1 nm around each line centre
to an energy grid.

Hα stick spectrum and broadened profile
-----------------------------------------

**Script:** ``examples/example_halpha.py``

Side-by-side comparison of the discrete transition stick spectrum (``compute_discrete``)
and the static Stark-Zeeman broadened profile at B = 5 T, 100 T, and 1000 T,
showing the transition from the Paschen-Back regime to the strong-field limit.

Transition anatomy
------------------

**Script:** ``examples/example_transitions.py``

Explores the discrete transitions of Ly-α (n=2→1) and Hβ (n=4→2) as a
function of B, grouping transitions by polarization and plotting their
positions and strengths as a function of microfield :math:`F_z`.

Reproducing Figure 1 from Ferri *et al.* (2022)
-------------------------------------------------

**Script:** ``examples/reproduce_fig1.py``

Reproduces the full Balmer series profiles (Hα–Hε) weighted by Case B
Balmer decrements, at B = 1000 T, :math:`N_e = 10^{23}` m\ :sup:`-3`,
:math:`T_e = 5` eV.  Demonstrates convergence with respect to ``num_f``
and ``num_mu``.

Satellite features at B = 1000 T
----------------------------------

**Script:** ``examples/diag_halpha_satellites.py``

Searches for the :math:`\pm 2\mu_B B` Stark-Zeeman satellite features
in Hα at B = 1000 T.  Shows that:

- **Hβ** (n=4→2): the satellite is a resolved peak at ~2 % of the main σ+
  intensity because :math:`n=4` has degenerate :math:`|4d, m_l{=}2\rangle`
  and :math:`|4f, m_l{=}2\rangle` states that amplify Stark mixing.
- **Hα** (n=3→2): no resolved peak appears — the satellite amplitude
  (~0.07 %) is buried in the Lorentzian tail.

**Script:** ``examples/diag_satellite_comparison.py``

Direct Hα vs. Hβ comparison over a ±200 meV window, showing both the
:math:`\pm\mu_B B` main peaks and the :math:`\pm 2\mu_B B` satellite
positions.

Quadratic Zeeman polarization wings
-------------------------------------

**Script:** ``examples/reproduce_halpha_wings.py``

Demonstrates the polarization-wing effect at B = 500 T and B = 1000 T:
at high field the quadratic Zeeman term shifts the 3p(:math:`m_l = \pm 1`)
→ 2s transitions away from the main Hα cluster, creating distinct "wings"
that only appear in ``include_quadratic=True`` profiles.

Lyman-α at high B
------------------

**Scripts:** ``examples/test_lyman_alpha.py``,
``examples/test_lyman_alpha_500T.py``,
``examples/test_lyman_alpha_1000T.py``

Lyman-α (n=2→1) for C VI (Z=6) at 100 T, 500 T, and 1000 T, exploring the
transition from the intermediate-field to the Paschen-Back regime.
