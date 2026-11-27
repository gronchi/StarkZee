.. _sec:multielectron:

Multi-electron emitters
=======================

The hydrogenic code builds each line from the shells :math:`n_u \to n_l` of
one electron in a Coulomb field. A multi-electron line has non-degenerate
levels with their own energies, Landé factors and dipole couplings. StarkZee
describes such a line with a *radiator* object. The static and FFM solvers then
take the radiator in place of :math:`(n_u, n_l, Z)`. The microfield, the
quadrature over field strength and direction, the electron-impact operators,
Doppler broadening and the polarization channels are the same as for hydrogen.

This chapter covers:

* the radiator interface and the ways to build a radiator;
* the Hamiltonian in the :math:`|\mathrm{level}, J, M\rangle` basis;
* the sources of radial elements;
* the Zeeman treatments;
* electron broadening and ion dynamics;
* the atomic-data readers and the limits of the model.

When to use it
--------------

Use a multi-electron radiator when the hydrogenic shell picture does not hold.
Examples:

* He I lines, whose terms are split by exchange and whose "forbidden"
  components appear with density (He I 447.1 nm);
* lines of one valence electron outside closed shells (Li-like C IV, N V,
  O VI; Na-like ions; alkali atoms), where the quantum defects lift the
  :math:`l` degeneracy of a shell;
* any line with level data from an atomic-structure code (FAC, Cowan's RCG).

Hydrogen and hydrogen-like ions keep using the built-in hydrogenic path. That
path is itself a radiator (:class:`~starkzee.radiators.HydrogenicTransition`),
and passing it explicitly gives the same profile as the default.

Using a radiator
----------------

Build the radiator once per magnetic field and pass it to a solver:

.. code-block:: python

    import numpy as np
    from starkzee.he_i import he_i_4471
    from starkzee.static_profile import calculate_static_profile
    from starkzee.ffm import calculate_ffm_profile

    B = 2.0
    radiator = he_i_4471(B)                  # He I 447.1 nm, n = 4 triplet manifold
    E0 = radiator.line_energy_ev
    grid = np.linspace(E0 - 3e-3, E0 + 4e-3, 1401)

    # quasi-static ions
    pi, sp, sm = calculate_static_profile(4, 2, 1, B, 1e21, 5.0, grid,
                                          radiator=radiator, Ti_ev=2.0)

    # ion dynamics (FFM); n_u, n_l and Z may be None
    pi, sp, sm = calculate_ffm_profile(None, None, None, B, 1e21, 5.0, 2.0, None, grid,
                                       radiator=radiator, electron_operator=True)

With a radiator:

* the solvers ignore ``n_u``, ``n_l``, ``Z`` and the hydrogenic structure
  options (``quadratic_zeeman``, ``fine_structure``, ``use_empirical_data``,
  ``atom``). Those options belong to the radiator;
* ``emitter_charge`` defaults to the radiator charge;
* the Doppler width uses the radiator mass and its line reference energy.

In the FFM, ``A_ion`` defaults to the radiator mass. The perturbing ions are
set by ``Z_bar`` and ``A_perturber``, which defaults to protons.

The B field is fixed when the radiator is built, so build a new radiator for
each field. The solver's ``B`` argument still sets the field-dependent parts of
the electron-impact cutoff, so pass the same value to both.

Building a radiator
-------------------

There are three ways to build one, from the most ready-made to the most general.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Entry point
     - Use
   * - :func:`~starkzee.he_i.he_i_4471`
     - He I 447.1 nm (2p :sup:`3`\ P - 4d :sup:`3`\ D) from NIST data, with
       the n = 4 triplet terms 4s, 4p, 4d and 4f in the upper manifold.
       ``singlets=True`` adds the n = 4 singlet terms: the NIST 4f 1F3 level is
       0.024 cm^-1 from 4f 3F3, so spin-orbit makes the two J = 3 levels nearly
       equal singlet-triplet mixtures (4d J = 2: 0.5 %), at any field. With the
       equal sublevel populations of the profiles this leaves the line unchanged
       to 0.3 %, so the triplet manifold is the default.
   * - :func:`~starkzee.single_valence.c_iv_5_4`,
       :func:`~starkzee.single_valence.n_v_5_4`,
       :func:`~starkzee.single_valence.o_vi_5_4`,
       :func:`~starkzee.single_valence.single_valence_transition`
     - Li-like n = 5 → 4 lines from NIST levels, or any
       :math:`n_u \to n_l` line of a one-valence-electron ion from a level
       table (``data/<ion>_nist.json``).
   * - :class:`~starkzee.ls_radiator.LSTransition`
     - Any line between two sets of LS terms. You give the J-level energies
       of each term and the signed reduced E1 elements between terms.
   * - :class:`~starkzee.jlevel_radiator.JLevelTransition`
     - Any line between two sets of J levels of a
       :class:`~starkzee.level_data.LevelData`, e.g. read from FAC or RCG
       output. No LS labels are needed.

The ready-made radiators also set ``line_energy_ev``, the centroid of the
strongest component, which is convenient for placing the energy grid.

An LS radiator from terms
~~~~~~~~~~~~~~~~~~~~~~~~~

An :class:`~starkzee.ls_radiator.LSTerm` holds L, S, parity and the energy of
each J level [cm\ :sup:`-1`], optionally with its E1 decay rates. The
couplings are signed reduced elements :math:`\langle a\|d\|b\rangle`
[:math:`e a_0`], given once for each unordered pair of terms:

* pairs inside one manifold give the Stark mixing;
* pairs between the manifolds give the radiative dipoles;
* pairs with a perturbing term enter only the collision self terms.

For one active electron,
:func:`~starkzee.ls_radiator.reduced_element` computes them from the term
energies, using the Coulomb approximation or hydrogenic radial integrals:

.. code-block:: python

    from starkzee.ls_radiator import ActiveElectron, LSTerm, LSTransition, reduced_element

    T2s = LSTerm("2s 2S", 0, 0.5, +1, {0.5: E_2s})
    T2p = LSTerm("2p 2P", 1, 0.5, -1, {0.5: E_2p12, 1.5: E_2p32}, decay_rates=A_2p)
    T1s = LSTerm("1s 2S", 0, 0.5, +1, {0.5: E_1s})
    e = {"1s 2S": ActiveElectron(1, 0), "2s 2S": ActiveElectron(2, 0), "2p 2P": ActiveElectron(2, 1)}
    t = {"1s 2S": T1s, "2s 2S": T2s, "2p 2P": T2p}
    couplings = {(a, b): reduced_element(t[a], e[a], t[b], e[b], Z_core=6, radial="hydrogenic")
                 for a, b in (("2s 2S", "2p 2P"), ("2p 2P", "1s 2S"))}

    radiator = LSTransition([T2s, T2p], [T1s], couplings, B, mass_amu=12.0,
                            emitter_charge=5, collision_n=2, collision_Z=6)

``collision_n`` turns on the generalized electron-impact operator (see below).
Without it the line has only the constant widths ``electron_widths_ev``.

A J-level radiator from atomic-structure data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The readers return a :class:`~starkzee.level_data.LevelData`. You then choose
the level indices of the two manifolds:

.. code-block:: python

    from starkzee.fac_reader import read_fac
    from starkzee.jlevel_radiator import JLevelTransition

    data = read_fac("ion.lev", "ion.tr", g_factors="jj")
    radiator = JLevelTransition(data, upper=[10, 11, 12], lower=[3, 4], B=5.0,
                                emitter_charge=2, collision_n=4.0)

For RCG output, use :func:`~starkzee.cowan_reader.read_rcg`, and
:func:`~starkzee.cowan_reader.select_levels` to pick the levels of named
configurations.

Hamiltonian
-----------

Each manifold (upper or lower) has the basis :math:`|a, J, M\rangle`, where
:math:`a` is the level. The field-free part is diagonal and holds the
level energies :math:`E_a`. The magnetic field adds

.. math::

   H_Z = \mu_B B\,\langle a J M | L_z + g_s S_z | b J' M \rangle ,

which has two parts:

* the diagonal elements :math:`g_J M`;
* for an LS radiator, the elements between the J levels of one term (and,
  with level data, the dataset's magnetic elements between levels).

Diagonalizing :math:`H_0 + H_Z` exactly gives the intermediate-field Zeeman
pattern and its transition to the Paschen-Back regime. The orbital factor
defaults to the finite-mass value :math:`g_L = 1 - m_e/M` when
``mass_amu`` is given, and :math:`g_s` is the CODATA value.

With ``diamagnetic`` set, :math:`H_0` also gets the quadratic Zeeman term,

.. math::

   \frac{e^2B^2}{8m_e}\,r^2\sin^2\theta
   = \frac{e^2B^2}{12m_e}\,r^2\bigl(C^0 - C^2_0\bigr),

built from the reduced elements of :math:`r^2C^0` and :math:`r^2C^2` between
the levels of one manifold. The one-valence-electron radiators include it by
default, with :math:`\langle r^2\rangle` from the same radial model as the
dipoles. They keep the field-free radial functions: the field-induced mixing
of other shells scales as :math:`\approx 0.2\,\gamma^2 n^7/Z^4`
(:math:`\gamma = B/2.35\times10^5` T), which is :math:`3\times10^{-4}` for
C IV n = 5 at 500 T.

The plasma microfield enters as :math:`V = e\vec F\cdot\vec r = F_z M_z + F_x M_x`.
The Stark templates :math:`M_z, M_x` come from the E1 elements between levels
of the same manifold, and the radiative dipoles :math:`D_q` from the elements
between the manifolds.

All matrix elements follow the Edmonds / Condon-Shortley Wigner-Eckart
convention,

.. math::

   \langle J M | T^1_q | J' M' \rangle
   = (-1)^{J-M}\begin{pmatrix} J & 1 & J' \\ -M & q & M' \end{pmatrix}
     \langle J \| T \| J' \rangle ,

and an LS reduced element is reduced to the one-electron radial integral with
the usual 6j recoupling (see :mod:`starkzee.ls_radiator`).

Signs
~~~~~

Stark mixing and the interference between radiative paths depend on the
*relative* signs of the reduced elements. These signs are meaningful only
within one consistent set of phases, such as one structure calculation or one
radial model. The handling of unsigned elements depends on where they are used:

* **Stark pairs:** :class:`~starkzee.jlevel_radiator.JLevelTransition` raises
  an error if an element between levels of the same manifold has no sign.
* **Radiative pairs:** it warns if an element between the manifolds has no
  sign, because the interference of mixed states is then not represented.

FAC tables with ``MULTIP = -1`` and RCG spectrum blocks carry signs.
``MULTIP = 0`` FAC tables do not.

Radial elements
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``radial=``
     - Model
   * - ``"coulomb"``
     - Coulomb approximation (Bates and Damgaard, *Phil. Trans. R. Soc. A*
       **242**, 101 (1949)). The effective quantum number :math:`n^*` comes
       from the measured binding energy. The result is exact (Gordon's formula)
       for integer :math:`n^*`, and typically within a few percent for
       low-:math:`l` valence lines. :func:`~starkzee.coulomb_approximation.coulomb_r2`
       gives the :math:`r^2` moments.
   * - ``"hydrogenic"``
     - Hydrogenic integrals with the core charge (Gordon's formula).
   * - data
     - Reduced elements read from FAC or RCG, used as given.

For He I 447.1 nm, ``scale_to_nist=True`` (the default) sets the magnitude of
the 2p-4d element from the NIST multiplet line strength and keeps the sign of
the approximation. The Coulomb approximation is within 1.3 % of NIST there,
and within 1.3-2.6 % of the NIST grade-A line strengths for C IV n = 5 → 4.

Zeeman treatments
-----------------

``zeeman=`` selects how the field is treated. It is available on
:class:`~starkzee.radiators.HydrogenicTransition`,
:class:`~starkzee.ls_radiator.LSTransition` and
:class:`~starkzee.jlevel_radiator.JLevelTransition`.

.. list-table::
   :header-rows: 1
   :widths: 14 43 43

   * - ``zeeman``
     - Treatment
     - Valid when
   * - ``"exact"`` (default)
     - Zeeman operator and fine structure diagonalized together
       (intermediate field).
     - Always.
   * - ``"weak"``
     - The field acts only inside each J level (:math:`g_J M`, anomalous
       Zeeman effect). The diamagnetic term is restricted in the same way.
     - :math:`\mu_B B` much smaller than the spacing of the levels of equal
       :math:`M`.
   * - ``"strong"``
     - Fine structure removed: the J levels of each term are put at the
       :math:`(2J+1)`-weighted term mean, then the field is treated exactly
       (Paschen-Back limit, :math:`\mu_B B (g_L m_L + g_s m_S)`).
     - :math:`\mu_B B` much larger than the fine-structure splitting.
       LS and hydrogenic radiators only: a J-level dataset has no term
       structure, so ``JLevelTransition`` raises an error.

For the hydrogenic radiator, ``"weak"`` keeps the field terms inside each
:math:`(l, j)` level, and ``"strong"`` replaces the :math:`j` splitting of each
:math:`l` by its mean. The two constructions agree: C VI :math:`n = 2` built as
a hydrogenic radiator and as LS terms gives the same levels under all three
treatments.

The approximations are mainly for comparison with codes and papers that use
them. Profiles for measurements should use ``"exact"``.

Electron broadening
-------------------

With ``collision_n`` (the effective principal quantum number of the upper
manifold), a radiator uses the impact-limit operator of
:mod:`starkzee.level_collision`. It generalizes the PPP operator
(PPP manual, Appendix B) from one degenerate shell to non-degenerate levels.
Each :math:`\vec d\cdot\vec d` product carries the impact factor of its own
pair of field-free levels, with the cutoff

.. math::

   \omega_c(a, b) = \max(\omega_p, \omega_L, \omega_e, |E_a - E_b|)

(Ferri, Peyrusse and Calisti 2022). The level gap therefore limits the
broadening between distant levels. With all gaps below the plasma cutoff, the
operator reduces exactly to the hydrogenic one.

The usual solver options all work with this operator:

* the default scalar widths;
* ``electron_operator=True``: per-state widths from the operator;
* ``electron_interference=True``: the full PPP operator with interference
  (static solver only).

Levels outside the manifolds (``upper_perturbers``, ``lower_perturbers``), such
as 4p and 4f for a 4d manifold, or levels of other shells, add to the collision
self terms only.

As in the hydrogenic code, :math:`n` and the charge in the GBK argument are
those of the upper manifold for every term. ``collision_n_lower`` sets a
separate :math:`n` for the lower manifold's scalar and per-state widths.

The electron option changes multi-electron widths strongly. For C IV
n = 5 → 4 at :math:`n_e = 2\times10^{24}` m\ :sup:`-3`, :math:`T_e = 10` eV
and B = 0, the static FWHM is:

* 19.5 Å with scalar widths;
* 10.8 Å with the per-state operator;
* 5.4 Å with the full interference operator.

Choose the option deliberately and state it with the result.

Ion dynamics
------------

:func:`~starkzee.ffm.calculate_ffm_profile` accepts the same radiators. Each
Stark-dressed transition carries its electron width (scalar or per-state, with
``electron_operator=True``) and its natural width.

At B = 0 the transitions of all three polarizations are pooled before the
fluctuation step (each polarization gets one third), because without a field
the channels are not distinct.

Two FFM options are not available with a radiator, and raise
``NotImplementedError``:

* ``electron_interference``;
* ``sdt_frequency_dependent_width``.

The validation tests check the radiator path against the built-in FFM:

* a hydrogenic radiator reproduces the built-in hydrogenic FFM to
  :math:`10^{-10}`;
* with :math:`\nu_i = 0`, the FFM equals the static profile computed with
  ``frequency_dependent_width=False``.

Atomic-data readers
-------------------

The readers fill a :class:`~starkzee.level_data.LevelData`. It stores levels
(energy, J, parity, labels, :math:`g_J`) and E1 reduced elements
:math:`\langle \mathrm{lower}\|d\|\mathrm{upper}\rangle`. Each element records
whether its sign came from the source.

**FAC** (:func:`~starkzee.fac_reader.read_fac`). Use verbose ASCII tables
(``PrintTable(..., 1)``).

* ``MULTIP = -1`` blocks give signed E1 elements,
  :math:`\langle \mathrm{lower}\|d\|\mathrm{upper}\rangle = M/\sqrt2`, checked
  against gf.
* ``MULTIP = 0`` blocks give unsigned magnitudes from gf (all multipoles
  summed).
* FAC level tables have no Landé factors, so supply them as a dict, or use
  ``g_factors="jj"`` to estimate them in pure jj coupling from the level names.

**Cowan RCG** (:func:`~starkzee.cowan_reader.read_rcg`). It reads RCG's main
print output or ADAS801's unit-52 spectrum file.

* The sign of each E1 element is the sign of RCG's cancellation factor.
* Magnetic-dipole blocks give the elements between levels that the exact
  Zeeman treatment needs.
* ``g_factors="rcg"`` reads RCG's printed g-values; ``"ls"`` uses pure-LS
  values.

The reduced-element convention, :math:`S = |\langle J\|d\|J'\rangle|^2` and
:math:`gf = \tfrac23\,\Delta E[\mathrm{Ha}]\,S`, is the one used throughout.

Limits
------

* **Mixing between terms.** Mixing between terms of the same J and parity
  (intermediate coupling, e.g. He I :math:`n` d :sup:`1`\ D\ :sub:`2` /
  :sup:`3`\ D\ :sub:`2`) is present only if the level data contain it. The
  LS radiator assumes pure LS terms, so intercombination lines need level data.
* **Diamagnetic term.** It acts only inside each manifold. Mixing of other
  shells by the field is neglected (see the estimate above).
* **Collision parameters.** The impact operator uses one effective :math:`n`
  and one charge for all the terms of a line.
* **FFM options.** Two FFM options are not available with a radiator (see
  above).
* **FFM electron widths.** In the FFM, the scalar electron widths of a
  radiator are evaluated at the line reference energy. The static solver
  evaluates them across the frequency grid, measured from that reference
  (``frequency_dependent_width``).
