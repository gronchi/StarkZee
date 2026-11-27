stark_map
=========

DC electric-field maps in a multi-shell hydrogenic basis
--------------------------------------------------------

See :doc:`../atomic_api` for a runnable example and plots. This diagnostic
includes inter-shell coupling in the selected basis and uses analytical
energies. It does not use the empirical H/D/T tables and does not change the
single-shell profile solvers.

``shells`` selects principal quantum numbers; ``max_l=None`` retains every
allowed orbital angular momentum in those shells. ``B`` is in tesla and
defines the z axis. ``quadratic_zeeman`` controls the diamagnetic term.
The legacy ``fine_structure=False`` option omits mass-velocity and Darwin
corrections; spin-orbit coupling remains present.

``compute`` accepts a nonempty one-dimensional array of finite, nonnegative
field magnitudes in V/m. ``angle_deg`` is the E--B angle. A fixed ``mj``
sector is available only for parallel or antiparallel fields; when a target
is provided in that geometry, its projection is selected automatically.
For crossed fields, leave ``mj=None``. A target has the form
``(n, l, j, mj)`` and must belong to the retained basis.

Result conventions
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Meaning
   * - ``fields_v_m``
     - Input field magnitudes, shape ``(N_fields,)``.
   * - ``energies_ev``
     - Sorted absolute energies relative to ionization, shape
       ``(N_fields, N_states)``.
   * - ``zero_field_energies_ev``
     - Sorted energies at F=0 and the configured B, shape ``(N_states,)``;
       evaluated even if zero is absent from the sweep.
   * - ``highlights``
     - Squared overlaps with the coupled target state, same shape as
       ``energies_ev``. Zero without a target.
   * - ``reference_energy_ev``
     - Energy at F=0 of the eigenstate with the largest target overlap, or
       the lowest F=0 energy if there is no target.
   * - ``basis``, ``shells``
     - Retained uncoupled basis states and principal shells.
   * - ``B``, ``Z``, ``A``, ``angle_deg``, ``target_state``, ``mj``
     - Calculation settings, with the selected symmetry sector recorded.

``shifted_energies`` accepts ``eV``, ``meV``, ``GHz`` (ordinary frequency),
and ``cm^-1``. Reference choices are ``target_zero_field``,
``lowest_zero_field``, ``ionization``, or a numeric eV energy. The default
``target_zero_field`` requires a target; zero field here means zero
electric field at fixed B. ``plot`` returns ``(fig, ax)`` and accepts
``V/m`` or ``V/cm`` for the horizontal axis.

The dominant-target properties choose the largest overlap at each field
independently and do not track a continuous eigenstate branch. The overlap
weights are not oscillator strengths or populations. Convergence with
respect to the retained shells and angular momenta must be checked for the
chosen field range.

.. automodule:: starkzee.stark_map
   :members:
   :show-inheritance:
