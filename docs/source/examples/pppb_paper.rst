PPPB paper comparison scripts
=============================

The ``examples/PPPB paper`` directory contains two separate checks inspired by
Ferri, Peyrusse & Calisti (2022).

``figure1.py`` constructs a qualitative full Balmer-series spectrum at 100,
500, and 1000 T, with and without the quadratic Zeeman contribution. It does
not establish quantitative reproduction because the production profile solver
still omits inter-shell configuration interaction. Its static profiles use
bundled NIST hydrogen energies by default (``use_empirical_data=True``).

``figure2_gbk_magnetic.py`` audits the magnetic GBK cutoff for Lyman-alpha at
``Ne=1e23 m^-3`` and ``Te=5 eV``. It compares StarkZee's field-dependent
``G(delta_omega)`` with the ZEST-equivalent GBK, Lee, and Dufty functions. The
script documents why the published Figure 2 behavior is consistent with the
implemented ``omega_e=1/tau_e`` convention rather than the paper text's
``2*pi/tau_e``. This is a figure-consistency argument, not an archived PPPB
code-output comparison. This script evaluates collision kernels directly;
it does not construct atomic levels, so ``use_empirical_data`` does not
affect Figure 2.

Run them from the repository root:

.. code-block:: powershell

   python "examples/PPPB paper/figure1.py"
   python "examples/PPPB paper/figure2_gbk_magnetic.py"

Figure 1 script
---------------

.. literalinclude:: ../../../examples/PPPB paper/figure1.py
   :language: python
   :linenos:

Figure 2 GBK magnetic-cutoff audit
-----------------------------------

.. literalinclude:: ../../../examples/PPPB paper/figure2_gbk_magnetic.py
   :language: python
   :linenos:
