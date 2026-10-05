Examples
========

All example scripts are in the ``examples/`` directory of the repository.
Run them from the project root, e.g.:

.. code-block:: bash

    python examples/example_balmer_lineprofile.py

Each row below links to a page with the full source code and the resulting
figure(s).

.. list-table::
   :header-rows: 1
   :widths: 30 55 15

   * - Example
     - What it shows
     - Script(s)
   * - :doc:`examples/example_balmer_lineprofile`
     - Hα–Hε at DIII-D-like edge conditions via the high-level
       ``LineProfile`` class
     - 1
   * - :doc:`examples/example_halpha`
     - Hα stick spectrum + broadened profile across B and :math:`N_e`
     - 1
   * - :doc:`examples/example_transitions`
     - Discrete transition tables, oscillator strengths, and the
       fine-structure → Zeeman-dominated progression
     - 1
   * - :doc:`examples/reproduce_fig1`
     - Full Balmer-series spectrum in the style of Ferri et al. (2022) Fig. 1
     - 1
   * - :doc:`examples/pppb_paper`
     - PPPB-paper Figure 1 qualitative spectrum and Figure 2 magnetic GBK
       cutoff audit
     - 2
   * - :doc:`examples/test_lyman_alpha`
     - H Ly-α from the intermediate-field to the Paschen-Back regime
       (100–1000 T)
     - 3
   * - :doc:`examples/model_comparison`
     - StarkZee (static + FFM) vs. all five built-in reference models
     - 1
   * - :doc:`examples/model_comparison_non_hermit`
     - Full non-Hermitian PPP impact-limit collision operator vs. the built-in
       reference models
     - 1

.. toctree::
   :maxdepth: 1
   :hidden:

   examples/example_balmer_lineprofile
   examples/example_halpha
   examples/example_transitions
   examples/reproduce_fig1
   examples/pppb_paper
   examples/test_lyman_alpha
   examples/model_comparison
   examples/model_comparison_non_hermit
