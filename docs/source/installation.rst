Installation
============

Requirements
------------

- Python ≥ 3.9
- NumPy ≥ 1.22
- SciPy ≥ 1.8

Optional:

- Matplotlib ≥ 3.5 — for plotting examples
- numexpr ≥ 2.8 — fused, multithreaded static-profile kernel (``pip install -e ".[fast]"``);
  same results to rounding, 2–6× faster profiles
- pytest ≥ 7 — for running the test suite

Install
-------

Clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/<your-org>/starkzee.git
    cd starkzee

    # Core only (numpy + scipy)
    pip install -e .

    # With matplotlib
    pip install -e ".[plot]"

    # With dev tools (pytest + matplotlib)
    pip install -e ".[dev]"

Running the tests
-----------------

.. code-block:: bash

    pytest tests/ -v

The current test-suite snapshot contains 869 passing tests. It also reports
93 warnings for documented fallbacks, domain checks, and optional
dependency paths.
