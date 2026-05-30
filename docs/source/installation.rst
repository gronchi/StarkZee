Installation
============

Requirements
------------

- Python ≥ 3.9
- NumPy ≥ 1.22
- SciPy ≥ 1.8

Optional:

- Matplotlib ≥ 3.5 — for plotting examples
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

All 332 tests should pass.

Command-line interface
----------------------

After installation the ``starkzee`` command is available:

.. code-block:: bash

    starkzee --help
    starkzee -Z 1 -B 5 --Ne 1e20 --Te 5 -o profile.txt -p profile.png
