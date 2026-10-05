microfield
==========

Ion microfield distributions and quadrature grids.  The unscreened Holtsmark
field distribution is written in reduced field units :math:`\beta = F/F_0` as

.. math::

   W_H(\beta) = \frac{2\beta}{\pi}
   \int_0^\infty y\sin(\beta y)\,e^{-y^{3/2}}\,dy.

The default screened distribution uses Potekhin fits. The explicitly selected,
unvalidated legacy Hooper-like ansatz modifies the exponent with the screening
function :math:`S(y,a)`,

.. math::

   W(\beta,a) = \frac{2\beta}{\pi}
   \int_0^\infty y\sin(\beta y)\,
   e^{-y^{3/2}S(y,a)}\,dy,
   \qquad
   a = r_e/\lambda_D.

The net radiator ``emitter_charge`` is separate from background ``Z_bar``.
For hydrogen-like profile calls it defaults to ``Z-1`` and derives the
neutral/charged-point selector. The implemented Potekhin fits use only that
binary selector; they do not model the magnitude of the emitter charge. Use a
traceable custom APEX/MD table when that dependence is required.

.. automodule:: starkzee.microfield
   :members:
   :show-inheritance:
