microfield
==========

Ion microfield distributions and quadrature grids.  The unscreened Holtsmark
field distribution is written in reduced field units :math:`\beta = F/F_0` as

.. math::

   W_H(\beta) = \frac{2\beta}{\pi}
   \int_0^\infty y\sin(\beta y)\,e^{-y^{3/2}}\,dy.

The default Hooper screened form modifies the exponent with the screening
function :math:`S(y,a)`,

.. math::

   W(\beta,a) = \frac{2\beta}{\pi}
   \int_0^\infty y\sin(\beta y)\,
   e^{-y^{3/2}S(y,a)}\,dy,
   \qquad
   a = r_e/\lambda_D.

.. automodule:: starkzee.microfield
   :members:
   :show-inheritance:
