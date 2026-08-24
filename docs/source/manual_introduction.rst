Introduction
====================

StarkZee computes hydrogenic plasma line shapes when electric microfields and magnetic fields must be treated together rather than as independent corrections. The target problem is common in fusion spectroscopy, dense laboratory plasmas, and compact-object atmospheres: ionic microfields broaden and mix the emitter states through the Stark effect, while an imposed magnetic field splits and further mixes the same states through the Zeeman effect. When the two energy scales are comparable, a post-processed Stark profile with a Zeeman triplet added afterward is no longer sufficient.

The code therefore diagonalizes the Stark-Zeeman Hamiltonian directly at each sampled ionic microfield. It then averages the resulting polarized profiles over the microfield distribution and adds electron-impact broadening and optional ion dynamics through the Frequency Fluctuation Model (FFM). Thermal Doppler broadening can be included inside either solver; instrumental broadening is an explicit post-processing convolution.

These docs are organized from use to derivation. The Calculation Workflow and Package Map section gives the computational workflow and package map. The Physics and Numerical Formulation section derives the line-shape equations and the atomic, plasma, and numerical ingredients. Later sections summarize validation features, built-in comparison models, and the physical limits of the approximations.

.. _`sec:workflow`:

