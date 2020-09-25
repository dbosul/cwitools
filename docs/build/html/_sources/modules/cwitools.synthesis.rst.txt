.. _Synthesis:

Synthesis Module (cwitools.synthesis)
=======================================

The synthesis module contains functions focused on generating common products from an extracted signal. For example, generating surface brightness maps, integrated spectra, radial profiles, and velocity (z-moment) maps for 3D objects identified after segmentation. This module also contains tools that can be applied without having a 3D mask pre-made, such as generating white-light images from data cubes, generating pseudo-narrow-band images, and getting an integrated spectrum from an annular region.

.. automodule:: cwitools.synthesis







   .. rubric:: Functions

   .. autosummary::

      cylindrical
      obj_moments
      obj_moments_doublet
      obj_sb
      obj_spec
      pseudo_nb
      radial_profile
      sum_spec_r
      whitelight
