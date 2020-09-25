.. _Extraction:

Extraction Module (cwitools.extraction)
=======================================

The extraction module contains functions focused on isolating a 3D signal (e.g. a nebular emission region) within a data cube. The first part of the extraction process typically involves modeling and subtracting continuum sources, removing slowly-varying background signals, and masking foreground sources or other regions as needed. When the cube only contains the desired signal, a segmentation process can be used to identify the 3D contours of the emitting region and extract it for further analysis.

.. automodule:: cwitools.extraction

   .. rubric:: Functions

   .. autosummary::

      apply_mask
      bg_sub
      cutout
      detect_lines
      obj2binary
      psf_sub
      psf_sub_all
      reg2mask
      segment
      smooth_cube_spatial
      smooth_cube_wavelength
      smooth_nd
