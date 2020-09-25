.. _Reduction:

﻿Reduction Module (cwitools.reduction)
===================================

The reduction module contains functions used in order to prepare data for analysis. This includes cropping, coadding, correcting coordinate systems, and correcting variance estimates. Since it is quite a large module, it is broken down further into four sub-modules:

* :ref:`Reduction.Cubes`
* :ref:`Reduction.Units`
* :ref:`Reduction.WCS`
* :ref:`Reduction.Variance`


Cubes
------
Functions applied or related directly do data cubes, such as re-binning, cropping, or coadding.

Units
------
Functions related to unit corrections, currently consisting of an air-to-vacuum wavelength correction and a heliocentric wavelength correction.

WCS
----
Functions for correcting the world-coordinate system (WCS) of the pipeline data cubes. This is done either in absolute terms, by fitting to a known feature, or in relative terms, by cross-correlating the input data and making them at least self-consistent.

Variance
----
Functions for estimating variance empirically, scaling variance estimates, and measuring the effects of covariance in the data.
