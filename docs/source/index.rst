.. _Overview:

######################
Overview
######################

| Welcome to the documentation for CWITools!Here, you will find a list of the modules, sub-modules and functions within the package.


Executable Scripts
==================
| For most users, the core functionality of CWITools is contained in the :ref:`Scripts`.

| This module proides a number of high-level, executable Python which provides a number of high-level scripts (e.g. coadding a list of cubes). These can be executed directly from the command line, e.g.:

.. code-block:: bash

   $ cwi_crop cube1.fits cube2.fits cube3.fits -out my_coadd.fits

| or from within a Python session, e.g.:

.. code-block:: python

   >>> from cwitools.scripts.coadd import coadd
   >>> coadd(["cube1.fits", "cube2.fits", "cube3.fits"], out="my_coadd.fits")


Core Modules
============

The package structure of CWITools is designed as a modular set of tools from which observers can construct data analysis pipelines to suit their own scientific needs.

| The flow of such data analysis pipelines for IFU data tends to follow a universal pattern:

1. **reduction**: cropping and coadding the pipeline data cubes
2. **extraction**: isolating a target signal and removing foreground/background
3. **synthesis**: making emission maps, spectra, and other products
4. **modeling**: fitting emission line profiles, radial profiles etc.
5. **measurement**: obtaining final, scalar quantities such as size and luminosity.

| In CWITools, each of these broad steps is represented by a top-level module:

* :ref:`Reduction`
* :ref:`Extraction`
* :ref:`Synthesis`
* :ref:`Modeling`
* :ref:`Measurement`

Helper Modules
==============
| In addition to these core modules, there are two library modules for useful functions:

* The :ref:`Coordinates` contains commonly-used functions relating to coordinate systems and FITS Headers (e.g. obtain the wavelength axis from a 3D header).
* The :ref:`Utilities` is mostly a set of tools for internal use, but contains several functions that observers may find useful, such as obtaining an auto-populated list of nebular emission lines or sky lines.

Indices and tables
==================

.. toctree::
   :hidden:

   installation
   scripts
   listfiles
   examples
   citation
   genindex



* :ref:`modindex`
* :ref:`search`
