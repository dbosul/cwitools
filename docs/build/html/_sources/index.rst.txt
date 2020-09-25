.. _Overview:

######################
Overview
######################

| Welcome to the documentation for CWITools!

| Here, you will find a list of the modules, sub-modules and functions within the package. CWITools is designed as a modular set of tools from which observers can construct data analysis pipelines to suit their own scientific needs.

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

| In addition to these core modules, there are two library modules for useful functions:

* The :ref:`Coordinates` contains commonly-used functions relating to coordinate systems and FITS Headers (e.g. obtain the wavelength axis from a 3D header).
* The :ref:`Utilities` is mostly a set of tools for internal use, but contains several functions that observers may find useful, such as obtaining an auto-populated list of nebular emission lines or sky lines.

| Finally, CWITools contains a :ref:`Scripts`, which provides high-level functionality accessible both from a Python session or the command-line of a bash terminal.

Indices and tables
==================

.. toctree::
   :hidden:

   installation
   scripts
   listfiles
   examples
   genindex


* :ref:`modindex`
* :ref:`search`
