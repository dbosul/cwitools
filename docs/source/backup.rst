.. _CWITools:

######################
CWITools Documentation
######################

| Welcome to the documentation for CWITools. Here, you will find a list of the modules, sub-modules and functions within the package.

*********
Overview
*********

| CWITools is designed as a modular set of tools from which observers can construct data analysis pipelines to suit their own scientific needs.

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

| Finally, CWITools contains a :ref:`Scripts`, which provides high-level functionality accessible both from a Python session or the command-line of a bash terminal. See below for more on this.

*****************
CWITools Scripts
*****************

The :ref:`Scripts` represents the intended primary usage-mode of CWITools for most users. It provides a number of high-level functions which can be strung together to form a pipeline. Unlike functions in the core modules (which typically take HDU objects as input and return updated HDU objects), these scripts read and write FITS files directly, such that each one provides a complete analysis step (e.g. load input cubes, crop, save cropped cubes).

Script Usage: Python Environment
================================

These scripts can be used by importing them into a Python session as functions, allowing users to string analysis steps together in a Python script. All scripts are imported with the format ``from cwitools.scripts.XXX import XXX`` where ``XXX`` is the script name. For example, here is a simple example of how you would load two FITS data cubes (cube1.fits and cube2.fits), crop them to a wavelength range of 4000A-5000A and save the cropped cubes with the extension ".cropped.fits":

.. code-block:: python

   >>> from cwitools.scripts.crop import crop
   >>> crop(["cube1.fits", "cube2.fits"], wcrop=(4000, 5000), ext=".cropped.fits")


Script Usage: Console
================================
For users who prefer to work from the console/terminal or write bash scripts, these scripts can be executed directly from the command-line. Upon installation of CWITools, a number of aliases of the form ``cwi_XXX``, where ``XXX`` is the script name, are added to the user's environment. They can be executed directly from the command line as follows. Again, this is an example of how you would load two FITS data cubes (cube1.fits and cube2.fits), crop them to a wavelength range of 4000A-5000A and save the cropped cubes with the extension ".cropped.fits":

.. code-block:: bash

   $ cwi_crop cube1.fits cube2.fits -wcrop 4000 5000 -ext .cropped.fits

| Each of these scripts comes with a help menu, which can be accessed by running the script with the **-h** flag (e.g. ``cwi_crop -h``).

*********************
Downloadable Examples
*********************

To help new users get familiar with developing their own analysis pipeline and using CWITools, we have prepared a Github repository with sample data and scripts that the user can download and run on their own machine. See the README at https://github.com/dbosul/cwitools-examples for more information.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
