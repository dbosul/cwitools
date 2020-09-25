.. _Scripts:

######################
Scripts Module
######################

The :ref:`Scripts` represents the intended primary usage-mode of CWITools for most users. It provides a number of high-level functions which can be strung together to form a pipeline. Unlike functions in the core modules (which typically take HDU objects as input and return updated HDU objects), these scripts read and write FITS files directly, such that each one provides a complete analysis step (e.g. load input cubes, crop, save cropped cubes).

Python Environment
================================

These scripts can be used by importing them into a Python session as functions, allowing users to string analysis steps together in a Python script. All scripts are imported with the format ``from cwitools.scripts.XXX import XXX`` where ``XXX`` is the script name. For example, here is a simple example of how you would load two FITS data cubes (cube1.fits and cube2.fits), crop them to a wavelength range of 4000A-5000A and save the cropped cubes with the extension ".cropped.fits":

.. code-block:: python

   >>> from cwitools.scripts.crop import crop
   >>> crop(["cube1.fits", "cube2.fits"], wcrop=(4000, 5000), ext=".cropped.fits")


Console
================================
For users who prefer to work from the console/terminal or write bash scripts, these scripts can be executed directly from the command-line. Upon installation of CWITools, a number of aliases of the form ``cwi_XXX``, where ``XXX`` is the script name, are added to the user's environment. They can be executed directly from the command line as follows. Again, this is an example of how you would load two FITS data cubes (cube1.fits and cube2.fits), crop them to a wavelength range of 4000A-5000A and save the cropped cubes with the extension ".cropped.fits":

.. code-block:: bash

   $ cwi_crop cube1.fits cube2.fits -wcrop 4000 5000 -ext .cropped.fits

| Each of these scripts comes with a help menu, which can be accessed by running the script with the **-h** flag (e.g. ``cwi_crop -h``).
