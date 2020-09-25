.. _ListFiles:

######################
CWITools .list Files
######################

The ``.list`` file is a central element for a number of CWITools scripts (:ref:`Scripts`). This file is used to tell the scripts which input cubes to work with for a specific set of observations, and where to find them. They make it easy to run batch operations editing many input cubes at once. The best way to understand them is with an example.

Usage Example
===========

Let's say I observed the galaxy M51 and took two exposures (#111 and #112) with KCWI on 191227, and two exposures (#100 and #101) on the following night (191228). Let's say that KCWI data is stored in ``/home/donal/data/kcwi/`` on my computer. Specifically, the reduced data cubes for these exposures are in ``/home/donal/data/kcwi/191227/redux/`` and ``/home/donal/data/kcwi/191228/redux/``.

To work with my M51 data, I create a file called M51.list containing the following::

  #
  # CWITools LIST File
  #

  # Location of input data
  DATA_DIRECTORY = /home/donal/data/kcwi/

  # Number of directory levels to search
  SEARCH_DEPTH =  2

  # ID_LIST: one unique ID string per line, starting with '>'
  >kb191227_00111
  >kb191227_00112
  >kb191228_00100
  >kb191228_00101

Now, I can pass this file to CWITools scripts along with which *type* (e.g. `icubes.fits` or `icube.fits`) of data cube I want to work with. For example, let's say I want to crop the flux-calibrated intensity cubes for these exposures (type: `icubes.fits`) to a wavelength range of 4000A-5000A. All I have to do is:

.. code-block:: bash

   $ cwi_crop M51.list -ctype icubes.fits -wcrop 4000 5000 -ext .c.fits

or, in Python:

.. code-block:: python

   >>> from cwitools.scripts.crop import crop
   >>> crop("M51.list", ctype="icubes.fits", wcrop=(4000, 5000), ext=".c.fits")

Now, if I want to coadd these cropped data cubes, again, I just have to do:

.. code-block:: bash

   $ cwi_coadd M51.list -ctype icubes.c.fits

or, in Python:

.. code-block:: python

   >>> from cwitools.scripts.coadd import coadd
   >>> crop("M51.list", ctype="icubes.c.fits")
