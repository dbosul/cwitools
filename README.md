CWITools Documentation on ReadTheDocs
-------------------------------------
Detailed method and module breakdowns, with examples, available at https://cwitools.readthedocs.io/en/latest/

General Description
-------------------
CWITools is a Python3 package developed to help observers crop, coadd, and analyze data taken with the Palomar and Keck Cosmic Web Imagers (PCWI and KCWI.) The package should be easily extendable to all 3D integral-field spectrograph data, so please contact me if you wish to use it for another instrument and encounter problems!

CWItools has several command-line utilities, which require no Python knowledge to use, and can also be used within the Python3 environment (e.g. `from cwitools import coordinates`.) The packge is currently in its first official beta release - and a paper is in the works! Please report bugs here on GitHub, or email me if you need help at dosulliv@caltech.edu.

Installation
------------

CWITools is available from the Python Package Index (PyPI) via pip, and can be installed with `pip install cwitools` or by downloading this GitHub repository and running `python3 setup.py install` from the main directory. I am working on uploading the package to the Anaconda cloud. 

Package Organizational Chart
----------------------------

![CWITools Organizational Chart](https://github.com/dbosul/cwitools/blob/v0.5/cwitools/data/CWITools_Org.png)
