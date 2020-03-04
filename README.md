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

CWITools Parameter Files
-------------------------
A central feature of CWITools is the parameter file. This file contains information about the target you are working with such as the true RA/DEC, the directories for input and output, and a set of unique IDs identifying input data (e.g. kbYYMMDD_XXXXX for KCWI, or imageXXXXX for PCWI.)

A template parameter file is provided in the main directory (template.param.) Copy, rename and paste this file anywhere you want (ideally the directory you're working in) and update it with your target's parameters. This can then be passed as an argument to most of the command-line tools. CWITools also has a parameters module which is used to read/write these files within the python shell.

Command-line Usage
------------------
When installed with pip, CWITools adds command-line utilities (e.g. `$cwi_coadd -h`) which can be used directly in a terminal. When installed via GitHub, the tools can still be run using `$python3 /path/to/cwitools/scripts/cwi_coadd.py -h`. Each of the command-line tools has a help menu available with the `-h` flag. 

Package Organizational Chart
----------------------------

![CWITools Organizational Chart](https://github.com/dbosul/cwitools/blob/v0.5/cwitools/data/CWITools_Org.png)
