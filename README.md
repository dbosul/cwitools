# CWI Tools

'CWI Tools' is a python (currently 2.7 - working on python3 compatibility) tool-kit for working with PCWI and KCWI data. The main function is to create stacked/coadded cubes from a set of input cubes, but there are also scripts correct the WCS, crop data, PSF subtract and continuum (polynomial spectral fit) subtract the data.

The way it works is that for each set of input cubes (the final data products from the main reduction pipelines)you create a parameter (.param) file. This file contains the RA/DEC/redshift of the target, the directories for loading/saving data products, and a DS9 region file indicating the location of continuum sources. The parameter files also store automatically generated parameters such as the number of pixels to crop in each dimension when coadding, the sky image to be used for each object image (for non nod-and-shuffle data) and the position angle of each exposure. You can modify these manually if needed.

CWITools can be used either as a command-line tool, or as a Python library. 

To use as a command line tool: 

1. Download/clone the repository to a directory on your computer (e.g. /home/user/CWITools/)
2. Download any python dependencies you need (e.g. NumPy, SciPy, Shapely, Matplotlib)
3. Make parameter files for each target you want to coadd (see below.)
3. Run the scripts as you would any other python script (e.g. ">python /home/user/CWITools/coadd.py myparamfile.param icubes.fits" )

To use as a python package:

1. Download/clone the repository to a directory on your computer (e.g. /home/user/code/CWITools/)
2. Add the parent directory to your $PYTHONPATH variable
3. In any python script, add "import CWITools", "import CWITools.multicube" or whatever you want to import.

Making parameter files:

1. Make a copy of the template parameter file (multicube/template.param) 
2. Update all relevant info in the file (RA/DEC/z etc.)
3. For each input image, add a unique image identifier on a new line preceded by. Image identifiers are usually an image number for PCWI data or a date-number string for KCWI data (e.g kb181014_00033)

Reporting bugs and platform issues: email me at dosulliv@caltech.edu! 
