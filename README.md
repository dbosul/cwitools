# CWITools 

### Installation

1. Make sure you have Python 2.7 installed as well as python packages NumPy, SciPy, Shapely, and Matplotlib.
2. Download or clone the CWITools repository into a directory on your computer
3. Run the scripts as you would any other python script

It is very helpful to add shortcuts for these tools so you don't have to type the full python command every time. For example, if you add the following line to the .bash_profile in your home directory:

> alias coadd="python /home/donal/CWITools/coadd.py"

Then (after refreshing/restarting your terminal) you will be able to just run the following:

> coadd mytarget.param icubes.fits

This will stack the icubes files for "mytarget" (see parameter files, below.)

### Parameter Files

CWITools functions using a file for each target that contains relevant information about the target such as its name, RA, DEC, redshift, etc. A template parameter file is in the main CWITools directory. You can make a copy of it and modify the values as needed for each of your targets. A quick rundown of the contents of the parameter file is:

* *NAME/RA/DEC/Z** - Self-explanatory. Basic target information ("Z" is redshift here.)
* *ZLA** - Redshift of LyA emission (can be different to systemic QSO redshift due to absorption etc.)
* *REG_FILE** - Path to a DS9 region file that indicates the location of continuum sources, for the purpose of masking and PSF subtraction. Set to "None" if not using a region file.
* *DATA_DIR* + DATA_DEPTH** - Upper level directory in which input data is located, and how many subdirectory levels to go down from there when searching for the files.
* *PRODUCT_DIR** - Directory where coadd products will be saved.

The next part of the param file is a table, with the headers:

* *IMG_ID**: A unique string identifying the image number in question. In PCWI data, this is usually just a 5-digit number. In KCWI data, this might be a longer date+number string (e.g. kb181015_00075.) The string just needs to be uniquely identifiable as that exposure.
* *SKY_ID*: Will be automatically filled during initParams.py with same value as IMG_ID for Nod-and-Shuffle data. User must manually add the appropriate SKY_ID for each IMG_ID if the data is non-NAS and they want to run the skySub.py script. 
* *XCROP/YCROP*: These specify the pixels that will be trimmed from the cube during cubeCrop.py. Auto-populated during initParams.py but can be modified by user.
* *WCROP*: This specifies (in Angstroms) the lower and upper wavelengths (by default: WAVGOOD0/WAVGOOD1 from the Header values.) 

(\*Asterisks indicate the fields that the user must populate manually before running initParams.py)

### Executing Scripts

Every script is run with the same syntax:

> python \<scriptName\> \<paramFile\> \<cubeType\>
  
* *script* - the script name.
* *target.param* - pointer to the target parameter file you want to use.
* *cubeType* - the type of input cube you want to work with (should include the .fits file extension.)

The scripts in CWITools are:

* *initParams* -Loads FITS objects and uses headers to populate parameter file.
* *fixWCS* - Use RA/DEC of the target and sky lines to fix WCS. Appends ".wc"
* *cubeCrop* - Trims bad/unwanted pixels from the input cubes. Appends ".c"
* *skySub* - Performs slice-by-slice sky subtraction using SKY_IDs and IMG_IDs. Appends ".ss"
* *coadd* - Stack the input frames. Output is saved as NAME+cubeType+.fits in PRODUCT_DIR.
* *lineCrop* - Crops the cube to a limited velocity window around a particular emission line. Appends ".lyA", ".CIV" etc.
* *psfSub* - Subtract point-sources in the field with a 2D scaling method. Appends ".ps"
* *bkgSub* - Fits a low-order polynomial to the continuum wavelengths in each spaxel of the cube. Appends ".bs"
* *aSmooth* - Adaptively smooth a coadded cube. (Same syntax, only works at coadd level.)

