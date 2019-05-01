# CWITools 

CWITools is a package for working with data from the Palomar and Keck Cosmic Web Imagers, with the ultimate goal of extracting and studying nebular emission in 3D. The package is intended to pick up where the standard data reduction pipelines for these instruments ends. The most widely useful functionality it provides is the ability to coadd data cubes produced by the PCWI/KCWI pipelines (provided the WCS information in the headers is accurate.) CWITools also provides a suite of smaller tools for actions such as cropping, re-binning, masking, and sky-subtraction. 

The package provides a pipeline for extracting and analysing extended line emission in Cosmic Web Imager data. Beyond correcting and coadding the input data cubes, this pipeline includes point-source subtraction, background/continuum subtraction, variance estimation, adaptive kernel smoothing, segmentation and generation of science products such as pseudo-narrow-band images, velocity maps, velocity dispersion maps and 1D nebular spectra.

The documentation for this package is still under development, as are some of the routines within. Please contact dos@astro.caltech.edu if you run into any trouble trying to use it.

## Table of Contents

1. Installation
1. Overview of Usage
1. Creating and using CWITools parameter files
1. Correcting/adjusting Cubes
1. Coadding
1. QSO Subtraction
1. Background/Continuum Subtraction
1. Variance Estimation
1. Adaptive Kernel Smoothing
1. Object Segmentation
1. Generating science products

## 1. Installation

Although this package is not available via pip or the standad linux apt-get methods, manual installation is very straight-forward. 

1. Make sure you have Python 2.7 installed as well as the Python packages NumPy, SciPy, Shapely, and Matplotlib.
2. Download or clone the CWITools repository into a directory anywhere on your computer
3. (Optional) Add the CWITools directory to your python path
4. Run the scripts with the usual python syntax (e.g. "python coadd.py -h")

It can also be very helpful to add aliases as shortcuts for these tools so you don't have to type the full python command every time. For example, on Linux, if you add the following line to the .bash_profile file in your home directory:

> alias coadd="python /home/donal/CWITools/coadd.py"

Then (after refreshing/restarting your terminal) you will be able to just run the following instead of typing out the full python command:

> coadd mytarget.param icubes.fits

##  2. Overview of Usage

For the most part, CWITools functions as any other python scripts. All of the scripts are run with the syntax "python <scriptName> <parameters>". Currently, most of the scripts also have help menus, which can be accessed with the flag "-h"; e.g. "python coadd.py -h".



## 1. Creating a parameter file 

1. Copy the template parameter file
> cp ~/CWITools/template.param M81_blue.param

2. Open and edit the contents of the file.
> gedit M81_blue.param

3. Once the minimum required fields are filled in (see above) fill out the rest of the parameters automatically using initParams.
> python ~/CWITools/fillParams.py M81_blue.param icubes.fits

The parameter file is now ready to be used. Alternatively, you can use the new script "initParams" to start a parameter file from scratch and auto-fill the header details. This script takes no arguments:

> python ~/CWITools/initParams.py

Then follow the instructions.

#### 2. Correcting and coadding data

Correct the WCS of the cubes. This is an interactive step.
> python ~/CWITools/fixWCS.py M81_blue.param icubes.fits

Crop the WCS-corrected cubes to trim off bad/excess pixels in each dimension.
> python ~/CWITools/cubeCrop.py M81_blue.param icubes.wc.fits

Coadd the WCS-corrected, cropped cubes.
> python ~/CWITools/coadd.py M81_blue.param icubes.wc.c.fits




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

Most scripts now have a syntax help menu built in, which you can access using the flag '-h'. E.g. "python coadd.py -h". 

In general, CWITools scripts are run with the following syntax:

> python \<scriptName\> \<paramFile\> \<cubeType\>
  
* *script* - the script name.
* *target.param* - pointer to the target parameter file you want to use.
* *cubeType* - the type of input cube you want to work with (should include the .fits file extension.)

The scripts in CWITools are:

* *initParams* - Create a parameter file through interactive script instead of copying template.
* *fillParams* - Loads FITS objects and uses headers to populate existing parameter file.
* *fixWCS* - Use RA/DEC of the target and sky lines to fix WCS. Appends ".wc"
* *cubeCrop* - Trims bad/unwanted pixels from the input cubes. Appends ".c"
* *skySub* - Performs slice-by-slice sky subtraction using SKY_IDs and IMG_IDs. Appends ".ss"
* *coadd* - Stack the input frames. Output is saved as NAME+cubeType+.fits in PRODUCT_DIR.
* *lineCrop* - Crops the cube to a limited velocity window around a particular emission line. Appends ".lyA", ".CIV" etc.
* *psfSub* - Subtract point-sources in the field with a 2D scaling method. Appends ".ps"
* *bkgSub* - Fits a low-order polynomial to the continuum wavelengths in each spaxel of the cube. Appends ".bs"
* *aSmooth* - Adaptively smooth a coadded cube. (Same syntax, only works at coadd level.)

### Examples
