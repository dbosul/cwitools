# CWITools (Alpha Version)

CWITools is a package for working with data from the Palomar and Keck Cosmic Web Imagers, with the ultimate goal of extracting and studying nebular emission in 3D. The package is intended to pick up where the standard data reduction pipelines for these instruments ends. The most widely useful functionality it provides is the ability to coadd data cubes produced by the PCWI/KCWI pipelines (provided the WCS information in the headers is accurate.) CWITools also provides a suite of smaller tools for actions such as cropping, re-binning, masking, and sky-subtraction. 

The package provides a pipeline for extracting and analysing extended line emission in Cosmic Web Imager data. Beyond correcting and coadding the input data cubes, this pipeline includes point-source subtraction, background/continuum subtraction, variance estimation, adaptive kernel smoothing, segmentation and generation of science products such as pseudo-narrow-band images, velocity maps, velocity dispersion maps and 1D nebular spectra.

The documentation for this package is still under development, as are some of the routines within. Please contact dos@astro.caltech.edu if you run into any trouble trying to use it.

## Table of Contents

1. Installation
1. Overview of Usage
1. CWITools Parameter files
1. Correcting Data Cubes
1. Coadding
1. QSO Subtraction
1. Background/Continuum Subtraction
1. Variance Estimation
1. Adaptive Kernel Smoothing
1. Object Segmentation
1. Generating science products
1. Contributing
1. Credits
1. License

## 1. Installation

Although this package is not available via pip or the standad linux apt-get methods, manual installation is very straight-forward. 

1. Make sure you have Python 2.7 installed as well as the Python packages NumPy, SciPy, Shapely, and Matplotlib.
2. Download or clone the CWITools repository into a directory anywhere on your computer
3. (Optional) Add the CWITools directory to your python path
4. Run the scripts with the usual python syntax (e.g. "python coadd.py -h")

See below for more on usage.

##  2. Overview of Usage

### Executing scripts in a terminal

For the most part, CWITools functions as any other python scripts. All of the scripts are run with the syntax "python <scriptName> <parameters>". Currently, most of the scripts also have help menus, which can be accessed with the flag "-h"; e.g. "python coadd.py -h".

If you have CWITools scripts located in, say "/home/donal/CWITools/" then you would nominally need to include the full path to the scripts (e.g. "python /home/donal/CWITools/coadd.py <parameters>") or execute the scripts from within that directory. However, there are a few easy ways to make life easier here. First, on linux, you could add aliases which serve as shortcuts for the longer commands. To do this add the following line (for example) to the .bash_profile file in your home directory:

> alias coadd="python /home/donal/CWITools/coadd.py"

After refreshing/restarting your terminal, you will be able to just run the following instead of typing out the full python command:

> coadd <parameters>

Alternatively, if you want to use the Python syntax but not type the full path every time - you can add the CWITools directory to your PATH environment variable. A quick google will show you how to do this!

### Executing from within Python

If you want to import CWITools as a python package (in order to call subroutines and write your own scripts using any of the CWITools methods,) all you need to do is add the CWITools directory (wherever it is installed on your computer) to your PYTHON_PATH environmental variable. Once this is done, you'll be able to use import statements like the following:

> from CWITools import libs

> parameters = libs.params.loadparams("/home/donal/data/targets/mytarget.param")


## 3. CWITools Parameter Files 

A core component of CWITools is a type of file called a parameter file. This is simply a structured text file in which you fill out information that CWITools may need to know about a certain target such as the target's name, RA, DEC, redshift, etc. Tile is also the file that tells the pipeline which image numbers you would like to coadd, where to find them on your machine, and where to store new data products. 

### Initializing a new parameter file

A template parameter file, template.param, is provided in the main CWITools directory. To create your own, simply copy this file, rename it and modify the contents to match your own target. When creating a parameter file like this, the only information you need to add below the column headers on line 16 is one image identifier per line, preceded by a '>'. An image identifier is a unique string associated with that input frame. For PCWI, this is usually a five digit image ID, since the images are by default named "imageXXXXX.fits." For KCWI, this is usually a date_imageID string such as "190203_00017." 

Alternatively to modifying the template parameter, you can run the script "initParams.py" to walk through an interactive process to create the initial parameter file for your target.

### Auto-filling a newly made parameter file

After following either of the above methods to initialize a parameter file (let's imagine the target is M81, so we call it 'M81.param'), you can run

> python fillParams.py M81.param icubes.fits

(you don't have to use icubes.fits, you can use icubed.fits, icuber.fits, or whatever pipeline data product you have available) and CWITools will auto-populate the table with the default cropping parameters. For data that is not taken in nod-and-shuffle mode, the column 'SKY_ID' must be manually filled or verified if you would like to perform sky subtraction or correct the wavelength solution. Even if you skip the fillParams.py step and run another script such as coadd.py - they should be automatically filled for you.

The parameter file is now ready to be used. 

### General notes about using parameter files

The basic syntax of how CWITools works (for coadding and cube manipulation) is that scripts are given a parameter file and a "cube type" such as 'icubes.fits', 'ocubes.fits', 'icubed.fits' or any other PCWI/KCWI cube types. The parameter file tells the script which unique image IDs to use, while the cube type tells it which kinds of data products to work with. For example, if you wanted to coadd the flux-calibrated "icubes.fits" files for your M81 target, you would run:

> python coadd.py M81.param icubes.fits

But if you wanted to coadd a different kind of cube, such as icubed.fits, you would run

> python coadd.py M81.param icubed.fits

Straight forward enough. The reason this is useful is that CWITools adds extensions to filenames to indicate what has been done to that file. For example, a file with the name "icubes.wc.c.fits" has been wcs-corrected ('wc') and cropped ('c'). Stringing these operations together via the filename makes it easy to choose your own order of operations and work with whatever kind of data product you want.

## 4. Correcting Data Cubes

"Correcting" here means any modifications you need/want to apply before coadding. The two main functions of this kind that CWITools allows you perform are 1) to correct their WCS information (the default header WCS can sometimes be off due to pointing error or observer error) and 2) to crop the data cubes in order to trim off edge artefacts. 

### Correcting the WCS of Data Cubes

Correcting the WCS is an interactive step. The syntax for this script is:

> python fixWCS.py <paramFile> <optional flags>
  
The script 'fixWCS.py' has a help menu built in, which you can access by running the script with the flag "-h". This menu will explain the optional flags. Basically, fixWCS uses the known RA/DEC of your target (or an alternative point source) to adjust the header WCS, and uses a known sky emission line to ensure the wavelength solution is correct.

The fixWCS script saves output files with the appendix ".wc.fits" (wc = wcs corrected.)

### Cropping Data Cubes

Sometiems you will want to crop the cube to remove edge artefacts, empty colums/rows and bad wavelength regions. While CWITools automatically populates default cropping parameters in the parameter file, you are free to edit this manually depending on your preference and visual inspection of the data. To crop 
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
