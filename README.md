# CWITools (Alpha Version)

CWITools is a package for working with data from the Palomar and Keck Cosmic Web Imagers, with the ultimate goal of extracting and studying nebular emission in 3D. The package is intended to pick up where the standard data reduction pipelines for these instruments ends. The most widely useful functionality it provides is the ability to coadd data cubes produced by the PCWI/KCWI pipelines (provided the WCS information in the headers is accurate.) CWITools also provides a suite of smaller tools for actions such as cropping, re-binning, masking, and sky-subtraction. 

The package provides a pipeline for extracting and analysing extended line emission in Cosmic Web Imager data. Beyond correcting and coadding the input data cubes, this pipeline includes point-source subtraction, background/continuum subtraction, variance estimation, adaptive kernel smoothing, segmentation and generation of science products such as pseudo-narrow-band images, velocity maps, velocity dispersion maps and 1D nebular spectra.

The documentation for this package is still under development, as are some of the routines within. Please contact dos@astro.caltech.edu if you run into any trouble trying to use it.

## Development Status

I am currently in the process of updating all scripts to use Python's argparse package, which allows users to use flags and view help menus. Most of the main scripts currently have it (coadd, psfSub, bkgSub, etc.). I am also working on updating documentation for the package. See below for the new, more detailed README. My next step will be to add docstrings to the code to generate an API and finally create a wiki.

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

> python fixWCS.py <paramFile> <cubeType> [<optional flags>]
  
The script 'fixWCS.py' has a help menu built in, which you can access by running the script with the flag "-h". This menu will explain the optional flags. Basically, fixWCS uses the known RA/DEC of your target (or an alternative point source) to adjust the header WCS, and uses a known sky emission line to ensure the wavelength solution is correct.

The fixWCS script saves output files with the extension ".wc.fits" (wc = wcs corrected.)

### Cropping Data Cubes

Sometiems you will want to crop the cube to remove edge artefacts, empty colums/rows and bad wavelength regions. While CWITools automatically populates default cropping parameters in the parameter file, you are free to edit this manually depending on your preference and visual inspection of the data. To crop your data, simply use the syntax

> python crop.py <paramFile> <cube type>
  
The crop script saves output files with extension ".c.fits" (c = cropped.)

## 5. Coadding

Before coadding, you should make sure your WCS information is correct in each of the input files. You can do this using the fixWCS script described above, or by your own custom means. The coadd script relies on the WCS information in input files to rotate, scale and align them onto a single canvas. 

If you do not have a parameter file for your target, read above also. It's quite simple but needed for this step. 

To coadd, say, the "icubes.fits" files for your target M81 (for which you made the parameter file M81.param) - you would simply run:

> python coadd.py M81.param icubes.fits

The output cube will be saved as <PRODUCT_DIRECTORY>/<TARGET_NAME>.<TARGET_TYPE>.fits - where target name and product directory are the ones you have defined in the parameter file. To see a full list of options for coadd.py, use the "-h" flag.

## 6. Point Source Subtraction

Point source subtraction with psfSub.py can be done in one of three ways

1. One source at a time, where the user provides the x,y coordinates for each source
1. By providing a DS9 region file which indicates the locations of sources to be subtracted
1. Automatically (detects sources above a given S/N threshold in a white-light image)

The following command would subtract the point source at position (x,y)=(43,32) from the cube 'myCoadd.fits':

> python psfSub.py myCoadd.fits -pos 43,32

The following command would instead use a region file we have called 'myRegFile.reg':

> python psfSub.py myCoadd.fits -reg myRegFile.reg

And the following command would automatically detect sources that stand out above a signal-to-noise threshold of 5:

> python psfSub.py myCoadd.fits -auto 5.0

Two optional, but important parameters are 'rmin' and 'rmax'. 'rmin' sets the inner radius, which is used to fit the PSF. Data inside this radius cannot be used to measure extended emission, so it should ideally be about the HWHM value of your source's point-spread function. 'rmax' sets the radius out to which the model PSF subtracted, this should be just large enough to fully subtract the source but ideally no larger, as all subtraction adds noise. A full help menu is available by executing "python psfSub.py -h"

psfSub.py saves output cubes with the extension ".ps.fits" (ps = PSF subtracted.)


## 7. Background Subtraction

Background subtraction is performed using one of two methods.

1. Performing a low-order polynomial fit to the spectrum in each source
2. Subtracting a median-filtered spectrum from each spaxel

As an example, to perform background subtraction on the cube "myCoadd.ps.fits" using a polynomial of degree three as the continuum model, we would execute:

> python bkgSub.py myCoadd.ps.fits -method polyfit -k 3

To perform background subtraction with a median filtering approach, where the window for the median filter is 31 pixels wide, we would execute:

> python bkgSub.py myCoadd.ps.fits -method medfilt -w 31

The flag -zmask can be used to mask emission wavelengths, but be careful when doing this with median filtering! The unmasked points in any median filter window must be sufficient to get a reliable median. For a full list of options, execute "python bkgSub.py -h".

bkgSub.py saves output files with the extension ".bs.fits" (bs = background subtracted.)

## 8. Variance Estimation

The variance estimation tool is very straight forward. Given an input data cube, it will output a corresponding variance cube which can be then used to esimate signal-to-noise and/or perform adaptive smoothing. To estimate the variance in a cube called "myCoadd.fits" - one would simply execute:

> python estimateVariance.py myCoadd.fits

There are a few optional flags in this method. A help menu can be accessed by executing "python estimateVariance.py -h"

## 9. Adaptive Kernel Smoothing

Adaptive Kernel Smoothing is a method which produces optimally smoothed cubes - small bright features are smoothed on small scales and large, faint features are smoothed on large scales (see "ASMOOTH" paper by Ebeling, White & Rangarajan for full 2D version of the algorithm.)

aSmooth.py takes an input cube and a variance cube as its main parameters. An optional but important parameter is the minimum signal-to-noise level you would like to smooth to. To run adaptive smoothing on a cube "myCoadd.fits" with the variance estimate "myCoadd.var.fits" and aim for a signal-to-noise of 3, the code would be executed as follows:

> python aSmooth.py myCoadd.fits myCoadd.var.fits -snr 3

aSmooth produces output files with the extension ".AKS.fits" and has a help menu which can be accessed using "python aSmooth.py -h"

## 10. Object Segmentation

This method segments data into regions in which the values lie above a certain threshold (fixed at 0 for now, to be updated soon) and returns a 3D mask cube with object labels. Ideally, this script is run on adaptively smoothed data, and the thresholding will return a mask of "detected" objects. This mask can then be used either to inspect the data visually, or with CWITools 'make.py' script, which uses 3D object masks to generate science products.

If we wanted to segment our adaptively smoothed cube, called "myCube.AKS.fits", then simply enough, the script is run as follows:

> python segment.py myCoadd.AKS.fits

It will output a file with the extension ".OBJ.fits".

## 11 Generating Science Products

There are four main types of science product that can be generated from input cubes using CWITools' "make.py"

- White light image
- pseudo-Narrow-band Image
- velocity maps (central vel. and dispersion)
- nebular spectra

The flag "-type" lets you choose which of these you make. The first option, 'wl', used to make a white-light image, requires no additional arguments. The next four require you to provide an object mask and indicate the ID(s) of the object(s) you wish to include when making the products:

- nb: pseudo-narrow-band image
- vel: velocity maps (produces both velocity offset and dispersion)
- spc: one-dimensional spectrum for the nebula
- tri: produce all three of the above

For example, say you have an object mask (produced using 'segment.py' - see above) called 'myCoadd.AKS.OBJ.fits' and, by visual inspection in ds9, you see the nebula you are interested in has an ID of 32. Then, to make a narrow-band image of this nebula, you would use the following command:

> python make.py myCoadd.fits -type nb -obj myCoadd.AKS.OBJ.fits -objID 32

There are a number of other options in make.py for controlling the size, centering, and central velocities of the data products. You can access a full list of options by running "python make.py -h"

## 12 Contributing

If you would like to get involved with developing this package, adding your own tools, and fixing any bugs you find, please contact me at dos@astro.caltech.edu.


