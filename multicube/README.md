# Multi-cube Scripts

Each of the multi-cube scripts takes only a parameter file and cube type as command line arguments.

The three scripts are run as follows:

**python coadd.py [parameterFile] [cubeType]**

**python bkgSub.py [parameterFile] [cubeType]**

**python psfSub.py [parameterFile] [cubeType]**

[parameterFile] is based on a template and specifies the ra, dec, redshift, location of the data etc. See below for how to create one.

[cubeType] is the search string that will be used to find the files you want (e.g. 'icuber.fits'). Make sure to include the file extension to avoid confusion (e.g. just using 'icuber' would also locate 'icuber_csub', 'icuber_cont' etc.)

# Example: Creating a coadded, continuum/PSF subtracted cube

1. Copy template.param and edit details to fit your target (say you called it "target.param")

> $cp template.param /my/data/myTarget.param
> $gedit /my/data/myTarget.param

2. Run the coadd script on non-subtracted cubes to generate stacking geometry.

> $python coadd.py target.param icuber.fits

3. (Optional) Run background/polynomial subtraction to handle scattered light or diffuse continuum. (Outputs _bs.fits cubes)

> $python bkgSub.py target.param icuber.fits

4. Run PSF Subtraction. Note that you now tell it to work on '_bs.fits' cubes - which is the output from the previous step.

> $python psfSub.py target.param icuber_bs.fits 

5. Coadd the PSF subtracted cubes (outputs _ps.fits cubes)

> $python coadd.py target.param icuber_bs_ps.fits

Done! 

