#
# CWITOOLS - SIMPLE PIPELINE EXAMPLE - Extracting nebular emission around a QSO
#
# This script provides example of how to use CWITools from the command-line in
# a linux environment, and should extend simply to MacOS environments as well.
#
#
# The pipeline shown here follows a typical flow: correct data, coadd data,
# extract 3D signal, and then make scientific products.
#

# Step 1 - Cropping the data cubes.
cwi_crop example.list -ctype icubes.fits mcubes.fits vcubes.fits -xcrop 5 28 -ycrop 15 80 -wcrop 4085 6079

# Step 2-A - Measure the coordinate system to create a 'WCS correction table'
cwi_measure_wcs example.list -ctype icubes.c.fits -xymode src_fit -zmode xcor -radec 149.689272107 47.056788021 -plot

# Step 2-B - Apply the new WCS table to the cropped data cubes
cwi_apply_wcs example.wcs -ctype icubes.c.fits mcubes.c.fits vcubes.c.fits

# Step 3 - Subtract the QSO from the input cubes
# We mask nebular emission at a redshift z with a line-width of 750 km/s.
# We also mask the wavelength ranges 4210A:4270A and 5570A:558A, which contain
# broad LyA emission and a bright sky line.
# Masking these regions improves the empirical PSF model.
cwi_psf_sub icubes.c.wc.fits -clist example.list -radec 149.689272107 47.056788021 -r_fit 1.0 -r_sub 5.0 -mask_neb_z 2.49068 -mask_neb_dv 750 -wmask 4210:4270 5570:5585 -var vcubes.c.wc.fits

# Step 4A - Coadd the cropped, wcs-corrected data cubes
cwi_coadd example.list -ctype icubes.c.wc.fits -masks mcubes.c.wc.fits -var vcubes.c.wc.fits -verbose -out example_coadd.fits

# Step 4B - Coadd the PSF-subtracted data cubes
cwi_coadd example.list -ctype icubes.c.wc.ps.fits -masks mcubes.c.wc.fits -var icubes.c.wc.ps.var.fits -verbose -out example_coadd.ps.fits

# Step 5 - Subtract residual background
# Again, we mask the same wavelengths to avoid over-fitting signal.
cwi_bg_sub example_coadd.fits -method polyfit -poly_k 3 -var example_coadd.var.fits -mask_neb_z 2.49068 -mask_neb_dv 750 -wmask 4210:4270 5570:5585

#Step 6A - Create source mask for the coadd based on our DS9 region file
cwi_get_mask example.reg example_coadd.fits -out psf_mask.fits

#Step 6B - Apply the mask to the data
cwi_apply_mask psf_mask.fits example_coadd.ps.bs.fits

#Step 7 - Segment 
