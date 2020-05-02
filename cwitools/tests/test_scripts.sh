#Run each script on test data
python3 ../scripts/cwi_pnb.py test_coadd.fits -wav 4450 -dw 20 -par ~/data/flashes/deep/deep_params/SDSS0958+4703_L_BL_4800.param -mask_psf -fit_rad 5 -sub_rad 30 -ext .pNBtest.fits
python3 ../scripts/cwi_getmask.py test_reg.reg test_coadd.fits -fit -out test_coadd.mask.fits
python3 ../scripts/cwi_applymask.py test_coadd.mask.fits test_coadd.fits -out test_coadd.M.fits
python3 ../scripts/cwi_getvar.py test_coadd.fits -out test_coadd.var.fits
python3 ../scripts/cwi_measurewcs.py test_param.param icubes.fits -out test_wcs.wcs
python3 ../scripts/cwi_applywcs.py test_wcs.wcs icubes.fits -ext .wcs_cor.fits
python3 ../scripts/cwi_crop.py -cube test_icubes.fits -auto -ext .crop.fits
python3 ../scripts/cwi_rebin.py test_icubes.fits -xybin 2 -zbin 2 -out test_icubes.rebin.fits
python3 ../scripts/cwi_psfsub.py -cube test_coadd.fits -ext .ps.fits
python3 ../scripts/cwi_bgsub.py test_coadd.ps.fits -method polyfit -k 1 -ext .bs.fits
python3 ../scripts/cwi_coadd.py -param test_param.param -cubetype icubes.fits -out test_icubes_ca.fits
python3 ../scripts/cwi_moments.py test_coadd.fits -method positive -mode vel

#Clean up
rm test_coadd.mask.fits
rm test_coadd.M.fits
rm test_coadd.var.fits
rm test_wcs.wcs
rm *.wcs_cor.fits
rm *pNBtest*
rm test_icubes.crop.fits
rm test_icubes.rebin.fits
rm test_coadd.ps.fits
rm test_coadd.ps.bs.fits
rm test_icubes_ca.fits
rm *.vel*
rm *.dsp*
