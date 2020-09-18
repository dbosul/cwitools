import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cwitools',
    version='0.8',
    author="Donal O'Sullivan",
    author_email="dosulliv@caltech.edu",
    description="Analysis and Reduction Tools for PCWI/KCWI Data",
    long_description=long_description,
    url="https://github.com/dbosul/cwitools",
    download_url="https://github.com/dbosul/cwitools/archive/v0.8.tar.gz",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['data/sky/*.txt', 'data/gal_lines/*.csv']},
    entry_points={
        'console_scripts': [
            'cwi_apply_mask = cwitools.scripts.apply_mask:apply_mask',
            'cwi_apply_wcs = cwitools.scripts.apply_wcs:apply_wcs',
            'cwi_bg_sub = cwitools.scripts.bg_sub:bg_sub',
            'cwi_coadd = cwitools.scripts.coadd:coadd',
            'cwi_crop = cwitools.scripts.crop:crop',
            'cwi_fit_covar = cwitools.scripts.fit_covar:fit_covar',
            'cwi_get_mask = cwitools.scripts.get_mask:get_mask',
            'cwi_get_nb = cwitools.scripts.get_nb:get_nb',
            'cwi_get_rprof = cwitools.scripts.get_nb:get_rprof',
            'cwi_get_var = cwitools.scripts.get_var:get_var',
            'cwi_get_wl = cwitools.scripts.get_wl:get_wl',
            'cwi_mask_z = cwitools.scripts.mask_z:mask_z',
            'cwi_measure_wcs = cwitools.scripts.measure_wcs:measure_wcs',
            'cwi_obj_lum = cwitools.scripts.obj_lum:obj_lum',
            'cwi_obj_morpho = cwitools.scripts.obj_morpho:obj_morpho',
            'cwi_obj_sb = cwitools.scripts.obj_sb:obj_sb',
            'cwi_obj_spec = cwitools.scripts.obj_spec:obj_spec',
            'cwi_obj_zmoments = cwitools.scripts.obj_zmoments:obj_zmoments',
            'cwi_psf_sub = cwitools.scripts.psf_sub:psf_sub',
            'cwi_rebin = cwitools.scripts.rebin:rebin',
            'cwi_scale_var = cwitools.scripts.scale_var:scale_var',
            'cwi_segment = cwitools.scripts.segment:segment',
            'cwi_slice_corr = cwitools.scripts.slice_corr:slice_corr'
            ]
        },
    install_requires=[
        'astropy',
        'argparse',
        'matplotlib',
        'numpy',
        'photutils',
        'scipy',
        'shapely',
        'tqdm',
        'PyAstronomy',
        'pyregion',
        'reproject',
        'scikit-image'
        ]
)
