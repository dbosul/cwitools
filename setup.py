import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cwitools',
    version='0.8.4',
    author="Donal O'Sullivan",
    author_email="dosulliv@caltech.edu",
    description="Analysis and Reduction Tools for PCWI/KCWI Data",
    long_description=long_description,
    url="https://github.com/dbosul/cwitools",
    download_url="https://github.com/dbosul/cwitools/archive/v0.8.2.tar.gz",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['data/sky/*.txt', 'data/gal_lines/*.csv']},
    entry_points={
        'console_scripts': [
            'cwi_apply_mask = cwitools.scripts.apply_mask:main',
            'cwi_apply_wcs = cwitools.scripts.apply_wcs:main',
            'cwi_asmooth = cwitools.scripts.asmooth:main',
            'cwi_bg_sub = cwitools.scripts.bg_sub:main',
            'cwi_coadd = cwitools.scripts.coadd:main',
            'cwi_crop = cwitools.scripts.crop:main',
            'cwi_fit_covar = cwitools.scripts.fit_covar:main',
            'cwi_get_mask = cwitools.scripts.get_mask:main',
            'cwi_get_nb = cwitools.scripts.get_nb:main',
            'cwi_get_rprof = cwitools.scripts.get_rprof:main',
            'cwi_get_var = cwitools.scripts.get_var:main',
            'cwi_get_wl = cwitools.scripts.get_wl:main',
            'cwi_mask_z = cwitools.scripts.mask_z:main',
            'cwi_measure_wcs = cwitools.scripts.measure_wcs:main',
            'cwi_obj_lum = cwitools.scripts.obj_lum:main',
            'cwi_obj_morpho = cwitools.scripts.obj_morpho:main',
            'cwi_obj_sb = cwitools.scripts.obj_sb:main',
            'cwi_obj_spec = cwitools.scripts.obj_spec:main',
            'cwi_obj_zfit = cwitools.scripts.obj_zfit:main',
            'cwi_obj_zmoments = cwitools.scripts.obj_zmoments:main',
            'cwi_psf_sub = cwitools.scripts.psf_sub:main',
            'cwi_rebin = cwitools.scripts.rebin:main',
            'cwi_scale_var = cwitools.scripts.scale_var:main',
            'cwi_segment = cwitools.scripts.segment:main',
            'cwi_slice_corr = cwitools.scripts.slice_corr:main'
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
