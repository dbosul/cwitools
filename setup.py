import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
        name='cwitools',
        version='0.6',
        author="Donal O'Sullivan",
        author_email="dosulliv@caltech.edu",
        description="Analysis and Reduction Tools for PCWI/KCWI Data",
        long_description=long_description,
        url="https://github.com/dbosul/cwitools",
        download_url="https://github.com/dbosul/cwitools/archive/v0.5.tar.gz",
        packages=setuptools.find_packages(),
	    include_package_data=True,
        package_data={'': ['data/sky/*.txt']},
        entry_points = {
             'console_scripts': [
                 'cwi_applymask = cwitools.scripts.applymask:main',
                 'cwi_applywcs = cwitools.scripts.applywcs:main',
                 'cwi_bgsub = cwitools.scripts.bgsub:main',
                 'cwi_coadd = cwitools.scripts.coadd:main',
                 'cwi_crop = cwitools.scripts.crop:main',
                 'cwi_getmask = cwitools.scripts.getmask:main',
                 'cwi_getnb = cwitools.scripts.getnb:main',
                 'cwi_getvar = cwitools.scripts.getvar:main',
                 'cwi_initparams = cwitools.scripts.initparams:main',
                 'cwi_measurewcs = cwitools.scripts.measurewcs:main',
                 'cwi_moments = cwitools.scripts.moments:main',
                 'cwi_psfsub = cwitools.scripts.psfsub:main',
                 'cwi_rebin = cwitools.scripts.rebin:main'
             ]
        },
        install_requires = [
        'astropy',
        'argparse',
        'matplotlib',
        'numpy',
        'photutils',
        'scipy',
        'shapely',
        'tqdm',
        'PyAstronomy'
        ]
)
