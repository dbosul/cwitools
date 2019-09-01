import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
        name='cwitools',
        version='0.1dev10',
        author="Donal O'Sullivan",
        author_email="dosulliv@caltech.edu",
        description="Analysis and Reduction Tools for PCWI/KCWI Data",
        long_description=long_description,
        url="https://github.com/dbosul/cwitools",
        download_url="https://github.com/dbosul/cwitools/archive/v0.1dev10.tar.gz",
        packages=setuptools.find_packages(),
        entry_points = {
             'console_scripts': [
                 'cwi_applymask = cwitools.scripts.cwi_applymask:main',
                 'cwi_bgsub = cwitools.scripts.cwi_bgsub:main',
                 'cwi_coadd = cwitools.scripts.cwi_coadd:main',
                 'cwi_crop = cwitools.scripts.cwi_crop:main',
                 'cwi_fixwcs = cwitools.scripts.cwi_fixwcs:main',
                 'cwi_getvar = cwitools.scripts.cwi_getvar:main',
                 'cwi_psfsub = cwitools.scripts.cwi_psfsub:main',
                 'cwi_rebin = cwitools.scripts.cwi_rebin:main'
             ]
        }
)
