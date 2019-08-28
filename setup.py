import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
        name='CWITools',
        version='0.1',
        author="Donal O'Sullivan",
        author_email="dosulliv@caltech.edu",
        description="Analysis and Reduction for CWI Data",
        long_description=long_description,
        url="https://github.com/dbosul/cwitools",
        packages=setuptools.find_packages(),
        # entry_points = {
        #     'console_scripts': [
        #         'cwi_test01 = cwitools.analysis.test:main'
        #     ]
        # }
)
