from .. imports libs

from astropy.io import fits as fitsIO

import argparse


def run(paramPath,cubeType):
    """Auto-fills some of the values for a new CWITools parameter file.

    Args:
        paramPath (str): The path to the CWITools parameter file.
        cubeType (str): The type of cube to load for auto-filling (.e.g icubes.fits)

    """
    #Add file extension of omitted
    if not ".fits" in cubeType: cubeType += ".fits"

    #Check if any parameter values are missing (set to set-up mode if so)
    params = libs.params.loadparams(paramPath)

    #Get filenames
    files = libs.io.findfiles(params,cubeType)

    #Open custom FITS-3D objects
    fits = [fitsIO.open(f) for f in files]

    #1 - verify the parameter files
    libs.params.verify(params)

    #2 - re-initialize param values from FITS headers
    params = libs.params.parseHeaders(params,fits)

    #3 - overwrite param file
    params = libs.params.writeparams(params,paramPath)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Auto-fill certain values in parameter file using header data.')
    parser.add_argument('paramPath',
                        type=str,
                        help='Path to CWITools parameter file to be filled.'
    )
    parser.add_argument('cubeType',
                        type=str,
                        help='Type of PCWI/KCWI cube to load for filling parameters. ("e.g. icubes.fits, ocubes.fits") Should end in .fits'
    )
    args = parser.parse_args()

    run(args.paramPath,args.cubeType)
