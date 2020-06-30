"""Generic tools for saving files, etc."""
from astropy.io import fits
from astropy import units as u
from astropy import wcs
from cwitools import coordinates
from PyAstronomy import pyasl
from scipy import ndimage

import cwitools
import numpy as np
import os
import pkg_resources
import sys
import warnings

import matplotlib.pyplot as plt


clist_template = {
    "INPUT_DIRECTORY":"./",
    "SEARCH_DEPTH":3,
    "OUTPUT_DIRECTORY":"./",
    "ID_LIST":[]
}

def get_arg_string(args):
    """Construct a string displaying the arguments passed to argparse.

    Args:
        parser (argparse.ArgumentParser): The parser containing passed arguments

    Returns:
        string: A human-readable version of the passed arguments, for logging.
    """
    args_dict = vars(args)
    info_string = "\n"
    for key, value in args_dict.items():
        info_string += "\t\t{0} = {1}\n".format(key, value)
    return info_string

def get_cmd(sys_argv):
    """Re-construct the command issued from sys.argv array.

    Args:
        sys_argv (list): The value of sys.argv in a given script.

    Returns
        string: The Python3 command as issued.
    """
    #Get command that was issued
    argv_string = " ".join(sys_argv)
    cmd_string = "python3 " + argv_string + "\n"
    return cmd_string

def get_instrument(hdr):
    """Get the instrument ('PCWI' or 'KCWI') based on the header.

    Args:
        hdr (astropy.io.fits.hdu.header): The FITS header

    Returns:
        str: "PCWI" or "KCWI"

    Raises:
        ValueError: If the keyword 'INSTRUME' is not found in the header.

    """
    if 'INSTRUME' in hdr.keys():
        return hdr['INSTRUME']
    else:
        raise ValueError("Instrument not recognized.")

def get_specres(hdr):
    """Get the approximate spectral resolution based on the header

    The returned values for various KCWI settings are taken from the estimates
    on KCWI's home page at keck.edu. PCWI returns R = 2000 for the Richardson
    grating and R = 5000 for all others.

    Args:
        hdr (astropy FITS Header): The header

    Returns:
        float: The spectral resolution, R for the grating/slicer used.

    Raises:
        ValueError: If grating or slicer name is not recognized
    """
    inst = get_instrument(hdr)

    if inst == 'PCWI':
        if 'MEDREZ' in hdr['GRATID']: return 2000
        else: return 5000

    elif inst == 'KCWI':

        grating, slicer = hdr['BGRATNAM'], hdr['IFUNAM']

        if grating == 'BL':
            R0 = 900
        elif grating == 'BM':
            R0 = 2000
        elif 'BH' in grating:
            R0 = 4500
        else:
            raise ValueError("Grating not recognized (header:BGRATNAM)")

        if slicer == 'Small':
            mul = 4
        elif slicer == 'Medium':
            mul = 2
        elif slicer == 'Large':
            mul = 1
        else:
            raise ValueError("Slicer not recognized (header:IFUNAM)")

        return mul * R0

    else:
        raise ValueError("Instrument not recognized.")

def get_neblines(wav_low=None, wav_high=None, z=0):
    """Return a list of sky lines for PCWI or KCWI

    This method uses a list of common galaxy emission lines compiled by
    Drew Chojnowski, available publicly on his website at
    http://astronomy.nmsu.edu/drewski/tableofemissionlines.html

    Args:
        wav_low (float): The lower end of the observer frame bandpass to
            consider, in Angstrom. Optional.
        wav_high (float): The upper end of the observer frame bandpass to
            consider, in Angstrom. Optional.
        z (float): The redshift of the source, to apply cosmological redshift to
            the rest-frame values.

    Returns:
        numpy.ndarray: An array with labelled columns 'ION' and 'WAV'. 'ION'
            contains a label of the form <Name>_<Restframe> (e.g. LyA_1216)
            and 'WAV' contains the observed wavelength at redshift z.

    """
    rel_path = 'data/gal_lines/drewchojnowski_geldata.csv'
    data_path = pkg_resources.resource_stream(__name__, rel_path)
    data = np.genfromtxt(data_path,
        encoding='ascii',
        dtype=None,
        names=True,
        delimiter=','
    )
    data['ION'] = data['ION'].astype("<U16")
    data = [x for x in zip(data['ION'], data['WAV'])]
    data = np.array(data, dtype=[('ION', '<U16'), ('WAV', 'float')])
    #Update labels to be of the form 'LyA_1216' and wav to be observer frame
    for i, row in enumerate(data):
        ion, wav = row['ION'], row['WAV']
        label = "{0}_{1:.0f}".format(ion, wav)
        data['ION'][i] = label
        data['WAV'][i] *= (1 + z)

    if wav_low is not None:
        data = data[data['WAV'] > wav_low]

    if wav_high is not None:
        data = data[data['WAV'] < wav_high]

    return data

def get_nebmask(header, z=0, vel_window=500, use_vacuum=None, mode='bmask'):
    """Get mask of nebular emission lines for a specific FITS image and redshift.

    Args:
        header (astropy FITS Header): Header for 3D (z, y, x) or 1D (z) data.
        z (float): The redshift of the emission to consider.
        vel_window (float): The velocity width of each line, in km/s.
        use_vacuum (bool): Set to True to convert to vacuum wavelengths.
        mode (str): Mode of the return type.
            'bmask' - return a 1D array where 1 = masked and 0 = unmasked
            'tuples' - return a list of tuples indicating the (lower, upper)
                wavelength for each line, based on the velocity width.

    Returns:
        numpy.array: A 1D binary mask. Values of 1 indicate masked wavelengths.
        OR
        list of float tuples: (lower, upper) bounds for each emission line.

    """
    wav = coordinates.get_wav_axis(header)
    binmask = np.zeros_like(wav)
    tuples = []

    for row in get_neblines(wav[0], wav[-1], z=z):

        #Calculate the lower/upper bounds on the emission line
        label, w0 = row['ION'], row['WAV']
        wav_lo = w0 * (1 - vel_window / 3.0e5)
        wav_hi = w0 * (1 + vel_window / 3.0e5)

        #Append to list of tuples
        tuples.append((wav_lo, wav_hi))

        #Mask this line in the 1D mask
        ind_lo, ind_hi = coordinates.get_indices(wav_lo, wav_hi, header)
        binmask[ind_lo:ind_hi] = 1

    ### DEBUG/TEST
    if 0:
        fig, ax = plt.subplots(1, 1)
        ax.plot(wav, binmask, 'k-')
        fig.show()
        input("")
        plt.close()

    if mode == 'bmask':
        return binmask
    elif mode == 'tuples':
        return tuples
    else:
        raise ValueError("Return mode not recognized. Must be 'bmask' or 'tuples'")

def get_skylines(inst, use_vacuum=False, mode='centers'):
    """Return a list of sky lines for PCWI or KCWI

    This is based on a list of known sky-lines built into CWITools, which is
    currently a work in progress. It is not an exhaustive list by any means.

    Args:
        inst (str): 'KCWI' or 'PCWI' - to determine whether Keck or Palomar sky
            is requested.
        use_vacuum (bool): Set to True to convert the sky lines to vacuum
            wavelengths.

    Returns:
        numpy.array: An array of the sky lines in units of Angstrom.
    """
    if inst == 'PCWI':
        sky_file = 'palomar_lines.txt'
    elif inst == 'KCWI':
        sky_file = 'keck_lines.txt'
    else:
        raise ValueError("Instrument not recognized.")

    data_path = pkg_resources.resource_stream(__name__, 'data/sky/%s'% sky_file)
    data = np.loadtxt(data_path)

    if use_vacuum:
        data = pyasl.airtovac2(data)

    return data

def get_skymask(hdr, use_vacuum=None, linewidth=None, mode='bmask'):
    """Get mask of sky lines for specific instrument/resolution.

    This is based on a list of known sky-lines built into CWITools, which is
    currently a work in progress. It is not an exhaustive list by any means.

    Args:
        inst (str): 'KCWI' or 'PCWI' - to determine whether Keck or Palomar sky
            is requested.
        use_vacuum (bool): Set to True to convert the sky lines to vacuum
            wavelengths. Default is True if 'CTYPE3' is WAVE, False if 'AWAV' or
            otherwise.
        linewidth (float): The width (in Angstrom) of the mask for each line.
            If none provided, it will be based on the spectral resolution of
            the setting used, which is determined from the header.
        mode (str): The mode of the return type - 'binmask' or 'tuples'
            'bmask' - return a 1D array where 1 = masked and 0 = unmasked
            'tuples' - return a list of tuples indicating the (lower, upper)
                wavelength for each line, based on the linewidth.
    Returns:
        numpy.array: A 1D binary mask. Values of 1 indicate masked wavelengths.
        OR
        list of float tuples: (lower, upper) bounds for each emission line.

    """

    #Assign default if not assigned by user
    if use_vacuum is None:
        use_vacuum = hdr['CTYPE3'] == 'WAVE'
        if hdr['CTYPE3'] not in ['AWAV', 'WAVE']:
            warnings.warn("CTYPE3 ({0}) not recognized.".format(hdr['CTYPE3']))

    wav_axis = coordinates.get_wav_axis(hdr)

    inst = get_instrument(hdr)
    res = get_specres(hdr)
    skylines = get_skylines(inst, use_vacuum = use_vacuum)

    wav_mask = np.zeros_like(wav_axis, dtype = bool)
    linebounds = []
    for line in skylines:
        dlam = 1.4 * line / res #Get width of line from inst res.
        wav_mask[np.abs(wav_axis - line) <= dlam] = 1
        linebounds.append((line - dlam, line + dlam))

    if mode == "bmask":
        return wav_mask
    elif mode == "tuples":
        return linebounds
    else:
        raise ValueError("Mode not recognized. Must be 'bmask' or 'tuples'")

def bunit_todict(st):
    """Convert BUNIT string to a dictionary

    Args:
        st (str): The input BUNIT string

    Returns:
        dict: The output dictionary of {unit:power} e.g. cm^2 = {'cm':2}

    """
    numchar = [str(i) for i in range(10)]
    numchar.append('+')
    numchar.append('-')
    dictout = {}

    st_list = st.split()
    for st_element in st_list:
        flag = 0
        for i, char in enumerate(st_element):
            if char in numchar:
                flag = 1
                break

        if i == 0:
            key = st_element
            power_st='1'
        elif flag == 0:
            key = st_element
            power_st = '1'
        else:
            key = st_element[0:i]
            power_st = st_element[i:]

        dictout[key] = float(power_st)

    return dictout

def get_bunit(hdr):
    """"Get BUNIT string that meets FITS standard."""
    bunit = multiply_bunit(hdr['BUNIT'])
    return bunit

def multiply_bunit(bunit, multiplier='1'):
    """Unit conversions and multiplications.

    Docstring TBC.

    """

    # Electrons
    electron_power = 0.
    if 'electrons' in bunit:
        bunit = bunit.replace('electrons', '1')
        electron_power = 1
    if 'variance' in bunit:
        bunit=bunit.replace('variance', '1')
        electron_power = 2

    # Angstrom
    if '/A' in bunit:
        bunit=bunit.replace('/A', '/angstrom')

    # unconventional expressions
    if 'FLAM' in bunit:
        addpower = 1
        if '**2' in bunit:
            addpower = 2
            bunit = bunit.replace('**2', '')
        power = float(bunit.replace('FLAM', ''))
        v0 = u.erg / u.s / u.cm**2 / u.angstrom * 10**(-power)
        v0 = v0**addpower

    elif 'SB' in bunit:
        addpower = 1
        if '**2' in bunit:
            addpower = 2
            bunit = bunit.replace('**2','')
        power = float(bunit.replace('SB',''))
        v0 = u.erg / u.s / u.cm**2 / u.angstrom / u.arcsec**2*10**(-power)
        v0 = v0**addpower
    else:
        v0 = u.Unit(bunit)

    if type(multiplier) == type(''):
        if 'A' in multiplier:
            multiplier = multiplier.replace('A','angstrom')
        multi = u.Unit(multiplier)
    else:
        multi = multiplier

    vout=(v0 * multi)

    # Convert to quantity
    if type(vout) == type(u.Unit('erg/s')):
        vout = u.Quantity(1, vout)
    vout = vout.cgs

    stout = "{0.value:.0e} {0.unit:FITS}".format(vout)
    stout = stout.replace('1e+00 ','')
    stout = stout.replace('10**','1e')
    dictout = bunit_todict(stout)

    # clean up
    if 'rad' in dictout:
        vout = (vout * u.arcsec**(-dictout['rad'])).cgs * u.arcsec**dictout['rad']
        stout = "{0.value:.0e} {0.unit:FITS}".format(vout)
        dictout = bunit_todict(stout)

    if 'Ba' in dictout:
        vout = vout * (u.Ba**(-dictout['Ba'])) * (u.erg / u.cm**3)**dictout['Ba']
        stout = "{0.value:.0e} {0.unit:FITS}".format(vout)
        dictout = bunit_todict(stout)

    if 'g' in dictout:
        vout = vout * (u.g**(-dictout['g'])) * (u.erg * u.s**2 / u.cm**2)**dictout['g']
        stout = "{0.value:.0e} {0.unit:FITS}".format(vout)
        dictout = bunit_todict(stout)

    # electrons
    if electron_power > 0:
        stout = stout +' electrons'+'{0:.0f}'.format(electron_power)+' '
        dictout = bunit_todict(stout)

    # sort
    def unit_key(st):
        if st[0] in [str(i) for i in np.arange(10)]:
            return 0
        elif 'erg' in st:
            return 1
        elif 'electrons' in st:
            return 1
        elif st[0]=='s':
            return 2
        elif 'cm' in st:
            return 3
        elif 'arcsec' in st:
            return 4
        else:
            return 5

    st_list = stout.split()
    st_list.sort(key = unit_key)
    stout = ' '.join(st_list)

    return stout

def extractHDU(fits_in, nhdu=0):
    """Load a HDU whether the input type is HDUList, PrimaryHDU or ImageHDU.

    Args:
        fits_in: An astropy.fits.HDUlist, .ImageHDU or .PrimaryHDU or a string
            which is the path of a FITS file.
        nhdu (int): Which HDU to extract, if HDUList type or file given. Default
            is 0 (first HDU).
    Returns:
        HDU: The input HDU or the Primary HDU of an input HDUList or FITS file.

    """
    type_in = type(fits_in)
    if type_in is str and os.path.isfile(fits_in):
        try:
            return fits.open(fits_in)[nhdu]
        except:
            raise RuntimeError("Error opening file: {0}".format(fits_in))
    elif type_in is fits.HDUList:
        return fits_in[nhdu]
    elif type_in is fits.ImageHDU or type_in is fits.PrimaryHDU:
        return fits_in
    else:
        raise ValueError("Astropy ImageHDU, PrimaryHDU, HDUList or path to FITS file expected.")

def matchHDUType(fits_in, data, header):
    """Return a HDU or HDUList with data/header matching the type of the input

    Args:
        fits_in (HDU or HDUList): An astropy ImageHDU, PrimaryHDU or HDUList.
        data (numpy.ndarray): The data for the new object
        header (astropy FITS Header): The header for the new object

    Returns:
        HDU/HDUList: An object of the same type as the input, with the new data
            and header.
    """
    type_in = type(fits_in)
    if type_in == fits.HDUList:
        return fits.HDUList([fits.PrimaryHDU(data, header)])
    elif type_in == fits.ImageHDU:
        return fits.ImageHDU(data, header)
    elif type_in == fits.PrimaryHDU:
        return fits.PrimaryHDU(data, header)
    else:
        raise ValueError("Astropy ImageHDU, PrimaryHDU or HDUList expected.")

def find_files(id_list, datadir, cubetype, depth=3):
    """Finds the input files given a CWITools parameter file and cube type.

    Args:
        params (dict): CWITools parameters dictionary.
        cubetype (str): Type of cube (e.g. icubes.fits) to load.

    Returns:
        list(string): List of file paths of input cubes.

    Raises:
        NotADirectoryError: If the input directory does not exist.

    """

    #Check data directory exists
    if not os.path.isdir(datadir):
        raise NotADirectoryError("Data directory (%s) does not exist. Please correct and try again." % datadir)

    #Load target cubes
    N_files = len(id_list)
    target_files = []
    typeLen = len(cubetype)

    if depth != 0:
        for root, dirs, files in os.walk(datadir):

            if root[-1] != '/': root += '/'
            rec = root.replace(datadir, '').count("/")

            if rec > depth: continue
            else:
                for f in files:
                    if f[-typeLen:] == cubetype:
                        for i,ID in enumerate(id_list):
                            if ID in f:
                                target_files.append(root + f)
    else:
        # Using absolute path
        root = datadir
        if root[-1] != '/':
            root += '/'
        for id in id_list:
            path = root + id + '_' + cubetype
            if os.path.isfile(path):
                target_files.append(path)

    #Print file paths or file not found errors
    if len(target_files) < len(id_list):
        warnings.warn("Some files were not found:")
        for id in id_list:
            is_in = np.array([ id in x for x in target_files])
            if not np.any(is_in):
                warnings.warn("Image with ID %s and type %s not found." % (id, cubetype))


    return sorted(target_files)

def parse_cubelist(filepath):
    """Load a CWITools parameter file into a dictionary structure.

    Args:
        path (str): Path to CWITools .list file

    Returns:
        dict: Python dictionary containing the relevant fields and information.

    """
    global clist_template
    clist = {k:v for k, v in clist_template.items()}

    #Parse file
    listfile = open(filepath, 'r')
    for line in listfile:

        line = line.strip() #Trim new-line character
        #Skip empty lines
        if line == "":
            continue

        #Add IDs when indicated by >
        elif line[0] == '>':
            clist["ID_LIST"].append(line.replace('>', ''))

        elif '=' in line:

            line = line.replace(' ', '')     #Remove white spaces
            line = line.replace('\n', '')    #Remove line ending
            line = line.split('#')[0]        #Remove any comments
            key, val = line.split('=') #Split into key, value pair
            if key.upper() in clist:
                clist[key] = val
            else:
                raise ValueError("Unrecognized cube list field: %s" % key)
    listfile.close()

    #Perform quick validation of input, but only warn for issues
    input_isdir = os.path.isdir(clist["INPUT_DIRECTORY"])
    if not input_isdir:
        warnings.warn("%s is not a directory." % clist["INPUT_DIRECTORY"])

    output_isdir = os.path.isdir(clist["OUTPUT_DIRECTORY"])
    if not output_isdir:
        warnings.warn("%s is not a directory." % clist["OUTPUT_DIRECTORY"])

    try:
        clist["SEARCH_DEPTH"] = int(clist["SEARCH_DEPTH"])
    except:
        raise ValueError("Could not parse SEARCH_DEPTH to int (%s)" % clist["SEARCH_DEPTH"])
    #Return the dictionary
    return clist

def output(str, log=None, silent=None):
    """Generic output handler for internal use in CWITools.

    Args:
        str (str): The string to output.
        log (str): The log file to write to, if any
        silent (bool): Set to True to suppress standard output and only write to
            log file.

    Returns:
        None
    """
    uselog = True

    #First priority, take given log
    if log != None:
        logfilename = log

    #Second priority, take global log file
    elif cwitools.log_file != None:
        logfilename = cwitools.log_file

    #If neither log set, ignore
    else:
        uselog = False

    #If silent is actively set to False by function call
    if silent == False:
        print(str, end='')

    #If silent is not set, but global 'silent_mode' is False
    elif silent == None and cwitools.silent_mode == False:
        print(str, end='')

    else: pass

    if uselog:
        logfile = open(logfilename, 'a')
        logfile.write(str)
        logfile.close()

        