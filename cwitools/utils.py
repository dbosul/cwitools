"""Generic tools for saving files, etc."""

#Standard Imports
from datetime import datetime
import argparse
import os
import warnings

#Third-party Imports
from astropy.io import fits
from astropy import units as u
from PyAstronomy import pyasl
import numpy as np
import pkg_resources

#Local Imports
from cwitools import coordinates, config


def output_func_summary(func_name, local_vars_dict):
    """Print timestamp and summary of method parameters."""
    output(
        """\n\t{0}\n\t{1}:{2}""".format(
            datetime.now(),
            func_name,
            get_arg_string(local_vars_dict)
        )
    )

def get_arg_string(args):
    """Construct a string displaying the arguments passed to argparse.

    Args:
        parser (argparse.ArgumentParser): The parser containing passed arguments

    Returns:
        string: A human-readable version of the passed arguments, for logging.
    """
    if isinstance(args, argparse.Namespace):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        raise TypeError("args must be argparse.Namespace or dict object")

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
        if 'MEDREZ' in hdr['GRATID']:
            return 2000
        return 5000

    if inst == 'KCWI':

        grating, slicer = hdr['BGRATNAM'], hdr['IFUNAM']

        if grating == 'BL':
            r_0 = 900
        elif grating == 'BM':
            r_0 = 2000
        elif 'BH' in grating:
            r_0 = 4500
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

        return mul * r_0

    raise ValueError("Instrument not recognized.")

def get_neblines(wav_low=None, wav_high=None, redshift=0):
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
    data = np.genfromtxt(
        data_path,
        encoding='ascii',
        dtype=None,
        names=True,
        delimiter=','
    )
    data['ION'] = data['ION'].astype("<U16")
    data = list(zip(data['ION'], data['WAV']))

    data = np.array(data, dtype=[('ION', '<U16'), ('WAV', 'float')])
    #Update labels to be of the form 'LyA_1216' and wav to be observer frame
    for i, row in enumerate(data):
        ion, wav = row['ION'], row['WAV']
        label = "{0}_{1:.0f}".format(ion, wav)
        data['ION'][i] = label
        data['WAV'][i] *= (1 + redshift)

    if wav_low is not None:
        data = data[data['WAV'] > wav_low]

    if wav_high is not None:
        data = data[data['WAV'] < wav_high]

    return data

def get_nebmask(header, redshift=0, vel_window=500, mode='bmask'):
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

    for row in get_neblines(wav[0], wav[-1], redshift=redshift):

        #Calculate the lower/upper bounds on the emission line
        _, w_0 = row['ION'], row['WAV']
        wav_lo = w_0 * (1 - vel_window / 3.0e5)
        wav_hi = w_0 * (1 + vel_window / 3.0e5)

        #Append to list of tuples
        tuples.append((wav_lo, wav_hi))

        #Mask this line in the 1D mask
        ind_lo, ind_hi = coordinates.get_indices(wav_lo, wav_hi, header)
        binmask[ind_lo:ind_hi] = 1

    if mode == 'bmask':
        return binmask
    if mode == 'tuples':
        return tuples
    raise ValueError("Return mode not recognized. Must be 'bmask' or 'tuples'")

def get_skylines(inst, use_vacuum=False):
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
    skylines = get_skylines(inst, use_vacuum=use_vacuum)

    wav_mask = np.zeros_like(wav_axis, dtype=bool)
    linebounds = []
    for line in skylines:
        if linewidth is None:
            dlam = 2 * 1.4 * line / res #Get width of line from inst res.
        else:
            dlam = linewidth / 2.0
        wav_mask[np.abs(wav_axis - line) <= dlam] = 1
        linebounds.append((line - dlam, line + dlam))

    if mode == "bmask":
        return wav_mask
    if mode == "tuples":
        return linebounds
    raise ValueError("Mode not recognized. Must be 'bmask' or 'tuples'")

def extract_hdu(fits_in, nhdu=0):
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

def match_hdu_type(fits_in, data, header):
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
    if type_in == fits.ImageHDU:
        return fits.ImageHDU(data, header)
    if type_in == fits.PrimaryHDU:
        return fits.PrimaryHDU(data, header)
    raise ValueError("Astropy ImageHDU, PrimaryHDU or HDUList expected.")

def find_files(id_list, datadir, cubetype, depth=3):
    """Finds the input files given a CWITools parameter file and cube type.

    Args:
        id_list (list): A list of unique identifer sub-strings, such as an image number for PCWI
            data cubes (e.g. 34782) or a date_number string for KCWI data cubes (e.g. 190203_00072)
        datadir (str): The upper-level directory (or directories, separated by ';') to search.
        cubetype (str): The type of CWI data cube to load for each unique ID (i.e. the file
            extension, such as "icubes.fits"). Must end in ".fits"
        depth (int): The number of directory levels down from the given datadir to search for cubes

    Returns:
        list: List of file paths of input cubes.

    Raises:
        NotADirectoryError: If the input directory does not exist.

    """

    #Split into directories if multiple given in string
    if ';' in datadir:
        datadir = datadir.split(";")
    else:
        datadir = [datadir]

    #Load target cubes
    all_files = []
    target_files = []
    type_len = len(cubetype)

    for d_dir in datadir:

        #Check data directory exists
        if not os.path.isdir(d_dir):
            raise NotADirectoryError(datadir)

        for root, _, files in os.walk(d_dir):

            if root[-1] != '/':
                root += '/'

            rec = root.replace(d_dir, '').count("/")

            if rec > depth:
                continue

            for f_i in files:
                all_files.append(root + f_i)
                if f_i[-type_len:] != cubetype:
                    continue
                all_files.append(f_i)
                for id_i in id_list:
                    if id_i in f_i:
                        target_files.append(os.path.abspath(root + f_i))

    #Print file paths or file not found errors
    if len(target_files) < len(id_list):
        warn_str = ""
        for id_i in id_list:
            is_in = np.array([id_i in x for x in target_files])
            if not np.any(is_in):
                warn_str += "\t%s\n" % id_i
        if len(warn_str) > 0:
            output("\nWARNING:Files with type '%s' were not found for the following IDs:\n%s\n" %
                   (cubetype, warn_str)
                   )

    return sorted(target_files)

def parse_cubelist(filepath):
    """Load a CWITools parameter file into a dictionary structure.

    Args:
        path (str): Path to CWITools .list file

    Returns:
        dict: Python dictionary containing the relevant fields and information.

    """
    clist = {
        "DATA_DIRECTORY":"./",
        "SEARCH_DEPTH":3,
        "ID_LIST":[]
    }

    #Parse file
    listfile = open(filepath, 'r')
    for line in listfile:

        line = line.strip() #Trim new-line character
        line = line.split("#")[0] #Remove comments

        #Skip empty lines
        if line == "":
            continue

        #Add IDs when indicated by >
        if line[0] == '>':
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
    if ';' in clist["DATA_DIRECTORY"]:
        for d_dir in clist["DATA_DIRECTORY"].split(';'):
            if not os.path.isdir(d_dir):
                output("\nWARNING: %s is not a directory.\n" % d_dir)
    elif not os.path.isdir(clist["DATA_DIRECTORY"]):
        warnings.warn("%s is not a directory." % clist["DATA_DIRECTORY"])

    try:
        clist["SEARCH_DEPTH"] = int(clist["SEARCH_DEPTH"])
    except:
        raise ValueError("Could not parse SEARCH_DEPTH to int (%s)" % clist["SEARCH_DEPTH"])
    #Return the dictionary
    return clist

def output(str_in, log=None, silent=None):
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
    if log is not None:
        logfilename = log

    #Second priority, take global log file
    elif config.log_file is not None:
        logfilename = config.log_file

    #If neither log set, ignore
    else:
        uselog = False

    #If silent is actively set to False by function call
    if silent is not None and not silent:
        print(str_in, end='')

    #If silent is not set, but global 'silent_mode' is False
    elif silent is None and not config.silent_mode:
        print(str_in, end='')

    else: pass

    if uselog:
        logfile = open(logfilename, 'a')
        logfile.write(str_in)
        logfile.close()
