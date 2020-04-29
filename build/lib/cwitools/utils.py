"""Generic tools for saving files, etc."""
from astropy.io import fits
from cwitools import coordinates
import cwitools
import numpy as np
import os
import pkg_resources
import sys
import warnings

clist_template = {
    "INPUT_DIRECTORY":"./",
    "SEARCH_DEPTH":3,
    "OUTPUT_DIRECTORY":"./",
    "ID_LIST":[]
}

def get_arg_string(parser):
    """Construct a string displaying the arguments passed to argparse.

    Args:
        parser (argparse.ArgumentParser): The parser containing passed arguments

    Returns:
        string: A human-readable version of the passed arguments, for logging.
    """
    args_dict = vars(parser.parse_args())
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

def obj2binary(obj_mask, obj_id):
    """Get a binary mask of specific objects in a labelled object mask.

    Args:
        obj_mask (numpy.ndarray): Data cube containing labelled regions.
        obj_id (int or list): Object ID or list of object IDs to include.

    Returns:
        numpy.ndarray: The binary mask, where 1 = object, and 0 = background.

    """
    #Create 3D mask from object cube and IDs
    bin_cube = np.zeros_like(obj_mask, dtype=bool)
    if type(obj_id) == int:
        bin_cube = obj_mask == obj_id
    elif type(obj_id) == list and np.all(np.array(obj_id) == int):
        bin_cube = np.zeros_like(obj_mask, dtype=bool)
        for oid in obj_id:
            bin_cube[obj_mask == oid] = 1
    else:
        raise TypeError("obj_id must be an integer or list of integers.")
    return bin_cube

def get_instrument(hdu):
    if 'INSTRUME' in hdu.header:
        return hdu.header['INSTRUME']
    else:
        raise ValueError("Instrument not recognized.")

def get_specres(hdu):

    inst = get_instrument(hdu)

    if inst == 'PCWI':
        if 'MEDREZ' in hdu.header['GRATID']: return 2500
        else: return 5000

    elif inst == 'KCWI':

        grating, slicer = hdu.header['BGRATNAM'], hdu.header['IFUNAM']

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

def get_skylines(inst):

    if inst == 'PCWI':
        sky_file = 'palomar_lines.txt'
    elif inst == 'KCWI':
        sky_file = 'keck_lines.txt'
    else:
        raise ValueError("Instrument not recognized.")

    data_path = pkg_resources.resource_stream(__name__, 'data/sky/%s'% sky_file)
    data = np.loadtxt(data_path)

    return data

def get_skymask(hdr):
    """Get mask of sky lines for specific instrument/resolution."""
    wav_axis = coordinates.get_wav_axis(hdr)
    wav_mask = np.zeros_like(wav_axis, dtype=bool)
    inst = get_instrument(hdr)
    res = get_specres(hdr)
    sky_lines = get_skylines(inst)
    for line in sky_lines:
        dlam = line / res #Get width of line from inst res.
        wav_mask[np.abs(wav_axis - line) <= dlam] = 1
    return wav_mask

def extractHDU(fits_in):
    type_in = type(fits_in)
    if type_in == fits.HDUList:
        return fits_in[0]
    elif type_in == fits.ImageHDU or type_in == fits.PrimaryHDU:
        return fits_in
    else:
        raise ValueError("Astropy ImageHDU, PrimaryHDU or HDUList expected.")

def matchHDUType(fits_in, data, header):
    """Return a HDU or HDUList with data/header matching the type of the input."""
    type_in = type(fits_in)
    if type_in == fits.HDUList:
        return fits.HDUList([fits.PrimaryHDU(data, header)])
    elif type_in == fits.ImageHDU:
        return fits.ImageHDU(data, header)
    elif type_in == fits.PrimaryHDU:
        return fits.PrimaryHDU(data, header)
    else:
        raise ValueError("Astropy ImageHDU, PrimaryHDU or HDUList expected.")

def get_fits(data, header=None):
    hdu = fits.PrimaryHDU(data, header=header)
    hdulist = fits.HDUList([hdu])
    return hdulist

def set_cmdlog(path):
    cwitools.command_log = path


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

        line = line[:-1] #Trim new-line character
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


def diagnosticPcolor(data):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax  = plt.subplots(1, 1)
    ax.pcolor(data)
    #ax.contour(data)
    fig.show()
    plt.waitforbuttonpress()
    plt.close()
