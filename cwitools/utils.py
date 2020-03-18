"""Generic tools for saving files, etc."""
import cwitools
from astropy.io import fits
import warnings

def get_fits(data, header=None):
    hdu = fits.PrimaryHDU(data, header=header)
    hdulist = fits.HDUList([hdu])
    return hdulist

def set_cmdlog(path):
    cwitools.command_log = path

def log_command(sys_argv, logfile=None):
    """Append a terminal command reflected by sys.argv to the log file

    Args:
        sys_argv: The sys.argv attribute from the script that was called.

    Returns:
        int: 1 if successfully written to file, -1 if failed.
    """
    cmd_string = " ".join(sys.argv)

    #If no log-file given, warn user and return bad flag
    if logfile == None:
        warnings.warn("Command will not be saved. No log file given.")
        return -1

    #Let user know if new file is being made (can help avoid typos)
    if not(os.path.isfile(logfile)):
        warnings.warn("%s does not exist. File will be created." % logfile)

    try:
        cmd_log = open(logfile, 'a')
        cmd_log.write(cmd_string + '\n')
        cmd_log.close()

    except:
        raise ValueError("Error opening/writing to log file: %s" % logfile)

    return 1
