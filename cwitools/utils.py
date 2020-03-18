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

    if logfile == None:
        if cwitools.command_log == None:
            warnings.warn("No cwitools.command_log is not set and no file was given.")
            return -1
        else:
            logfile = cwitools.command_log

    cmd_log = open(logfile, 'a')
    cmd_log.write(cmd_string + '\n')
    cmd_log.close()
    return 1
