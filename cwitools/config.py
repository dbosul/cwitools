"""Package wide settings and methods to adjust them"""

log_file = None #No default log
silent_mode = False #Default is to use stdout

log_file_backup = None
silent_mode_backup = None


def set_temp_output_mode(log=None, silent=None):
    """Backup global output settings and assign new values"""
    global log_file, log_file_backup, silent_mode, silent_mode_backup

    log_file_backup = log_file
    silent_mode_backup = silent_mode

    if log is not None:
        log_file = log
    if silent is not None:
        silent_mode = silent

def restore_output_mode():
    """Restore global output settings from backed-up values"""
    global log_file, log_file_backup, silent_mode, silent_mode_backup

    silent_mode = silent_mode_backup
    log_file = log_file_backup
