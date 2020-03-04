"""Tools for kinematic calculations."""
import numpy as np

def first_moment(x, y, y_var=[], get_err=False):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        y_var (np.array): Variance on y. Taken as var(y) if not provided.
        get_err (bool): Set to TRUE to return (moment, error) tuple

    Returns:
        float: The first moment in x.
        float: The error on the first moment (if get_err == True)

    """
    m1 = np.sum(x * y) / np.sum(y)
    if not get_err:
        return m1

    else:
        m1_err = first_moment_err(x, y, y_var=y_var)
        return m1, m1_err


def second_moment(x, y, m1=None, y_var=[], get_err=False):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        m1 (float): The first moment. Calculated if not provided.
        y_var (np.array): Variance on y. Taken as var(y) if not provided.
        get_err (bool): Set to TRUE to return (moment, error), tuple

    Returns:
        float: The second moment in x.
        float: The error on the second moment (if get_err == True)

    """
    if m1 == None:
        m1 = first_moment(x, y)

    m2 = np.sqrt(np.sum(y * (x - m1)**2 ) / np.sum(y))

    if not(get_err):
        return m2

    else:

        if m1 == None: m1 = first_moment(x, y)

        m2_err = second_moment_err(x, y, m1, y_var=y_var)
        return m2, m2_err

def first_moment_err(x, y, y_var=[]):
    """Calculate propagated error on the first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        y_var (np.array): Variance on y. Taken as var(y) if not provided.

    Returns:
        float: The error on the first moment

    """
    A = np.sum(x * y) #Numerator of moments calculation
    B = np.sum(y) #Denominator of moments calculation

    if y_var == []: y_var = np.var(y) #Estimate if no variance given

    return  np.sqrt(np.sum(y_var * (B * x - A)**2 )) / B**2

def second_moment_err(x, y, m1=None, y_var=[]):
    """Calculate propagated error on the second moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        m1 (float): The value of the first moment
        y_var (np.array): Variance on y. Taken as var(y) if not provided.

    Returns:
        float: The error on the second moment

    """
    if m1 == None: m1 = first_moment(x, y)

    A = np.sum( x*y ) #Numerator of first moment calculation
    B = np.sum( y ) #Denominator of first/second moment calculation
    C = np.sum( np.power(x-m1, 2)*y ) #Numerator of second moment calc.
    R = np.sum( (x - m1)*y ) #Term needed for eq.
    dm2_dIj = (B*x - A)/(B**2) #Another term needed
    m2 = np.sqrt(C/B) #Second moment

    if y_var == []: y_var = np.var(y) #Estimate if no variance given

    #Two squared terms that are multiplied by variance
    term1 = (1/(2*B*B*m2))**2
    term2 = ( B*np.power(x - m1, 2) + 2*B*dm2_dIj*R - C )**2

    return np.sqrt( term1*np.sum(y_var*term2) )

#Basic moments calculatio
def basic_moments(x, y, y_var=[], pos_thresh=False, m1_init=None, window=30,
get_err=True):
    """Calculate first and second moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        pos_thresh (bool): Set to TRUE to exclude negative weights.
        m1_init (float): Initial guess for the first moment value
        window (float): Size of window, centered on m1_init, to use

    Returns:
        float: The first moment in x.
        float: The second moment in x.

    """

    if m1_init != None:
        usex = np.abs(x-m1_init) <= window/2
        x = x[usex]
        y = y[usex]
        if y_var!=[]: y_var = y_var[usex]

    #Points with y=0 add noise without influencing result - so remove
    x = x[y != 0]
    if y_var!=[]: y_var = y_var[ y!= 0]
    y = y[y != 0]

    if pos_thresh:
        x = x[y > 0]
        if y_var!=[]: y_var = y_var[ y > 0]
        y = y[y > 0]

    m1 = first_moment(x, y)
    m2 = second_moment(x, y, m1)

    if get_err:

        m1_err = first_moment_err(x, y, y_var)
        m2_err = second_moment_err(x, y, m1, y_var)

        return m1, m2, m1_err, m2_err

    else:

        return m1, m2

#Convergent method for moments calculation
def closing_window_moments(x, y, m1_init = None, window_max=25, window_min=15,
     window_step_size=1, y_var=[], get_err=True):
    """Calculate first and second moments using the 'closing-window method' (O'Sullivan et al. 2020).

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        m1_init (float): Initial guess for first moment, used to center window.
            If none given, center value of x will be used.
        window_max (float): Starting window size for calculation (in same unit as x)
        window_min (float): Minimum window size for calculation. (same units as x)
        window_step_size (float): Decrease in window size for each step (same units as x).

    Returns:
        float: The first moment in x.
        float: Error on the final first moment calculation (if get_err is True)
        float: The second moment in x
        float: Error on the final second moment calculation (if get_err is True)

    """
    #Take user input as initial guess or use center of x-range
    if m1_init == None: m1 = x[int(len(x)/2)]
    else: m1 = m1_init

    #Initialize window at maximum size
    window = window_max

    # Loop over window size
    while window > window_min:

        #Get indices of values to use for this calculation
        use = ( np.abs(x - m1) < window/2 ) & (y > 0)

        usex = x[use]
        usey = y[use]
        usevar = y_var if y_var==[] else y_var[use]

        #Calculate moments (i.e. update window center)
        m1 = first_moment(usex, usey)
        m2 = second_moment(usex, usey, m1 )


        #Update window size
        window -= window_step_size

    if get_err:

        m1_err = first_moment_err(usex, usey, usevar)
        m2_err = second_moment_err(usex, usey, m1, usevar)

        return m1, m2, m1_err, m2_err

    else:

        return m1, m2
