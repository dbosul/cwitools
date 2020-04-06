"""Tools for kinematic calculations."""
import numpy as np

def first_moment(x, y, y_var=[], get_err=False, method='basic', m1_init=None,
window_size=25, window_min=10, window_step=1):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        y_var (np.array): Variance on y. Taken as var(y) if not provided.
        get_err (bool): Set to TRUE to return (moment, error) tuple
        method (str): The method to use for calculating the moment.
            'basic': Use all input x and y data.
            'clw': Use the closing-window method (O'Sullivan+20, FLASHES Survey)
        m1_init (float): Initial estimate for m1, required if using 'clw' method
            or restricting calculatio to a certain window size.
        window_size (float): The default window size for the moment calculation
            if using m1_init, or the initial window size of using 'clw' method.
        window_min (float): The minimum window size if using 'clw' method.
        window_step (float): The decrement in window size each iteration if
            using the 'clw' method.

    Returns:
        float: The first moment in x.
        float: (if get_err == True) The estimated error on the first moment

    """

    #Estimate variance if no variance given
    y_var = np.var(y) if y_var == [] else y_var

    #Basic and positive methods are mostly the same
    if method == 'basic':

        #If an initial value is given, use only the window around m1_init
        if m1_init != None:
            usex = np.abs(x - m1_init) <= window_size
            x = x[usex]
            y = y[usex]

        #Perform moment calculation as normal
        num = np.sum(x * y) #Numerator
        den = np.sum(y) #Denominator

        m1 = num / den
        m1_err = np.sqrt(np.sum(y_var * (den * x - num)**2 )) / den**2

    #If iterative closing-window method is selected
    elif method == 'clw':

        #Start guess at center of array of no initial value given
        m1 = x[int(len(x) / 2)] if m1_init == None else m1_init

        #Initialize window at maximum size
        window = window_size

        # Loop with decreasing window size until minimum is reached
        while window > window_min:

             #Get indices of values to use for this calculation
             usex = ( np.abs(x - m1) < window/2 ) & (y > 0)

             usex = x[use]
             usey = y[use]

             #Update moment calculation using current window
             num = np.sum(usex * usey) #Numerator
             den = np.sum(usey) #Denominator
             m1 = num / den
             m1_err = np.sqrt(np.sum(y_var * (den * x - num)**2 )) / den**2

             #Update window size
             window -= window_step

    if get_err:
        return m1, m1_err

    else:
        return m1


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
    m1num = np.sum(x * y) #Numerator of first moment calculation
    m1den = m2den = np.sum(y) #Denominator of first/second moment calculation

    m1 = m1num / m1den if m1 == None else m1 #Calculate if not given

    m2num = np.sum(np.power(x-m1, 2) * y) #Numerator of second moment calc.
    m2den = np.sum((x - m1) * y) #Term needed for eq.

    m2 = np.sqrt(m2num / m1den) #Second moment


    if not(get_err):
        return m2

    else:

        R = np.sum((x - m1)*y) #Term needed for eq.
        dm2_dIj = (m1den * x - m1num) / (m1den**2) #Another term needed

        #Estimate if no variance given
        y_var = np.var(y) if y_var == [] else y_var

        #Two squared terms that are multiplied by variance
        term1 = (1 / (2 * m2den * m2den * m2))**2
        term2 = (m2den * np.power(x - m1, 2) + 2 * m2den * dm2_dIj * R - m2num)**2

        #Error on second moment
        m2_err = np.sqrt( term1*np.sum(y_var*term2) )

        return m2, m2_err
