"""CWITools Science Functions Library.

This module contains functions for use in scientific analysis or the generation
of scientific products from data cubes (such as pseudo-NarrowBand images).

"""
from cwitools.libs import cubes

from astropy.io import fits
from astropy.constants import G
from astropy.cosmology import WMAP9 as cosmo
from astropy.modeling import models,fitting
#from astropy.modeling.functional_models import Moffat2D

from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales as getPxScales
from scipy.ndimage import gaussian_filter as GaussND
from scipy.stats import sigmaclip
from scipy.optimize import differential_evolution

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pyregion

def first_moment(x, y):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)

    Returns:
        float: The first moment in x.

    """
    return np.sum(x*y)/np.sum(y)

def second_moment(x, y, mu):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)

    Returns:
        float: The second moment in x.

    """
    return np.sum(y*(x-mu)**2 )/np.sum(y)

#Convergent method for moments calculation
def closing_window(x, y, mu1_init = None, window_max=25, window_min=10,
     window_step_size=1 ):
    """Calculate first and second moments using the 'closing-window method' (O'Sullivan et al. 2020).

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        mu1_init (float): Initial guess for first moment, used to center window.
            If none given, center value of x will be used.
        window_max (float): Starting window size for calculation (in same unit as x)
        window_min (float): Minimum window size for calculation. (same units as x)
        window_step_size (float): Decrease in window size for each step (same units as x).

    Returns:
        float: The first moment in x.
        float: The second moment in x.

    """
    #Take user input as initial guess or use center of x-range
    if mu1_init == None: mu_1 = x[int(len(x)/2)]
    else: mu_1 = mu1_init

    #Initialize window at maximum size
    window = window_max

    # Loop over window size
    while window > window_min:

        #Get indices of values to use for this calculation
        usex = ( np.abs(x - mu_1) < window/2 ) & (y > 0)

        #Calculate moments (i.e. update window center)
        mu_1 = first_moment(x[usex], y[usex])
        mu_2 = second_moment(x[usex], y[usex], mu_1 )

        #Update window size
        window -= window_step_size

    return mu_1, mu_2
