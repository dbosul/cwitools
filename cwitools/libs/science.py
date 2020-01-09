"""CWITools Science Functions Library.

This module contains functions for use in scientific analysis or the generation
of scientific products from data cubes (such as pseudo-NarrowBand images).

"""
from cwitools.libs import cubes

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.constants import G
from astropy.convolution import Gaussian1DKernel,Box1DKernel,Gaussian2DKernel,convolve
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
