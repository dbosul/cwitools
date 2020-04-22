"""Tools for model fitting, evaluation and comparison"""
from scipy.optimize import differential_evolution
from scipy.special import voigt_profile
import numpy as np

###
### 1D MODELS in form f(params, x)
###
def gauss1d(params, x):
    """Simple wrapper for a one-dimensional Gaussian.

    I(x) = I0 * exp(-(x - x0)^2 / (2 * sigma^2))

    Args:
        params (list): 1D Gaussian parameters (amplitude, mean, std_dev)
        x (numpy.array): The input  on which to evaluate the model

    Returns:
        numpy.array: The Gaussian model output

    """
    amp, mean, std = params
    return amp*np.exp( -0.5*np.power((x - mean) / std, 2))

def moffat1d(params, x):
    """Simple wrapper for a one-dimensional Moffat profile.

    I(x) = I0 * (1 + ((x - x0)/ gamma)^2)^(-alpha)

    Args:
        params (list): 1D Moffat parameters (I0, x0, alpha, gamma)
        x (numpy.array): The input  on which to evaluate the model

    Returns:
        numpy.array: The Moffat model output

    """
    I0, x0, alpha, gamma = params
    return I0 * np.power(1 + np.power((x - x0) / gamma, 2), -alpha)

def voigt1d(params, x):
    """Wrapper for ~scipy.special.voigt_profile in the form f(params, x)

    From SciPy documentation:
    I(x) = Re[w(z)] / (sigma * sqrt(2 * pi))
    where w(z) is the Faddeeva function and z = (x + i * y) / (sqrt(2) * sigma)

    Args:
        params (list): 1D Voigt parameters (amplitude, sigma, gamma)
        x (numpy.array): The input  on which to evaluate the model

    Returns:
        numpy.array: The Voigt profile output

    """
    amplitude, sigma, gamma = params
    return amplitude * voigt_profile(x, sigma, gamma)

def sersic1d(params, r):
    """1D Sersic surface brightness profile.

    I(r) = I0 * exp(-b_n * (r / Re)^(1 / n))
    where b_n is approximated as (2 * n - 1 / 3)

    Args:
        params (list): Model parameters (I0, Re, n)
        r (numpy.array): Model input

    Returns:
        numpy.array: The Sersic model
    """
    I0, re, n = params
    bn = 2 * n - 1 / 3 #Approximate b_n
    return I0 * np.exp(-bn * (np.power(r / re, 1.0 / n) - 1))

def exp1d(params, r):
    """1D Exponential surface brightness profile.

    I(r) = I0 * exp(-beta * r/Re)

    Args:
        params (list): Model parameters (I0, Re, beta)
        x (numpy.array): Model input

    Returns:
        numpy.array: The Exponential model
    """
    I0,re,beta = params
    return I0 * np.exp(-beta * r / re )

def powlaw1d(params,r):
    """1D Exponential surface brightness profile.

    I(R) = I0 * (R/Re) ** alpha

    Args:
        params (list): Model parameters (I0, Re, alpha)
        x (numpy.array): Model input

    Returns:
        numpy.array: The Power-law model

    Examples:


    """
    c, re, alpha = params
    return c*((r/re)**alpha)


###
### MODEL FITTING
###
def fit_model1d(model_func, model_bounds, xdata, ydata):
    """Wrapper for fitting a 1D model using SciPy's differential evolution.

    Args:
        model_func (callable): The model function, of the form f(params, x)
            where params is a list of model parameters.
        model_bounds (list): List of tuples representing (lower, upper) bounds
            on the model parameters. e.g. [(0,1), (-1,-1), ... ]
        xdata (numpy.array): Input x data (e.g. wavelength)
        ydata (numpy.array): Input y data to fit to (e.g. flux)

    Returns:
        scipy.optimize.OptimizeResult: The result of the fit.

    """
    fit = differential_evolution(
            rss_func,
            model_bounds,
            args=(xdata, ydata, model_func)
    )
    return fit

def rss_func(model_params, x, y, model_func):
    """Calculate the residual sum of squares for a model + data.

    Args:
        model_params (list): The parameters of the model
        x (numpy.array): The input x axis data
        y (numpy.array): The input y axis data
        model_func (callable): The model function

    Returns:
        float: The residual sum of squares

    """
    return np.sum(np.power((y - model_func(model_params, x)), 2))

def rss(data, model):
    """Get the residual sum of squares for a model and data.

    Args:
        data (NumPy.ndarray): Observed data.
        model (NumPy.ndarray): Modeled data.

    Returns:
        rss (float): Sum of square residuals for the input data and model.

    """
    return np.sum(np.power(data-model, 2))

###
### MODEL COMPARISON
###
def bic(model, data, k):
        """Calculate the Bayesian Information Criterion for a model.

        Args:
            model (NumPy.ndarray): The model data being evaluated
            data (NumPy.ndarray): The data being modeled
            k (int): The number of parameters in the model

        Returns:
            bic (float): The Bayesian Information Criterion

        """
        n = model.size
        rss_in = rss(data, model)
        return n * np.log(rss_in / n) + k * np.log(n)

def aic(model, data, k):
    """Calculate the Akaike Information Criterion for a model.

    Args:
        model (NumPy.ndarray): The model data being evaluated
        data (NumPy.ndarray): The data being modeled
        k (int): The number of parameters in the model

    Returns:
        aic (float): The Akaike Information Criterion

    """
    n = model.size
    rss_in = rss(data, model)
    aic0 = 2 * k + n * np.log(rss_in)

    #Correction term for small samples
    #corr -> 0 as n -> inf
    corr = (2 * k * k + 2 * k) / (n - k - 1)

    return aic0 + corr

def bic_weights(bic_list):
    """Get weights representing relative likelihood of models based on BICs.

    Args:
        bic_list (array-like): Array of BIC values for models.

    Returns:
        weights (Numpy.array): Array of relative likelihoods for models.

    """
    if type(bic_list) is list:
        bic_list = np.array(bic_list)
    delta_i = bic_list - np.min(bic_list) #Minimum AIC value of set
    rel_L = np.exp(-0.5 * delta_i) #Proportional likelihood term
    weights = rel_L / np.sum(rel_L)
    return weights


###
### MISCELLANEOUS
###
def fwhm2sigma(fwhm):
    """Convert Gaussian FWHM (Full-Width at Half Maximum) to standard deviation.

    Args:
        fwhm (float): Full-Width at Half Maximum of a Gaussian function.

    Returns:
        float: The standard deviation of the equivalent Gaussian.

    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def sigma2fwhm(sigma):
    """Convert standard deviation to FWHM (Full-Width at Half Maximum).

    Args:
        sigma (float): Standard deviation of a Gaussian function.

    Returns:
        float: The full-width at half-maximum of the same Gaussian function

    """
    return sigma * 2 * np.sqrt(2 * np.log(2))
