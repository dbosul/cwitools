from scipy.optimize import differential_evolution

import numpy as np

def fit_de(model_func, model_bounds, xdata, ydata):
    fit = differential_evolution(
            rss_func,
            model_bounds,
            args=(xdata, ydata, model_func)
    )
    return fit

def rss_func(model_params, x, y, model_func):
    return np.sum( np.power((y - model_func(model_params, x)), 2))

def gauss1d(params, x):
    amp, mean, std = params
    return amp*np.exp( -0.5*np.power((x - mean) / std, 2))

def moffat1d(params, x):
    amp, mean, alpha, gamma = params
    return amp*np.power(1 + np.power((x - mean)/gamma, 2), -alpha)

def bic_weights(bic_list):
    """Get weights representing relative likelihood of models based on BICs.

    Args:
        bic_list (array-like): Array of BIC values for models.

    Returns:
        weights (Numpy.array): Array of relative likelihoods for models.

    """
    delta_i = aic_list - np.min(aic_list) #Minimum AIC value of set
    rel_L = np.exp(-0.5*delta_i) #Proportional likelihood term
    weights = rel_L/np.sum(rel_L)
    return weights

def rss(data, model):
    """Get the residual sum of squares for a model and data.

    Args:
        data (NumPy.ndarray): Observed data.
        model (NumPy.ndarray): Modeled data.

    Returns:
        rss (float): Sum of square residuals for the input data and model.

    """
    return np.sum(np.power(data-model, 2))

def fwhm2sigma(fwhm):
    """Convert a gaussian Full-Width at Half Maximum to a standard deviation.

    Args:
        fwhm (float): Full-Width at Half Maximum of a Gaussian function.

    Returns:
        float: The standard deviation of the equivalent Gaussian.

    """
    return fwhm/(2*np.sqrt(2*np.log(2)))


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
        return n*np.log(rss_in/n) + k*np.log(n)

def aic(model, data, k, correct=False):
    """Calculate the Akaike Information Criterion for a model.

    Args:
        model (NumPy.ndarray): The model data being evaluated
        data (NumPy.ndarray): The data being modeled
        k (int): The number of parameters in the model
        correct (bool): Apply correction for small sample sizes.

    Returns:
        aic (float): The Akaike Information Criterion

    """
    n = model.size
    rss_in = rss(data, model)
    aic0 = n*np.log(rss_in/n) + 2*k
    if correct: return aic0 + (2*k*k + 2*k)/(n - k - 1)
    else: return aic0
