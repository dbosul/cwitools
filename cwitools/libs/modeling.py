def akaike_weights(aic_list):
    """Get weights representing relative likelihood of models based on AIC values.

    Args:

    """
    delta_i = aic_list - np.min(aic_list) #Minimum AIC value of set
    rel_L = np.exp(-0.5*delta_i) #Proportional likelihood term
    weights = rel_L/np.sum(rel_L)
    return weights

def rss(data, model): return np.sum(np.power(data-model, 2))

def fwhm2sigma(fwhm):
    """Convert a gaussian Full-Width at Half Maximum to a standard deviation.

    Args:
        fwhm (float): Full-Width at Half Maximum of a Gaussian function.

    Returns:
        float: The standard deviation of the equivalent Gaussian.

    """
    return fwhm/(2*np.sqrt(2*np.log(2)))

def aic(rss, k, n): return n*np.log(rss/n) + 2*k

def aic_c(rss, k, n):
    aic0 = aic(rss, k, n)
    correction = (2*k*k + 2*k)/(n - k - 1)
    return aic0 + correction

def bic(rss, k, n): return n*np.log(rss/n) + k*np.log(n)

def gauss1D(x,par):
    """A simple one-dimensional gaussian.

    Args:
        x (scalar or np.array): The domain for the gaussian.
        par (list/tuple): A list/tuple of amplitude, mean, standard-deviation.

    Returns:
        scalar or numpy.array: The value of the gaussian function at/over x.
    """
    return par[0]*np.exp(-0.5*np.power(x-par[1],2)/par[2] )
