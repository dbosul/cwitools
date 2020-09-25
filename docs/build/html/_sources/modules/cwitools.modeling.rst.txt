.. _Modeling:

Modeling Module (cwitools.modeling)
=======================================

The modeling module contains wrappers for common models (e.g. Gaussian, Moffat and Voigt profiles) which facilitate model fitting using SciPy's differential_evolution (scipy.optimize.differential_evolution), a stochastic optimizer which is useful for well-bounded fitting scenarios. It also contains wrappers to fit these models to data, and wrappers for the Akaike and Bayesian information criteria (AIC and BIC) to facilitate model comparison.

.. automodule:: cwitools.modeling

   .. rubric:: Functions

   .. autosummary::

      aic
      bic
      bic_weights
      covar_curve
      doublet
      exp1d
      fit_model1d
      fit_model2d
      fwhm2sigma
      gauss1d
      gauss2d
      gauss2d_sym
      moffat1d
      moffat2d
      powlaw1d
      rss
      rss_func1d
      rss_func2d
      sersic1d
      sigma2fwhm
      voigt1d
