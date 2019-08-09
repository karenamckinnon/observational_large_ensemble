============================
Observational Large Ensemble
============================

This package contains code for creation of an "Observational Large Ensemble" of monthly temperature, precipitation, and sea level pressure. 

The approach is based upon modeling each variable as a linear combination of a mean, trend, contribution from large-scale modes of variability, and residual climate noise. The ensemble is created through randomization of the latter two components. The time series of the large-scale modes are randomized through application of the Iterative Amplitude Adjusted Fourier Transform. The climate noise is block bootstrapped in time. 

The forced component is estimated using the methodology of Dai et al., 2015, Nature Climate Change. Specifically, the observations are regressed against the time series of the global mean, ensemble mean of each variable. The forced trend for sea level pressure is assumed to be zero.

More complete technical and scientific documentation can be found in [McKinnon and Deser, 2018, Journal of Climate](https://journals.ametsoc.org/doi/full/10.1175/JCLI-D-17-0901.1). 

* Free software: MIT license

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
