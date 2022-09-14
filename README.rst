============================
Observational Large Ensemble
============================

This package contains code for creation of an "Observational Large Ensemble" of monthly temperature, precipitation, and sea level pressure. 

The approach is based upon modeling each variable as a linear combination of a mean, trend, contribution from large-scale modes of variability, and residual climate noise. The ensemble is created through randomization of the latter two components. The time series of the large-scale modes are randomized through application of the Iterative Amplitude Adjusted Fourier Transform. The climate noise is block bootstrapped in time. 

The forced component is estimated using the methodology of Dai et al., 2015, Nature Climate Change. Specifically, the observations are regressed against the time series of the global mean, ensemble mean of each variable. The forced trend for sea level pressure is assumed to be zero.

More complete technical and scientific documentation can be found in McKinnon and Deser, 2018, Journal of Climate. An updated methodology focused on precipitaton in published in McKinnon and Deser, 2021, Journal of Climate. The methodological changes are:

- The data product is monthly, rather than seasonal.
- The seasonality of ENSO variance is included.
- Rather than creating two orthogonal modes from applying PCA the ENSO and PDO, PDO is orthogonalized with respect to ENSO using the Gram-Schmidt orthogonalization procedure.
- The block boostrap size is estimated using the iterative formula of Wilks (1997), Journal of Climate, rather than fixed at two years.
- The AMV time series is lowpass filtered using a cutoff-frequency of 20 years.
- Precipitation is modeled after using a Box-Cox transform. UPDATE SEPTEMBER 2022: It has come to my attention that the use of the Box-Cox transform was leading to unrealistic precipitation values in tropical regions where it is common for there to be zero precipitation during the dry season. For the moment, please restrict use to the extratropics, where the model has been more thorouhgly validated. 

The code is designed to be applied either to gridded observational datasets or to the CESM1-LE, which is used as a testbed for the methodology. 

Please contact Karen McKinnon (kmckinnon@ucla.edu) if you are using the code or methods in your work.

* Free software: MIT license

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
