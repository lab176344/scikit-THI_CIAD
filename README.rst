.. -*- mode: rst -*-
Scikit-THI
============

Scikit-THI is a modified version of the scikit-learn [1] repository. The modifications are mainly in the Random Forest implementation of the scikit-learn [1] repository. 

About Scikit
---------------
[1] scikit-learn is a Python module for machine learning built on top of
SciPy and is distributed under the 3-Clause BSD license.

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <http://scikit-learn.org/dev/about.html#authors>`__ page
for a list of core contributors.

It is currently maintained by a team of volunteers.

Website: http://scikit-learn.org


Installation
------------

Dependencies
~~~~~~~~~~~~

scikit-THI requires 
   - conda (4.8.3)
   - python (3.8)
   - numpy (1.18.4)
   - scipy (1.4.1)
   - joblib ( 0.15.1)
   - cython (0.29.17)


User installation
~~~~~~~~~~~~~~~~~

The installation procedure is as follows :: 


Testing
~~~~~~~

After installation, you can launch the test suite from outside the
source directory (you will need to have ``pytest`` >= 3.3.0 installed)::

    pytest sklearn

See the web page http://scikit-learn.org/dev/developers/advanced_installation.html#testing
for more information.

    Random number generation can be controlled during testing by setting
    the ``SKLEARN_SEED`` environment variable.


Citation
~~~~~~~~

