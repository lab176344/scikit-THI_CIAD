.. -*- mode: rst -*-
scikit-THI
============

``scikit-THI`` is a modified version of the scikit-learn [1] repository. The modifications are mainly in the Random Forest implementation of the scikit-learn [1] repository. 

About scikit-learn
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

scikit-THI requires ::

   - conda (4.8.3)
   - python (3.8)
   - numpy (1.18.4)
   - scipy (1.4.1)
   - joblib ( 0.15.1)
   - cython (0.29.17)
   - MSVC v142 -VS2019 C++-64/86-Buildtools (14.26)
   - Windows SDK 10.0.18362.0)

User installation
~~~~~~~~~~~~~~~~~

The installation procedure is as follows ::
   
   - Install Anaconda from https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe
   - Install MSVC community 2019, in the VS installer please select the following 
      - Desktop development with C++
      - Development of universal windows plattform
   - Open Anaconda prompt
      - conda create -n venv (Recommended)
      - conda activate venv
      - git clone https://github.com/lab176344/scikit-THI.git
      - cd scikit-THI
      - pip install --verbose --editable .
      - python test.py

If the ``test pass`` message is received, the installation is proper and the Random Forest implementation can be used.


Using scikit-THI
~~~~~~~




Citation
~~~~~~~~

