.. -*- mode: rst -*-
scikit-THI
============
About scikit-THI
~~~~~~~~~~~~

scikit-THI is a modified version of the scikit-learn [1] repository. The modifications are mainly in the Random Forest implementation of the scikit-learn repository. The changes are the following

   - Unsupervised Random Forest implementaion from [2] is added
   - Node proximity from [3] is added
   - Path proximity from [2] is added
   - Random Forest Indexing is added

About scikit-learn
~~~~~~~~~~~~

scikit-learn is a Python module for machine learning built on top of
SciPy and is distributed under the 3-Clause BSD license.

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <http://scikit-learn.org/dev/about.html#authors>`__ page
for a list of core contributors.

It is currently maintained by a team of volunteers.

Website: http://scikit-learn.org


Installation
============

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
   - Follow the procedure in https://dzone.com/articles/executable-package-pip-install for wheel creation

If the ``test pass`` message is received, the installation is proper and the Random Forest implementation can be used.


Using scikit-THI
============
Using the proximity calculation
~~~~~~~~~~~~~~~~~
Example ::

   from sklearn.ensemble import RandomForestClassifier
   from sklearn.datasets import make_blobs
   X, Y = make_blobs(n_samples=10000, centers=5, n_features=100, random_state=0)
   estimator = RandomForestClassifier(min_samples_leaf=1,max_features="sqrt",n_estimators=nTrees_vect[n],oob_score=True)
   estimator.fit(X, Y)
   proximity = estimator.get_proximity_matrix(X, typeCalc = 'Node') # typeCalc = 'PathNormal' for path proximity
   
Using the random indexing
~~~~~~~~~~~~~~~~~
Example ::

   from sklearn.ensemble import RandomForestClassifier
   from sklearn.datasets import make_blobs
   X, Y = make_blobs(n_samples=10000, centers=5, n_features=100, random_state=0)
   estimator = RandomForestClassifier(min_samples_leaf=1,max_features="sqrt",n_estimators=100,oob_score=True)
   estimator.fit(X, Y)
   estimator.index()
   rfapX = estimator.encode_rfap(X)


Using the Unsupervised Forest
~~~~~~~~~~~~~~~~~
Example ::

   import numpy as np
   from sklearn import ensemble
   N = 2500, D = 10, posMean = 10, n_trees = 100
   X1 = np.random.randn(N//2,D) + posMean, X2 = np.random.randn(N//2,D) - posMean
   X = np.concatenate((X1,X2)), Y = None
   estimator = ensemble.UnsupervisedRandomForest(n_estimators=n_trees,random_state=42)
   estimator.fit(X, Y)
   matrix = estimator.get_proximity_matrix(X, typeCalc='PathNormal')
   
Citation
============
   @misc{lakwurst.2020, 
    author = {{L. Balasubramanian} and {J. Wurst}, 
    date = {2020},
    title = {{scikit - THI}},
    journal={GitHub repository},
    url = {\url{https://github.com/lab176344/scikit-THI}}
   }

Contributors
============

Lakshman Balasubramanian (lakshman.balasubramanian@thi.de), Jonas Wurst (jonas.wurst@thi.de)

Reference
============
[1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011

[2] F. Kruber, J. Wurst, E. Sánchez Morales, S. Chakraborty and M. Botsch, "Unsupervised and Supervised Learning with the Random Forest Algorithm for Traffic Scenario Clustering and Classification", 30th IEEE Intelligent Vehicles Symposium , 2019 

[3] Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001). https://doi.org/10.1023/A:1010933404324




