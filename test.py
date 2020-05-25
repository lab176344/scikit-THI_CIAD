# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:12:17 2020

@author: balasubramanian
"""
test_pass = False
try:
    from sklearn.ensemble import RandomForestClassifier
    test_pass = True
except  ImportError as err:
    print('There is problem in either tree.pyx or _classes.py')
    
try: 
    from sklearn.ensemble import _indexing_tree
    test_pass = True
except ImportError as err:
    print('There is problem in with the indexing')


if test_pass:
    print('test pass')