import pickle
import numpy as np
import scipy.io

with open('results2.pkl','rb') as f:
    [forest_time,prox_time,results] = pickle.load(f)

forest_time_mean = np.zeros((6,5,3))
for i in range(6):
    for j in range(5):
        for k in range(3):
            forest_time_mean[i,j,k] = forest_time[i,j,k,:].mean()
print(forest_time_mean)

scipy.io.savemat('results2.mat',{'forest_time': forest_time,'prox_time': prox_time, 'results': results})