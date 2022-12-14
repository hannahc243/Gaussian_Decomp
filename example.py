from gaussian_decomp_start import hyperparam_opt, analyse_series, error_analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

df = pd.read_csv('040522_data.csv')

params = {"alpha": np.arange(0.001,0.015,0.001), 
    "kernel": [ConstantKernel(k, constant_value_bounds="fixed")*RBF(l, length_scale_bounds="fixed") for l in np.arange(0.01,0.1,0.01) for k in np.arange(1,5,0.1)]}

counts= np.asarray(df['counts'])
time  = np.asarray(df['time'])
yerr = np.asarray(df['yerr'])

# ## Call function to find optimal hyperparameters for kernel 
best_score, best_params = hyperparam_opt(time, counts, params)
print(best_score, best_params)


# ## Best parameters obtained from hyperparameter optimisation
alpha = 0.007
ker_amplitude = 1.34**2
length_scale = 0.08

## Function to run full process on one time series
analyse_series(time, counts, yerr, alpha, ker_amplitude, length_scale) #, time_format=pd.to_datetime(df['time_format']))


## Function to run error analysis on a single time series -  50 iterations
error_analysis(time, counts, yerr, alpha, ker_amplitude, length_scale, time_format=pd.to_datetime(df['time_format']))
