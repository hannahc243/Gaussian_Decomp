from gaussian_decomp_start import analyse_series
import pandas as pd
import numpy as np

df = pd.read_csv('040522_data.csv')

counts= np.asarray(df['counts'])
time  = np.asarray(df['time'])
yerr = np.asarray(df['yerr'])

window_len = 7 # window length input to savgol_filter
## Function to run full process on one time series
analyse_series(time, counts, yerr, window_length=window_len, polyorder=3) #, time_format=pd.to_datetime(df['time_format']))


## Function to run error analysis on a single time series -  50 iterations
# error_analysis(time, counts, yerr, alpha, ker_amplitude, length_scale, time_format=pd.to_datetime(df['time_format']))
