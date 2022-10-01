from GP_regression import *
from Gaussian_fit import *
from plots import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/hannahcollier/Documents/solo/data/QPPS/300322/test_arr.csv')

X_scaled, y_scaled, scaler_x, scaler_y = rescale(np.asarray(df['time']), np.asarray(df['counts']))

X_train, y_train = train_set(X_scaled,y_scaled)

mean_prediction, std_prediction,  x_inv, mean_prediction_inv, std_inv = GP_fit(X_train, y_train, scaler_x, scaler_y, X_scaled, 0.009, 1**2, 0.16)

peak_time,peak_vals,width_time = peak_finder(mean_prediction_inv, x_inv)

guess = guess_fn(peak_time,peak_vals,width_time)

bounds_lower, bounds_upper = bounds(peak_time, peak_vals, width_time, guess)

popt, pcov, gaus_params, fit_para, fit, resid = gaussians_fit(np.asarray(df['time']),np.asarray(df['counts']), guess, bounds_lower, bounds_upper)

#peak_no, a, b, means_edit, c, d , e, f, spear, pearson = line_fit(popt, x_inv)

gaussian_decomp(np.asarray(df['time']), np.asarray(df['counts']), x_inv, mean_prediction_inv, std_inv, fit, fit_para)

