import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from gp_regression import *

def peak_finder(mean_prediction,x_inv):
	grad = np.gradient(mean_prediction.reshape(-1))
	y = [0 for i in range(len(mean_prediction))]
	idx = np.argwhere(np.diff(np.sign(grad - y)))
	val = [mean_prediction[i] for i in idx.reshape(-1)]
	if val[0] > val[1]:
		#print('deleted one')
		idx = np.insert(idx,0,0)
		val = np.insert(val,0,mean_prediction[0])
		
	if val[-1] < val[-2]:
		#print('added one')
		idx = np.delete(idx, -1, 0)
		val = np.delete(val, -1, 0)

	time_idx = [x_inv[i] for i in idx]
	# if len(time_idx)%2 != 0:
	# 	print(len(time_idx)%2)
	# 	plt.plot(x_inv, mean_prediction)
	# 	plt.plot(time_idx, [1 for i in range(len(time_idx))],'x')
	# 	plt.show()

	idx_every_second = idx[1::2] 
	base_time = np.asarray(time_idx[::2])
	peak_time = np.asarray(time_idx[1::2])
	width_time = peak_time-base_time #in s
	peak_vals = np.asarray([mean_prediction[i] for i in idx_every_second]).reshape(-1)
	return peak_time, peak_vals, width_time

def guess_fn(peak_time,peak_vals,width_time):
	guess=[]
	for i in range(len(width_time)):
	    guess.append(peak_time.reshape(-1)[i]) 
	    guess.append(peak_vals[i])  
	    guess.append(width_time.reshape(-1)[i]/2.355)
	return guess

def gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2] #sigma
        y = y + np.abs(amp) * np.exp( -((x - ctr)/wid)**2)
    return y


def bounds(peak_time, peak_vals, width_time, guess):
	bounds_lower=[]
	bounds_upper=[]
	for i in range(len(guess)//3):
		bounds_lower.append(peak_time[i]-10)
		bounds_lower.append(peak_vals[i]-200)
		bounds_lower.append(width_time[i]-20)
		bounds_upper.append(peak_time[i]+10)
		bounds_upper.append(peak_vals[i]+200)
		bounds_upper.append(width_time[i]+20)
	return bounds_lower, bounds_upper


def gaussians_fit(time, counts, guess, bounds_lower=None, bounds_upper=None):
	if bounds_lower is not None:
		popt, pcov = curve_fit(gaussian, time.reshape(-1), counts.reshape(-1), p0=guess,bounds=((bounds_lower),(bounds_upper)),
	                       maxfev=60000) 
	else:
		popt, pcov = curve_fit(gaussian, time.reshape(-1), counts.reshape(-1), p0=guess,
	                       maxfev=60000) 

	gaus_params = np.empty((int(len(popt)/3),3))
	for i in range(int(len(popt)/3)):
	    gaus_params[i] = popt[i*3:i*3+3]
    		
	fit_para = np.empty((int(len(popt)/3),len(time)))
	for i in range(int(len(popt)/3)):
	    fit_para[i] = gaussian(time, *gaus_params[i,:]).reshape(-1)
	    
	#print(gaus_params)
	fit = gaussian(time, *popt)
	resid = fit - counts

	return popt, pcov, gaus_params, fit_para, fit, resid


def line_fit(popt, x_inv):
	means= popt[::3]
	std = popt[2::3]/np.sqrt(2)
	peak_no = np.arange(1,len(means)+1)
	means_edit = means - x_inv[0]
	a, b = np.polyfit(peak_no, means_edit, 1)	
	spear = spearmanr(peak_no, means_edit)
	pearson = pearsonr(peak_no,means_edit)
	return a, b, spear, pearson


