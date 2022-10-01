import numpy as np
from scipy.optimize import curve_fit

def peak_finder(mean_prediction,x_inv):
	grad = np.gradient(mean_prediction.reshape(-1))
	y = [0 for i in range(len(mean_prediction))]
	idx = np.argwhere(np.diff(np.sign(grad - y)))
	val = [mean_prediction[i] for i in idx.reshape(-1)]
	if val[0] > val[1]:
		idx = np.insert(idx,0,0)
		time_idx = np.insert(time_idx,0,x_inv[0])
		val = np.insert(val,0,mean_prediction[0])

	elif val[-1] < val[-2]:
		idx = np.delete(idx, -1, 0)
		time_idx = np.delete(time_idx, -1, 0)
		val = np.delete(val, -1, 0)
	
	time_idx = [x_inv[i].reshape(-1) for i in idx]	
	peak_idx_base = np.arange(0,len(idx)-1,2)
	peak_idx_xval = np.asarray(idx[1::2]).reshape(-1)
	fwfm = np.asarray([(idx[i+1] - idx[i]) for i in peak_idx_base]) 
	heights = np.asarray([mean_prediction[idx[i+1]]-mean_prediction[idx[i]] for i in peak_idx_base]).reshape(-1)
	idx_every_second = idx[1::2] #starting from second one (time at peak)
	base_time = np.asarray(time_idx[::2])
	peak_time = np.asarray(time_idx[1::2])
	width_time = np.asarray(peak_time)-np.asarray(base_time) #in s
	peak_vals = np.asarray([mean_prediction[i] for i in idx_every_second]).reshape(-1)
	return peak_time, peak_vals, width_time

def guess_fn(peak_time,peak_vals,width_time):
	guess=[]
	for i in range(len(width_time)):
	    guess.append(peak_time.reshape(-1)[i]) #mean should be x val at peak of pulse
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


def gaussians_fit(time, counts, guess, bounds_lower, bounds_upper):
	popt, pcov = curve_fit(gaussian, time.reshape(-1), counts.reshape(-1), p0=guess,bounds=((bounds_lower),(bounds_upper)),
                       maxfev=60000)
	
	gaus_params = np.empty((int(len(popt)/3),3))
	for i in range(int(len(popt)/3)):
	    gaus_params[i] = popt[i*3:i*3+3]
    
	fit_para = np.empty((int(len(popt)/3),len(time)))
	for i in range(int(len(popt)/3)):
	    fit_para[i] = gaussian(time, *gaus_params[i,:]).reshape(-1)
	    
	fit = gaussian(time, *popt)
	resid = fit - counts

	return popt, pcov, gaus_params, fit_para, fit, resid
