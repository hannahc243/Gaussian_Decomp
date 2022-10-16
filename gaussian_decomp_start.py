from GP_regression import *
from Gaussian_fit import *
from Figures import *
import numpy as np
import json
import pickle


def hyperparam_opt(time, counts, params):
	"""Performs a grid search over a parameter space to find the optimal hyperparameters.
	Parameters
    ----------

    time : ndarray
        an array of times
    counts : ndarray
        an array of data points can be counts or count rate
    parans : dictionary
    	a dictionary of the range of kernel hyperparameters to search over.
	"""
	X_scaled, y_scaled, scaler_x, scaler_y = rescale(time, counts)
	plt.plot(X_scaled,y_scaled)
	plt.show()

	grid = grid_search(X_scaled,y_scaled, params)
	return grid.best_score_, grid.best_params_


def analyse_series(time, counts, yerr, alpha, ker_amplitude, length_scale, time_format=None):
	"""Analyse a single, generic timeseries using the Gaussian decomposition method.
	    
	    Parameters
	    ----------

	    time : ndarray
	        an array of times
	    counts : ndarray
	        an array of data points can be counts or count rate
	    yerr : ndarray
	    	an array of error values of counts
	    alpha : float
	    	a float value of the optimal kernel alpha parameter determined through Grid search
	    ker_amplitude : float
	    	a float value of the optimal kernel amplitude determined through Grid search
	    length_scale : float
	    	a float value of the optimal kernel length scale determined through Grid search
	    time_format : pandas series datetime object, optional
	    	a pandas series datetime object of the time array     
	"""	

	X_scaled, y_scaled, scaler_x, scaler_y = rescale(time, counts)

	X_train, y_train = train_set(X_scaled,y_scaled) #X_scaled, y_scaled 
	x_inv, mean_prediction_inv, std_inv = gp_fit(X_train, y_train, scaler_x, scaler_y, X_scaled, alpha, ker_amplitude, length_scale) 

	peak_time,peak_vals,width_time = peak_finder(mean_prediction_inv, x_inv) 

	guess = np.asarray(guess_fn(peak_time,peak_vals,width_time))

	bounds_lower, bounds_upper = bounds(peak_time, peak_vals, width_time, guess)

	popt, pcov, gaus_params, fit_para, fit, resid = gaussians_fit(time,counts, guess, bounds_lower, bounds_upper)

	resid_std = gaussian_decomp(time, counts, x_inv, mean_prediction_inv, std_inv, fit, fit_para, resid, yerr, time_format)
	
	slope, intersect, spear, pearsonr = line_fit(popt, x_inv)

	line_plot(popt[::3], x_inv, slope, intersect)

	save_run(time, counts, yerr, alpha, ker_amplitude, length_scale, resid, resid_std, slope, gaus_params, fit_para, fit, time_format=time_format)

	print(f'Analysis summary \n----------------- \n Number of Gaussians Fit: {len(popt)/3} \n Slope of Line Fit: {np.round(slope,1)} s \n Pearson correlation coefficient, pvalue: {np.round(pearsonr[0],3)}, {pearsonr[1]} \n Residual std: {np.round(resid_std,2)}' )	

	return 



def error_analysis(time, counts, yerr, alpha, ker_amplitude, length_scale, time_format=None, description=None):	
	"""Perform error analysis by adding random error to the original lightcurve and do fits 50 times.

    Parameters
    ----------

    time : ndarray
        an array of times
    counts : ndarray
        an array of data points can be counts or count rate
    yerr : ndarray
    	an array of error values of counts
    alpha : float
    	a float value of the optimal kernel alpha parameter determined through Grid search
    ker_amplitude : float
    	a float value of the optimal kernel amplitude determined through Grid search
    length_scale : float
    	a float value of the optimal kernel length scale determined through Grid search
    time_format : pandas series datetime object, optional
    	a pandas series datetime object of the time array   
    description : string, optional
    	name of file to call figure saves
    """

	mean = 0
	std = 1
	slopes = []
	intersects = []
	for i in range(50):
		print(f'iteration {i}')
		samples = np.random.normal(mean, std, size=counts.size)
		counts_new = counts + samples*yerr

		X_scaled, y_scaled, scaler_x, scaler_y = rescale(time, counts_new)

		X_train, y_train = train_set(X_scaled,y_scaled) 

		x_inv, mean_prediction_inv, std_inv = gp_fit(X_train, y_train, scaler_x, scaler_y, X_scaled, alpha, ker_amplitude, length_scale) 

		peak_time,peak_vals,width_time = peak_finder(mean_prediction_inv, x_inv)

		guess = np.asarray(guess_fn(peak_time,peak_vals,width_time))

		bounds_lower, bounds_upper = bounds(peak_time, peak_vals, width_time, guess)

		popt, pcov, gaus_params, fit_para, fit, resid = gaussians_fit(time,counts_new, guess, bounds_lower, bounds_upper)

		if description is None:
			file_name=datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + f'_Error_analysis_Iteration_{i}'
		else:
			file_name = description + f'_Error_analysis_Iteration_{i}'

		gaussian_decomp_plot(time, counts, x_inv, mean_prediction_inv, std_inv, fit, fit_para, resid, yerr, time_format, savedir=None, description=file_name)

		slope, intersect, spear, pearsonr = line_fit(popt, x_inv)

		line_plot(popt[::3], x_inv, slope, intersect, description=file_name)

		a,b, spear, pearsonr = line_fit(popt, x_inv)

		slopes.append(a)
		intersects.append(b)
	
	print(f'Analysis summary \n----------------- \n Slope mean: {np.round(np.mean(slopes),1)} s \n Std: {np.round(np.std(slopes),1)} s \n Min: {np.round(np.min(slopes),1)} \n Max: {np.round(np.max(slopes),1)}' )
	return 


def save_run(time, counts, yerr, alpha, ker_amplitude, length_scale, resid, resid_std, slope, gaus_params, fit_para, fit, time_format=None, description=None, savedir=None):
	"""Save parameters of run to pickle file
	Parameters
    ----------

    time : ndarray
        an array of times
    counts : ndarray
        an array of data points can be counts or count rate
    yerr : ndarray
    	an array of error values of counts
    alpha : float
    	a float value of the optimal kernel alpha parameter determined through Grid search
    ker_amplitude : float
    	a float value of the optimal kernel amplitude determined through Grid search
    length_scale : float
    	a float value of the optimal kernel length scale determined through Grid search
    resid : 
    resid_std : 
	slopes : 
	gaus_params : 
	fit_para : 
	fit : 
    time_format : pandas series datetime object, optional
    	a pandas series datetime object of the time array   
    description : string, optional
    	name to call file saves
    savedir : string, optional
   		directory to save plots to
	"""
	if time_format is not None:
		save_dict = {'time':time, 'counts':counts, 'yerr':yerr, 'alpha':alpha, 'ker_amplitude':ker_amplitude,
		'length_scale':length_scale, 'resid':resid, 'resid_std': resid_std, 'time_format':time_format, 'slope':slope, 'gaus_params':gaus_params, 'fit_para':fit_para, 'fit':fit}

	else:
		save_dict = {'time':time, 'counts':counts, 'yerr':yerr, 'alpha':alpha, 'ker_amplitude':ker_amplitude,
		'length_scale':length_scale, 'resid':resid}

	if not description:
		description = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	if not savedir:
		os.makedirs(os.path.expanduser('~/Gaussian_decomp_repository/save'), exist_ok=True)
		savedir = os.path.expanduser('~/Gaussian_decomp_repository/save')

	savefilename = 'variables_' + description + ".pkl"
	fname = os.path.join(savedir,savefilename)

	pickle.dump(save_dict,open(fname,'wb'))
        

