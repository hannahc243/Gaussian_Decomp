import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from matplotlib.dates import DateFormatter
import datetime
import os

def line_plot(means,x_inv, a, b, description=None, savedir = None):
	peak_no = np.arange(1,len(means)+1)
	means_edit = means - x_inv[0]

	fig,ax=plt.subplots(figsize=[10,8])

	ax.plot(peak_no, a*peak_no+b, color='crimson',label=f'Slope = {np.round(a,2)} s')
	ax.scatter(peak_no, means_edit, color='lightpink')
	
	ax.set_xlabel('Peak Number', fontsize='xx-large')
	ax.tick_params(labelsize='xx-large')

	ax.set_ylabel("Time of peak in seconds", fontsize='xx-large') 
	#plt.title('', fontsize='xx-large')
	plt.legend(fontsize='xx-large')

	if not description:
		description = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	if not savedir:
		os.makedirs(os.path.expanduser('~/Gaussian_decomp_repository/plots'), exist_ok=True)
		savedir = os.path.expanduser('~/Gaussian_decomp_repository/plots')

	savefilename = 'line_plot_' + description + '.pdf'
	plt.savefig(os.path.join(savedir,savefilename))
	plt.close()
	return peak_no, a, b, means_edit


def gaussian_decomp_plot(time, counts, x_inv, mean_prediction_inv, std_inv, fit, fit_para, resid, yerr, time_earth_format = None, savedir=None, description=None):
	fig=plt.figure(figsize=[12,8], constrained_layout=False)
	outer_grid = fig.add_gridspec(1, 1, hspace=0.3, wspace=0.2)

	inner_grid = outer_grid[0,0].subgridspec(2,1, wspace=0.0, hspace=0.0, height_ratios=[1,0.3])
	fig0 = fig.add_subplot(inner_grid[0])
	fig1 = fig.add_subplot(inner_grid[1])
	
	if time_earth_format is None:
		time_arr = time

	else:
		time_arr = time_earth_format

	fig0.errorbar(time_arr, counts, yerr=yerr, label='STIX 25-76 keV',zorder=0)
#	fig0.plot(time_arr, mean_prediction_inv.reshape(-1), label='Mean prediction',zorder=1)

	fig0.plot(time_arr, fit, label='Linear combination of Gaussians', linewidth='3', zorder=3)
	for i in range(len(fit_para)):
	    fig0.plot(time_arr,fit_para[i],'--')
	fig0.legend(fontsize='xx-large')
	fig0.set_ylabel('Count Rate', fontsize='xx-large')

	fig1.plot(time_arr,resid/yerr,'x')
	fig1.plot(time_arr, [0 for i in range(len(time_arr))], linewidth='3')
	fig1.set_ylabel('Residual', fontsize='xx-large')
	fig1.set_ylim(-6,6)
	fig1.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
	fig0.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
	fig0.tick_params(labelsize='xx-large')
	fig1.tick_params(labelsize='xx-large')

	if not description:
		description = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	if not savedir:
		os.makedirs(os.path.expanduser('~/Gaussian_decomp_repository/plots'), exist_ok=True)
		savedir = os.path.expanduser('~/Gaussian_decomp_repository/plots')

	savefilename = 'gaussian_decomp_plot_' + description + '.pdf'
	plt.savefig(os.path.join(savedir,savefilename))
	plt.close()

	return np.std(resid/yerr)


