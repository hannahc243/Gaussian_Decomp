import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

def line_fit(popt, x_inv):
	means= popt[::3]
	peak_no = np.arange(1,len(means)+1)
	means_edit = means - x_inv[0]
	a, b = np.polyfit(peak_no, means_edit, 1)
	c, d = np.polyfit(peak_no[:9], means_edit[:9], 1)
	e, f = np.polyfit(peak_no[9:], means_edit[9:], 1)
	spear = spearmanr(peak_no, means_edit)
	pearson = pearsonr(peak_no,means_edit)

	fig,ax=plt.subplots(figsize=[12,6])

	ax.plot(peak_no, a*peak_no+b, color='crimson',label=f'Slope = {np.round(a,2)} s')

	ax.scatter(peak_no[:9], means_edit[:9], color='orange', alpha=0.4)
	ax.scatter(peak_no[9:], means_edit[9:],color='lightblue')
	ax.plot(peak_no[:9], c*peak_no[:9]+d,  color='orange',alpha=1, label=f'Slope = {np.round(a,2)} s')
	ax.plot(peak_no[9:], e*peak_no[9:]+f, color='darkblue',label=f'Slope = {np.round(c,2)} s')

	ax.set_xlabel('Peak Number', fontsize='xx-large')
	ax.set_title('Line Fit', fontsize='xx-large')
	ax.tick_params(labelsize='large')

	ax.set_ylabel(f"Time of peak (s)", fontsize='xx-large')
	ax.legend(fontsize='xx-large')
	plt.show()
	return peak_no, a, b, means_edit, c, d , e, f, spear, pearson


def gaussian_decomp(time, counts, x_inv, mean_prediction_inv, std_inv, fit, fit_para):
	fig,ax=plt.subplots(figsize=[10,8])

	ax.plot(np.asarray(time), np.asarray(counts), label='Original time profile')
	ax.errorbar(x_inv.reshape(-1), mean_prediction_inv.reshape(-1), yerr=std_inv.reshape(-1), label='Mean prediction')
	ax.plot(np.asarray(time), fit, label='Linear combination of Gaussians')
	for i in range(len(fit_para)):
	    ax.plot(time,fit_para[i],'--')
	ax.legend()
	ax.set_ylabel('Count Rate')
	ax.set_xlabel('Time (s)')
	plt.show()