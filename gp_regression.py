from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn import preprocessing
import numpy as np

def rescale(time_arr,counts_arr):
	scaler_x = preprocessing.StandardScaler().fit(time_arr.reshape(-1,1))
	X_scaled = scaler_x.transform(time_arr.reshape(-1,1))
	scaler_y = preprocessing.StandardScaler().fit(counts_arr.reshape(-1,1))
	y_scaled = scaler_y.transform(counts_arr.reshape(-1,1))
	return X_scaled, y_scaled, scaler_x, scaler_y

def grid_search(X_scaled,y_scaled, params):
	gp = GaussianProcessRegressor()
	grid=GridSearchCV(estimator=gp, param_grid=params, scoring='neg_mean_squared_error').fit(X_scaled.reshape(-1,1), y_scaled.reshape(-1,1))
	return grid

def train_set(x_scaled, y_scaled):
	rng = np.random.RandomState(1)
	training_indices = rng.choice(np.arange(y_scaled.size), size=int(0.8*(y_scaled.size)), replace=False)
	X_train, y_train = x_scaled[training_indices].reshape(-1,1), y_scaled[training_indices].reshape(-1,1) 
	return X_train, y_train


def gp_fit(x_train,y_train, scaler_x, scaler_y, X_scaled, alpha, A, l):
	kernel = ConstantKernel(A, constant_value_bounds="fixed") * RBF(length_scale=l,length_scale_bounds="fixed") 
	gaussian_process = GaussianProcessRegressor(kernel=kernel,alpha=alpha,random_state=2)
	gaussian_process.fit(x_train,y_train)
	gaussian_process.kernel_ 

	resampled_arr=np.arange(np.min(X_scaled),np.max(X_scaled),0.01)

	mean_prediction, std_prediction = gaussian_process.predict(resampled_arr.reshape(-1,1), return_std=True)
	
	mean_prediction_inv = scaler_y.inverse_transform(mean_prediction)
	x_inv = scaler_x.inverse_transform(resampled_arr.reshape(-1,1))
	std_inv = (np.max(mean_prediction_inv)/np.max(mean_prediction))*std_prediction.reshape(-1,1)

	return x_inv, mean_prediction_inv, std_inv
