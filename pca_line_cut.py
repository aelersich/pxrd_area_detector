import numpy as np
from scipy import linalg
import os

def pca_int_spectra(data_folder, n_coeff = 10):
	#pca of integrated spectra along line cut
	#data has shape (n,m): n is lenth of spectra (in 2theta), m is number of points along line	
	data_files = [f for f in os.listdir(data_folder) if f.split(".")[1] == "txt"]
	data_files = sorted(data_files)
	n_files = len(data_files)
	n_angles = len(np.loadtxt(data_folder+data_files[0], delimiter=","))
	
	data_r = np.zeros((n_files, n_angles))
	for i,f_i in enumerate(data_files):
		data_r[i] = np.loadtxt(data_folder+f_i, delimiter=",")[:,1]

	cov_m = np.cov(data_r.T)
	w, v = linalg.eigh(cov_m)
	w_i = np.argsort(w)[::-1]
	w = w[w_i]
	v = v[:,w_i]
	v = v[:,:n_coeff].T	
	coeffs = np.zeros((n_coeff,data_r.shape[0]))
	for i in range(n_coeff):
		coeffs[i,:] = np.dot(data_r, v[i])
	
	return coeffs


if __name__ == "__main__":
	eomer_folder="data/HP-YBCO_poly_eomer_line_10keV_take2/"
	tigress_folder="data/HP-YBCO_poly_tigress_line_10keV/"

	from matplotlib import pyplot as plt
	n_coeff=3	
	coeffs = pca_int_spectra(eomer_folder, n_coeff=n_coeff)
	
	for i in range(n_coeff):
		plt.plot(coeffs[i])
	plt.show()
	
	n_coeff=10	
	coeffs = pca_int_spectra(tigress_folder, n_coeff=n_coeff)
	
	for i in range(n_coeff):
		plt.plot(coeffs[i])
	plt.show()
		
