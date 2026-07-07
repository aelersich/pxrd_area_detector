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
	
	two_theta=np.loadtxt(data_folder + data_files[0], delimiter=",")[:,0]	
	
		
	cov_m = np.cov(data_r.T)
	w, v = linalg.eigh(cov_m)  #eigenvectors v[i] should already be normalized
	w_i = np.argsort(w)[::-1]
	w = w[w_i]
	v = v[:,w_i]
	v = v[:,:n_coeff].T	
	print(w[:n_coeff+2])	
	coeffs = np.zeros((n_coeff,data_r.shape[0]))
	for i in range(n_coeff):
		coeffs[i,:] = np.dot(data_r, v[i])
	
	return coeffs, v, two_theta

def plot_pca(data_folder, n_coeff, n_comp):
	from matplotlib import pyplot as plt
	coeffs, v, two_theta = pca_int_spectra(data_folder, n_coeff=n_coeff)
	
		
	x_delta = 5e-3
	x = np.arange(0, x_delta * coeffs.shape[1], x_delta)
	legend=[]
	for i in range(n_coeff):
		plt.plot(x,coeffs[i])
		legend.append(f"Component {i+1}")
	plt.legend(legend)
	plt.xlabel("Position (mm)")
	plt.ylabel("PCA component (arb.)")
	plt.show()
	
	plt.plot(two_theta,v.T[:,:n_comp])
	plt.xlabel("2theta (deg)")
	plt.ylabel("Intensity (arb.)")
	plt.legend(legend[:n_comp])
	plt.show()
	

if __name__ == "__main__":
	eomer_folder="data/HP-YBCO_poly_eomer_line_10keV_take2/"
	tigress_folder="data/HP-YBCO_poly_tigress_line_10keV/"
	
	plot_pca(eomer_folder, 3,2)
	plot_pca(tigress_folder, 5,3)	
