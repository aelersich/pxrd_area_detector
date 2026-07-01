import os
from multiprocessing import Pool
from tqdm import tqdm

class integrate_and_save:
	def __init__(self, data_folder):
		self.data_folder = data_folder

	def __call__(self, file):
		import numpy as np
		from PIL import Image
		from xrd_cut import int_spectrum
		file_integrated = file.split(".")[0]+".txt"
		img_url = self.data_folder + file
		img = np.asarray(Image.open(img_url))
		two_theta, counts, _ = int_spectrum(img)
		np.savetxt(self.data_folder + file_integrated, np.stack((two_theta, counts), axis=1))	
		return 0


def batch_integrate_spectra(data_folder,n_workers = 6):
	files = [f for f in os.listdir(data_folder) if f.split(".")[1] == "tif"]

	with Pool(n_workers) as p:
		i = integrate_and_save(data_folder)
		r = list(tqdm(p.imap(i,files),total = len(files)))


def integrate_all_data():
	eomer_folder="data/HP-YBCO_poly_eomer_line_10keV_take2/"
	tigress_folder="data/HP-YBCO_poly_tigress_line_10keV/"
	
	print("Integrating Eomer spectra...")
	batch_integrate_spectra(eomer_folder)
	print("Integrating Tigress spectra...")
	batch_integrate_spectra(tigress_folder)


if __name__=='__main__':
	integrate_all_data()
