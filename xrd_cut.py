import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from PIL import Image


#detector center channel. For our measurements, this was on the 2theta = 40 ellipse
det_center_x = 526.0487
det_center_y = 292.6325

#Bragg angle of detector center channel, degrees
det_2theta = 40

#detector dimensions, mm
det_size_y = 169
det_size_x = 179

#detector angles, degrees
det_pitch = -0.18
det_yaw = 0.8

#distance from sample to detector, mm
det_dist = 187.6652 #150.5

#img size, pixels
img_size_y=981 #img.shape[1]
img_size_x=1043 #img.shape[0]

scale_y = det_size_y / img_size_y
scale_x = det_size_x / img_size_x

#distance of the start of a cut from the center, in mm
cut_offset = 0
cut_step = 0.5

data_folder = "data/HP-YBCO_poly_tigress_line_10keV/"


def get_cut(img, chi):
	chi *= np.pi/180


	# vector from center channel to pXRD center, mm
	cent_dist = det_dist  * np.sin(det_2theta * np.pi/180) / np.sin( (90 - det_pitch - det_2theta) * np.pi/180)
	cent_v = np.array([0,cent_dist,0])
	
	#array of vectors from pXRD center to points along cut, mm
	c_i = cut_offset
	c_f = cent_dist + det_center_y * scale_y
	c_x = np.arange(c_i,c_f,cut_step) * np.sin(chi)
	c_y = -np.arange(c_i,c_f,cut_step) * np.cos(chi)
	
	#re-center cut on center channel
	c_y = c_y + cent_v[1]	
	
	cut_v = np.stack([c_x,c_y,np.zeros_like(c_x)],axis=1)

	#rotate by det pitch and yaw
	pitch_r = det_pitch * np.pi/180
	rot_p = np.array([[1, 0, 0], [0, np.cos(pitch_r), -np.sin(pitch_r)], [0, np.sin(pitch_r), np.cos(pitch_r)]])
	cent_v = rot_p @ cent_v
	
	cut_v_temp = np.zeros_like(cut_v)
	for i, v_i in enumerate(cut_v):
		cut_v_temp[i,:] = rot_p @ v_i
	cut_v = cut_v_temp
	
	# vector from sample to detector center channel
	det_v = np.array([0,0,det_dist])
	
	a = cut_v + det_v
	b = cent_v + det_v

	v_l = lambda v: np.sqrt(np.vecdot(v,v))
	two_theta = np.arccos(np.dot(a,b.T) / (v_l(a) * v_l(b)))

	#interpolate image and get cut
	img_y = np.arange(0, img_size_y, 1)
	img_x = np.arange(0, img_size_x, 1)
	img_interp = RegularGridInterpolator((img_x, img_y), img, bounds_error = False, fill_value = -1)

	cut_int = img_interp((c_x / scale_x + det_center_x,c_y / scale_y + det_center_y))

	return two_theta * 180/np.pi, cut_int, c_x, c_y

def int_spectrum(img, chi_l=-45, chi_h=45,deg_spacing=0.05):
	chi = np.arange(chi_l, chi_h, 0.1)	
	int_2theta = np.arange(0,60,deg_spacing)

	integrand = np.zeros((chi.shape[0],int_2theta.shape[0]))
	for i, ch_i in enumerate(chi):
	    cut_2theta, cut_intensity, cut_x, cut_y = get_cut(img, ch_i)

	    dead_zone_index = cut_intensity<0
	    cut_intensity[dead_zone_index]=0
	    
	    integrand[i,:] = np.interp(int_2theta, cut_2theta, cut_intensity)

	#deal with background by adding a multiplier corresponding to number of non=zero pixels
	
	spectrum = np.trapezoid(integrand, axis=0)
	n_pixels = np.sum(integrand > 0, axis=0)
	
	n_pixels[n_pixels == 0] = 1
	spectrum /= n_pixels
	spectrum /= np.max(spectrum)

	return int_2theta, spectrum, integrand


#######
#
# Plotting functions
#
#######

def make_line_cut_plot(ax,data_folder,n_points, color='green', spacing=.02):
	#makes line cut plot. Run integrate_cuts.py first!
	import os
	from tqdm import tqdm
	
	plt.rcParams["text.usetex"] = True
	line_files = sorted([f for f in os.listdir(data_folder) if f.split(".")[1] == "txt"])
	file_spacing = np.round(len(line_files) / n_points)

	for i,f_i in enumerate(line_files):
		if i%file_spacing !=0:
			continue
		data = np.loadtxt(data_folder+f_i, delimiter=",")
		int_2theta = data[:,0]
		cut_integrated = data[:,1]
		
		mask = (int_2theta < 50) & (int_2theta >= 10)
		int_2theta = int_2theta[mask]
		cut_integrated = cut_integrated[mask]

		ax.plot(int_2theta,cut_integrated + i * spacing, 'tab:'+color)
	
def make_line_cut_plot_combined(n_points):
	import os
	from tqdm import tqdm
	eomer_folder="data/HP-YBCO_poly_eomer_line_10keV_take2/"
	tigress_folder="data/HP-YBCO_poly_tigress_line_10keV/"

	#plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,0.8,10)))

	plt.rcParams["text.usetex"] = True
	fig, ax = plt.subplots(1,2, figsize=(10,5))
	
	make_line_cut_plot(ax[0], eomer_folder, n_points, 'green')
	make_line_cut_plot(ax[1], tigress_folder, n_points, 'red') 
	ax[1].set(yticklabels=[])
	ax[1].tick_params(left=False)
	ax[0].set(yticklabels=[])
	ax[0].tick_params(left=False)

	fig.text(0.5, 0.04, '$2\\theta$ (degrees)', ha='center')
	plt.show()
	
def test_plot(chi=0):
	img_url = data_folder + "HP-YBCO_poly_tigress_line_10keV_00001.tif"
	img = np.asarray(Image.open(img_url))
	two_theta, cut_int, cut_x, cut_y = get_cut(img,chi)
	plt.imshow(img.T,cmap="hot",norm='log')
	plt.plot(cut_x/scale_x + det_center_x,cut_y/scale_y + det_center_y)
	plt.show()
	print(two_theta)
	plt.plot(two_theta, cut_int)
	plt.show()


def test_angle_cuts(chi_l=-45, chi_h=45):
	
	img_url = data_folder + "HP-YBCO_poly_tigress_line_10keV_00001.tif"
	img = np.asarray(Image.open(img_url))
	
	int_2theta, integrated_spec, stacked_spec = int_spectrum(img) 
	
	ref_file = "P241204.int"
	ref_counts=[]
	ref_2theta=[]
	with open(ref_file) as file:
		for line in file:
			two_theta, counts, _ = line.split()
			ref_counts.append(float(counts))
			ref_2theta.append(float(two_theta))
	ref_2theta = np.array(ref_2theta)
	ref_counts = np.array(ref_counts)

	A = np.max(ref_counts)
	ref_counts/=A

	fig, ax = plt.subplots(2,1)
	ax[0].imshow(stacked_spec,norm='log',extent=[int_2theta[0],int_2theta[-1],chi_h,chi_l])
	ax[0].set_aspect(stacked_spec.shape[1]/stacked_spec.shape[0])
	ax[0].set_ylabel("Azimuth (degrees)")
	ax[0].set_xlabel('$2\\theta$ (degrees)')
	ax[1].plot(int_2theta, integrated_spec)
	ax[1].plot(ref_2theta[ref_2theta<60], ref_counts[ref_2theta<60])
	
	plt.show()




