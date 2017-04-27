'''
Read PSF .fits file
Apply basic linear interpolation
Plot coefficient matrix
-------------------------------------
Divide input PSF to two branch: interpolate/validate
Get residual from the two and plot
'''

from astropy.io import fits
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import time

# Some constants
image_file = 'assets/star_power831555_01.fits'
# image_file = 'assets/star_831555_01.fits'
label_file = 'assets/star_info831555_01.dat'
# For calculating ellipticity
# Assume PSF is 48*48, center of light (23.5, 23.5)
[psf_mesh_x, psf_mesh_y] = np.mgrid[0:48, 0:48]
psf_mesh_x = psf_mesh_x - 23.5
psf_mesh_y = psf_mesh_y - 23.5




# star_power -> HDUList (Header Data Unit)
# HDUObj -> .header .data
# Eg for .header
#     SIMPLE  =                    T / file does conform to FITS standard
#     BITPIX  =                  -32 / number of bits per data pixel
#     NAXIS   =                    2 / number of data axes
#     NAXIS1  =                  720 / length of data axis 1
#     NAXIS2  =                  336 / length of data axis 2
#     COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
#     COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H
# .data return numpy ndarray object


def get_ellipticity(PSF, tag=0):
    q11 = np.sum(PSF*psf_mesh_x*psf_mesh_x)
    q12 = np.sum(PSF*psf_mesh_x*psf_mesh_y)
    q22 = np.sum(PSF*psf_mesh_y*psf_mesh_y)
    epsl1 = (q11 - q22) / (q11 + q22)
    epsl2 = 2 * q12 / (q11 + q22)
    epsl = math.sqrt(epsl1 * epsl1 + epsl2 * epsl2)
    print("No.{} e {}".format(tag, epsl))
    # print("No.{} e1 {} e2 {}".format(tag, epsl1, epsl2))
    # print("No.{} x1 {} x2 {}".format(tag, epsl1/(1+math.sqrt(1-epsl*epsl)), epsl2/(1+math.sqrt(1-epsl*epsl))))
    return [epsl1 / epsl, epsl2 / epsl]


def plot_stamp(stamp_data, plot_axes_extend=(0,48,0,48)):
    fig = plt.figure(1,(6,6))
    fig.clf()
    ax = fig.add_subplot(1,1,1)

    # draw locatable axes in easy way
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    im = ax.imshow(stamp_data, extent=plot_axes_extend, interpolation="none")

    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    for tl in ax_cb.get_yticklabels():
        tl.set_visible(False)
    ax_cb.yaxis.tick_right()

    plt.draw()
    plt.show()

def plot_ellipticities(coord, ellip):
    plt.figure(figsize=(4.2, 9.4), dpi=80)
    plt.xlim(0, 2100)
    plt.ylim(0, 4700)
    plt.xticks(np.arange(0,2100,500))
    plt.yticks(np.arange(0,4700,500))
    plt.axes().set_aspect('equal', 'datalim')
    num = len(coord)
    ellip_bar_len = 50
    ellip_bar_wid = 2

    color_max = 0
    color_min = 1
    ellip_bar_colors = []
    for n in range(num):
        bar = np.array([[coord[n][0] + 15 * ellip[n][0], coord[n][1] + 15 * ellip[n][1]],
                        [coord[n][0] - 15 * ellip[n][0], coord[n][1] - 15 * ellip[n][1]]])
        if ellip[n][0] > ellip[n][1]:
            color = ellip[n][0] / ellip[n][1]
        else:
            color = ellip[n][1] / ellip[n][0]
        if color > color_max:
            color_max = color
        if color < color_min:
            color_min = color
        ellip_bar_colors.append((bar, color))
    print(color_min, color_max)

    ellip_vector = [np.array([[coord[n][0] + ellip_bar_len * ellip[n][0],
                               coord[n][1] + ellip_bar_len * ellip[n][1]],
                              [coord[n][0] - ellip_bar_len * ellip[n][0],
                               coord[n][1] - ellip_bar_len * ellip[n][1]]])
                    for n in range(num)]
    for i in range(num):
        vertices = ellip_vector[i]
        color = 'r'
        plt.plot(vertices[:, 0], vertices[:, 1], color=color, linewidth=ellip_bar_wid)
    plt.show()

# find the range for ellipticity
def explosure_ellipticity_range(exp_num="831555", region="w2m0m0"):
    for i in 36:
        image_file = 'assets/explosure_{0}_{1}/star_power{0}_{2}.fits'.format(exp_num, region, '{0:02d}'.format(i))
        label_file = 'assets/explosure_{0}_{1}/star_info831555_01.dat'.format(exp_num, region, '{0:02d}'.format(i))
    label_f = open(label_file, 'r')
    labels = label_f.readlines()[1:]
    label_f.close()
    labels = [x.split()[1:3] for x in labels]
    labels = [[eval(x[0]), eval(x[1])] for x in labels]
    star_number = len(labels)
    # no. x y
    # x 0 - 2043
    # y 0 - 4606
    star_power = fits.open(image_file)
    star_power_data = star_power[0].data
    # t0=time.time()
    ellipticities = [
        get_ellipticity(star_power_data[((i // 15) * 48):((i // 15 + 1) * 48), ((i % 15) * 48):((i % 15 + 1) * 48)], i)
        for i in range(star_number)]




class PSF_interpolation:
    # psf data [[x,y,psf_numpy_48_48],]
    psf_data = []
    train_psf_data = []
    validate_psf_data = []
    star_number = 0
    def __init__(self, fits_file_path, info_file_path):
        '''
        read in info file
        read in .fits file
        chop up into psf_stamps
        save psf_data [[x,y,psf_numpy_48_48],]
        '''
        # TODO: Use np.loadtxt to rewrite
        with open(info_file_path, 'r') as info_file:
            info_file.readline()
            for line in info_file.readlines():
                # TODO: Check if the data is float
                raw_record = line.split('\s+')
                self.psf_data.append([float(raw_record[1]), float(raw_record[2])])
        self.star_number = len(self.psf_data)
        with open(fits_file_path, 'r') as fits_file:
            psf_power = fits.open(fits_file)
            psf_power_data = psf_power[0].data
            print(psf_power_data.size)
            for i in range(len(self.psf_data)):
                self.psf_data[i].append(psf_power_data[((i / 15) * 48):((i / 15 + 1) * 48),
                                        ((i % 15) * 48):((i % 15 + 1) * 48)])
        random.shuffle(psf_power_data)
        self.train_psf_data = psf_power_data[:int(star_number/2)]
        self.validate_psf_data = psf_power_data[int(star_number/2):]



    def linear_interpolation(self):
        '''
        apply linear interpolation
        plot coefficient matrix a, b
        :return:
        '''
        # get train/validate set coordinates
        train_coordinates = np.array([psf_data[:2] for psf_data in self.train_psf_data])
        validate_coordinates = np.array([psf_data[:2] for psf_data in self.validate_psf_data])
        # perform linear interpolation on train psf data
        t_x = train_coordinates[:, 0]
        t_y = train_coordinates[:, 1]
        t_z = np.array([psf_data[2] for psf_data in self.train_psf_data])
        coef_x_x = np.sum(t_x**2)
        coef_y_y = np.sum(t_y**2)
        coef_x_y = np.sum(t_x*t_y)
        coef_z_x = np.sum(t_x*t_y)
        pass

    def linear_interpolation_show_residual(self):
        '''
        divide psf_stamps into interpolation/validation
        :return:
        '''
        pass

    def plot_stamp(self, stamp_data, plot_axes_extend=(0,48,0,48)):
        fig = plt.figure(1,(6,6))
        fig.clf()
        ax = fig.add_subplot(1,1,1)

        # draw locatable axes in easy way
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        fig = ax.get_figure()
        fig.add_axes(ax_cb)
        im = ax.imshow(stamp_data, extent=plot_axes_extend, interpolation="none")

        plt.colorbar(im, cax=ax_cb)
        ax_cb.yaxis.tick_right()
        for tl in ax_cb.get_yticklabels():
            tl.set_visible(False)
        ax_cb.yaxis.tick_right()

        plt.draw()
        plt.show()





if __name__ == '__main__':
    label_f = open(label_file, 'r')
    labels = label_f.readlines()[1:]
    label_f.close()
    labels = [x.split()[1:3] for x in labels]
    labels = [[eval(x[0]), eval(x[1])] for x in labels]
    star_number = len(labels)
    # no. x y
    # x 0 - 2043
    # y 0 - 4606
    star_power = fits.open(image_file)
    star_power_data = star_power[0].data
    # t0=time.time()
    ellipticities = [
        get_ellipticity(star_power_data[((i // 15) * 48):((i // 15 + 1) * 48), ((i % 15) * 48):((i % 15 + 1) * 48)], i)
        for i in range(star_number)]
    i=0
    plot_stamp(star_power_data[((i // 15) * 48):((i // 15 + 1) * 48), ((i % 15) * 48):((i % 15 + 1) * 48)])
    # plot_ellipticities(labels, ellipticities)
    print(ellipticities)
