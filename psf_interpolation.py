'''
Read PSF .fits file
Apply 1st polynomial, tf_psfwise,  interpolation
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
import pickle
import random
from psf_interpolation_utils import get_ellp, get_ellipticity
import time
import tf_psfwise_interpolation as tf_psfwise
import argparse


# Some constants
image_file = 'assets/star_power831555_01.fits'
label_file = 'assets/star_info831555_01.dat'
part_map = {'all': 'chip_data', 'train': 'chip_train_data', 'validate': 'chip_validate_data'}
# For calculating ellipticity
# Assume PSF is 48*48, center of light (23.5, 23.5)
[psf_mesh_x, psf_mesh_y] = np.mgrid[0:48, 0:48]
psf_mesh_x = psf_mesh_x - 24
psf_mesh_y = psf_mesh_y - 24
# For plotting
chip_ellip_bar_len = 50
chip_ellip_bar_wid = 2
explosure_ellip_bar_len = 0.08
explosure_ellip_bar_wid = 1.6
# cmap = plt.get_cmap('Blues')
cmap = plt.get_cmap('YlOrRd')


class PSF_interpolation:
    # per exposure
    # psf_data -> [{'chip_no': 1,
    #               'chip_data': [[x,y,RA,Dec,psf_numpy_48_48], ...],
    #               'chip_train_data': <view_of_chip_data>,
    #               'chip_validate_data': <view_of_chip_data>,},
    #              ...]
    # plot info cache to file tag_part
    #     ellip_vector: data to draw ellip dist. bar
    #     coord: coord for each bar
    #     color: magnitude of each ellip
    # cal_info -> {'poly_1': {'a': <numpy_array_for_a>,
    #                         'b': <numpy_array_for_b>, ...
    #                         'predictions': [<numpy_array_psf>, ...]},
    #              'tf_psfwise': {'model': <model for prediction>, ...
    #                             'predictions': [<numpy_array_psf>, ...]},
    #              ...}
    # train_data_ratio -> partition of train data

    psf_data = []
    cal_info = {}
    train_data_ratio = 0.5
    # train_psf_data = []
    # validate_psf_data = []
    star_number = 0
    exp_num = ''
    region = ''

    def __init__(self, exp_num="831555", region="w2m0m0"):
        '''
        read in info file
        read in .fits file
        chop up into psf_stamps
        save psf_data as
        psf_data -> [{'chip_no': 1,
                     'chip_data': [[x,y,RA,Dec,psf_numpy_48_48], ...],
                     'chip_train_data': <view_of_chip_data>,
                     'chip_validate_data': <view_of_chip_data>,},
                    ...]
        '''
        self.region = region
        self.exp_num = exp_num
        # Try to load psf data from cache
        # This retains the train/validate data
        try:
            with open('assets/cache/{}_{}/psf_data.p'.format(self.region, self.exp_num),
                      'rb') as pickle_file:
                self.psf_data = pickle.load(pickle_file)
        except FileNotFoundError:
            for i in range(1, 37):
                fits_file_path = 'assets/explosure_{0}_{1}/star_power{0}_{2}.fits'.format(exp_num, region,
                                                                                          '{0:02d}'.format(i))
                info_file_path = 'assets/explosure_{0}_{1}/star_shape{0}_{2}.dat'.format(exp_num, region,
                                                                                         '{0:02d}'.format(i))
                chip_info = {'chip_no': i}
                chip_data = []
                with open(info_file_path, 'r') as info_file:
                    info_file.readline()
                    for line in info_file.readlines():
                        raw_record = line.split()
                        chip_data.append([float(raw_record[0]), float(raw_record[1]),
                                          float(raw_record[2]), float(raw_record[3])])
                with fits.open(fits_file_path) as fits_file:
                    psf_power_data = fits_file[0].data
                    # print(type(psf_power_data))
                    # exit()
                    for k in range(len(chip_data)):
                        chip_data[k].append(psf_power_data[((k // 15) * 48):((k // 15 + 1) * 48),
                                            ((k % 15) * 48):((k % 15 + 1) * 48)].copy())
                random.shuffle(chip_data)
                star_number = len(chip_data)
                chip_info['chip_train_data'] = chip_data[:int(star_number * self.train_data_ratio)]
                chip_info['chip_validate_data'] = chip_data[int(star_number * self.train_data_ratio):]
                chip_info['chip_data'] = chip_data
                self.psf_data.append(chip_info)
            # save psf data to cache
            pickle.dump(self.psf_data, open('assets/cache/{}_{}/psf_data.p'.
                                            format(self.region, self.exp_num), 'wb'))

    def plot_ellipticities(self, tag='exposure', part='all'):
        '''
        calculate and plot ellipticity distribution
        :param tag: 'exposure' or 1 to 36
        :param part: 'all' or 'train' or 'validate' or 'linear'
        :return:
        '''
        part_name = part
        part = part_map[part]
        coord = []
        ellip = []
        color = []
        if tag == 'exposure':
            # Try to load cached plot data
            try:
                with open('assets/cache/{}_{}/{}_{}.p'.format(self.region, self.exp_num, tag, part_name),
                          'rb') as pickle_file:
                    pickle_data = pickle.load(pickle_file)
                    ellip_vector = pickle_data['ellip_vector']
                    coord = pickle_data['coord']
                    color = pickle_data['color']
            except FileNotFoundError:
                for i in range(36):
                    temp_data = self.psf_data[i][part]
                    coord += [[data[2], data[3]] for data in temp_data]
                    ellip += [get_ellipticity(data[4]) for data in temp_data]
                    color += [get_ellp(data[4]) for data in temp_data]
                star_num = len(coord)
                ellip_vector = [np.array([[coord[n][0] + explosure_ellip_bar_len * ellip[n][0],
                                           coord[n][1] + explosure_ellip_bar_len * ellip[n][1]],
                                          [coord[n][0] - explosure_ellip_bar_len * ellip[n][0],
                                           coord[n][1] - explosure_ellip_bar_len * ellip[n][1]]])
                                for n in range(star_num)]
                color = np.array(color)
                ellip_vector = np.array(ellip_vector)

                # Cache plot data to file
                pickle_data = {'coord': coord, 'ellip': ellip, 'color': color, 'ellip_vector': ellip_vector}
                pickle.dump(pickle_data, open('assets/cache/{}_{}/{}_{}.p'.
                                              format(self.region, self.exp_num, tag, part_name), 'wb'))
            # Pick out ellipticity less than a range
            # ellip_vector = ellip_vector[color < 0.15]
            # color = color[color < 0.15]

            star_num = len(ellip_vector)
            x_coord = [tcoord[0] for tcoord in coord]
            y_coord = [tcoord[1] for tcoord in coord]
            x_min, x_max = min(x_coord), max(x_coord)
            y_min, y_max = min(y_coord), max(y_coord)
            print(x_max-x_min)
            print(y_max-y_min)
            color_min, color_max = min(color), max(color)

            # Plot the distributions
            plt.figure(figsize=(10, 10))
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(np.arange(x_min, x_max, 0.2))
            plt.yticks(np.arange(y_min, y_max, 0.2))
            norm = plt.Normalize(vmin=color_min, vmax=color_max)

            for i in range(star_num):
                vertices = ellip_vector[i]
                cl = cmap(norm(color[i]))
                plt.plot(vertices[:, 0], vertices[:, 1], color=cl, linewidth=explosure_ellip_bar_wid)

            # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            # sm._A = []
            # plt.colorbar(sm)
            plt.show()
            plt.set_aspect('equal', 'datalim')
            return
        # TODO: Finish chip plot
        # TODO: save the chip data to cal_info, retrieve to avoid recalculation
        if tag in range(1, 37):
            tag -= 1
            plt.figure(figsize=(4.2, 9.4), dpi=80)
            plt.xlim(0, 2100)
            plt.ylim(0, 4700)
            plt.xticks(np.arange(0, 2100, 500))
            plt.yticks(np.arange(0, 4700, 500))
            plt.axes().set_aspect('equal', 'datalim')
            num = len(coord)
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

            ellip_vector = [np.array([[coord[n][0] + chip_ellip_bar_len * ellip[n][0],
                                       coord[n][1] + chip_ellip_bar_len * ellip[n][1]],
                                      [coord[n][0] - chip_ellip_bar_len * ellip[n][0],
                                       coord[n][1] - chip_ellip_bar_len * ellip[n][1]]])
                            for n in range(num)]

            for i in range(num):
                vertices = ellip_vector[i]
                color = 'r'
                plt.plot(vertices[:, 0], vertices[:, 1], color=color, linewidth=chip_ellip_bar_wid)
            plt.show()

    def poly_1_interpolation(self):
        '''
        apply linear interpolation
        save matrix a, b, c to cal_info['poly_1']
        plot coefficient matrix a, b
        with cache support
        :return:
        '''

        # psf_data -> [{'chip_no': 1,
        #               'chip_data': [[x,y,RA,Dec,psf_numpy_48_48], ...],
        #               'chip_train_data': <view_of_chip_data>,
        #               'chip_validate_data': <view_of_chip_data>,},
        #              ...]

        # Try to load cached plot data
        try:
            with open('assets/cache/{}_{}/cal_info.p'.format(self.region, self.exp_num),
                      'rb') as pickle_file:
                self.cal_info = pickle.load(pickle_file)
                if not ("poly_1" in self.cal_info):
                    # TODO: repeat the algor. below
                    pass
        except FileNotFoundError:
            # get train/validate set coordinates
            train_coordinates = np.array([chip_psf_data['chip_train_data'][2:4] for chip_psf_data in self.psf_data])
            # perform linear interpolation on train psf data
            t_x = train_coordinates[:, 0]
            t_y = train_coordinates[:, 1]
            t_z = np.array([chip_psf_data['chip_train_data'][4] for chip_psf_data in self.psf_data])
            coef_x_x = np.sum(t_x ** 2)
            coef_y_y = np.sum(t_y ** 2)
            coef_x_y = np.sum(t_x * t_y)
            coef_z_x = np.sum(t_x * t_y)
            # TODO: Finish poly 1 interpolation
            self.cal_info['poly_1'] = []
            # Cache plot data to file
            pickle.dump(self.cal_info, open('assets/cache/{}_{}/cal_info.p'.
                                          format(self.region, self.exp_num), 'wb'))
            pass

    def evaluate(self, methods):
        '''
        evaluate pixel-wise MSE over validate set for each methods
        :return:
        '''
        if not methods:
            methods = ['poly_1', 'tf_psfwise']
        pass

    def tf_psfwise_interpolation(self):
        '''
        train neural network define in tf_psfwise_interpolation.py
        save the trained model to cal_info
        with cache support
        :return:
        '''


        # psf_data -> [{'chip_no': 1,
        #               'chip_data': [[x,y,RA,Dec,psf_numpy_48_48], ...],
        #               'chip_train_data': <view_of_chip_data>,
        #               'chip_validate_data': <view_of_chip_data>,},
        #              ...]

        # TODO: prepare datasets
        data_sets = {}
        for tag in ('train', 'validate'):
            coord = []
            psf_labels = []
            chip_data_name = 'chip_{}_data'.format(tag)
            for chip_psf_data in self.psf_data:
                coord += [data[2:4] for data in chip_psf_data[chip_data_name]]
                psf_labels += [data[4].ravel() for data in chip_psf_data[chip_data_name]]
            coord = np.array(coord)
            psf_labels = np.array(psf_labels)
            data_sets[tag] = tf_psfwise.DataSet(coord, psf_labels)
        learning_rate = 0.01
        max_steps = 200000
        # hidden unit for pixel: 3-12
        # hidden unit for psf: 91-100
        hidden1 = 128
        hidden2 = 32
        batch_size = 100
        tf_psfwise.execute(learning_rate=learning_rate, max_steps=max_steps, hidden1=hidden1,
                           hidden2=hidden2, batch_size=batch_size,
                           log_dir='assets/log/{}_{}/lr{}_ms{}_h1.{}_h2.{}_bs{}'.format(self.region, self.exp_num,
                                                                                        learning_rate, max_steps,
                                                                                        hidden1, hidden2, batch_size),
                           datasets=data_sets)
        pass

    def plot_stamp(self, stamp_data, plot_axes_extend=(0, 48, 0, 48)):
        fig = plt.figure(1, (6, 6))
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)

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
    my_psf = PSF_interpolation()
    my_psf.tf_psfwise_interpolation()
    # my_psf.plot_ellipticities()
