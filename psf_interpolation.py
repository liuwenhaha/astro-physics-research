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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import random
from psf_interpolation_utils import get_ellp, get_ellipticity
import time
import tf_psfwise_interpolation as tf_psfwise
import poly1_interpolation as poly1
import poly_interpolation as poly
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
explosure_ellip_bar_len = 0.008
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
    # cal_info -> {'poly1': {'a': <numpy_array_for_a>,
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

    def plot_ellipticities(self, tag='exposure', method='tf_psfwise', part='train', cache=False):
        '''
        calculate and plot ellipticity distribution
        :param tag: 'exposure' or 1 to 36
        :param part: 'all' or 'train' or 'validate' or 'linear'
        :return:
        '''
        part_name = part
        part = part_map[part]
        if method:
            part_name = 'train'
            part = part_map[part_name]
        coord = []
        ellip = []
        color = []
        if tag == 'exposure':
            # Try to load cached plot data
            try:
                with open('assets/cache/{}_{}/{}_{}.p'.format(self.region, self.exp_num, tag, part_name),
                          'rb') as pickle_file:
                    pickle_data = pickle.load(pickle_file)
                    # TODO: Cache for diff methods
                    part_data = pickle_data[part_name]
                    ellip_vector = part_data['ellip_vector']
                    coord = part_data['coord']
                    color = part_data['color']
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
                coord = np.array(coord)
                color = np.array(color)
                ellip_vector = np.array(ellip_vector)

                # Cache plot data to file
                if cache:
                    pickle_data = {part_name: {'coord': coord, 'ellip': ellip, 'color': color,
                                               'ellip_vector': ellip_vector}}
                    pickle.dump(pickle_data, open('assets/cache/{}_{}/{}_{}.p'.
                                                  format(self.region, self.exp_num, tag, part_name), 'wb'))

                psf_num = len(ellip_vector)
                (x_min, y_min), (x_max, y_max) = np.min(coord, axis=0), np.max(coord, axis=0)
                color_min, color_max = np.min(color), np.max(color)
            # Pick out ellipticity less than a range
            # ellip_vector = ellip_vector[color < 0.15]
            # color = color[color < 0.15]

            if method:
                met_coord = []
                met_pred = []
                # self.predict(method)
                if method.startswith('poly'):
                    order = int(method[4:])
                    if order>1:
                        method = 'poly_{}'.format(str(order))
                path_prefix = 'assets/predictions/{}_{}/{}/'.format(self.region, self.exp_num, method)
                info_file_path = path_prefix + 'info.dat'
                fits_file_path = path_prefix + 'predictions.fits'
                with open(info_file_path, 'r') as info_file:
                    info_file.readline()
                    for line in info_file.readlines():
                        raw_record = line.split()
                        met_coord.append([float(raw_record[2]), float(raw_record[3])])
                with fits.open(fits_file_path) as fits_file:
                    psf_power_data = fits_file[0].data
                    # print(type(psf_power_data))
                    # exit()
                    for k in range(len(met_coord)):
                        met_pred.append(psf_power_data[((k // 15) * 48):((k // 15 + 1) * 48),
                                        ((k % 15) * 48):((k % 15 + 1) * 48)].copy())
                met_ellip = [get_ellipticity(data) for data in met_pred]
                met_color = [get_ellp(data) for data in met_pred]
                pred_num = len(met_coord)
                met_ellip_vector = [np.array([[met_coord[n][0] + explosure_ellip_bar_len * met_ellip[n][0],
                                               met_coord[n][1] + explosure_ellip_bar_len * met_ellip[n][1]],
                                              [met_coord[n][0] - explosure_ellip_bar_len * met_ellip[n][0],
                                               met_coord[n][1] - explosure_ellip_bar_len * met_ellip[n][1]]])
                                    for n in range(pred_num)]
                met_color = np.array(met_color)
                met_ellip_vector = np.array(met_ellip_vector)
                met_psf_num = len(met_ellip_vector)
                (met_x_min, met_y_min), (met_x_max, met_y_max) = np.min(met_coord, axis=0), np.max(met_coord, axis=0)
                met_color_min, met_color_max = np.min(met_color), np.max(met_color)


            # Plot the distributions

            plt.figure(figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 4])
            ax_orig = plt.subplot(gs[0])
            plt.title('Original Validate PSF Ellip-dist.')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(np.arange(x_min, x_max, 0.2))
            plt.yticks(np.arange(y_min, y_max, 0.2))
            norm = plt.Normalize(vmin=color_min, vmax=color_max)
            for i in range(psf_num):
                vertices = ellip_vector[i]
                cl = cmap(norm(color[i]))
                plt.plot(vertices[:, 0], vertices[:, 1], color=cl, linewidth=explosure_ellip_bar_wid)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            plt.colorbar(sm)

            ax_met = plt.subplot(gs[1])
            plt.title('{} PSF Ellip-dist.'.format(method))
            plt.xlim(met_x_min, met_x_max)
            plt.ylim(met_y_min, met_y_max)
            plt.xticks(np.arange(met_x_min, met_x_max, 0.2))
            plt.yticks(np.arange(met_y_min, met_y_max, 0.2))
            met_norm = plt.Normalize(vmin=met_color_min, vmax=met_color_max)
            for i in range(met_psf_num):
                vertices = met_ellip_vector[i]
                met_cl = cmap(met_norm(met_color[i]))
                plt.plot(vertices[:, 0], vertices[:, 1], color=met_cl, linewidth=explosure_ellip_bar_wid)
            met_sm = plt.cm.ScalarMappable(cmap=cmap, norm=met_norm)
            met_sm._A = []
            plt.setp(ax_met.get_yticklabels(), visible=False)
            plt.colorbar(met_sm)
            plt.tight_layout()

            # plt.show()
            plt.savefig('assets/predictions/ellip_dist/{}.png'.format(method))
            # plt.set_aspect('equal', 'datalim')
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

    def interpolate(self, method='poly1', **para_dict):
        if method == 'poly1':
            poly1.poly1_interpolation(self)
        if method == 'tf_psfwise':
            tf_psfwise.tf_psfwise_interpolation(self, **para_dict)
        if method.startswith('poly'):
            poly.poly_interpolation(self, int(method[4:]))


    def predict(self, method='poly1'):
        '''
        use different methods to predict the galaxy psf
        :return:
        '''
        # Currently using validate dataset
        coord = []
        fits_info = []
        for chip_psf_data in self.psf_data:
            coord += [data[2:4] for data in chip_psf_data['chip_train_data']]
            fits_info += [data[0:4] for data in chip_psf_data['chip_train_data']]
        coord = np.array(coord)
        if method == 'poly1':
            poly1.predict(self, coord, fits_info)
        elif method == 'tf_psfwise':
            tf_psfwise.predict(self, coord, fits_info)
        elif method.startswith('poly'):
            poly.predict(self, coord, fits_info, int(method[4:]))
        else:
            return None


if __name__ == '__main__':
    my_psf = PSF_interpolation()
    # for i in range(2, 15):
    #     my_psf.predict(method='poly'+str(i))
    for i in range(4, 15):
        my_psf.plot_ellipticities(tag='exposure', method='poly'+str(i), part='train', cache=False)
    # my_psf.predict('tf_psfwise')
    # my_psf.interpolate(method='tf_psfwise', hidden1=32, hidden2=144, learning_rate=10)
    # my_psf.interpolate(method='poly1')
    # for hidden1, hidden2, learning_rate in (
    #         (32, 128, 10), (32, 128, 1), (32, 128, 0.1),
    #         (36, 144, 10), (36, 144, 1), (36, 144, 0.1),
    #         (144, 36, 10), (144, 36, 1), (144, 36, 0.1)):
    #     my_psf.tf_psfwise_interpolation(hidden1=hidden1, hidden2=hidden2, learning_rate=learning_rate)
