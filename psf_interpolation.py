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
import tf_pixelwise_interpolation as tf_pixelwise
import poly1_interpolation as poly1
import poly_interpolation as poly
import psf_interpolation_utils as utils
import poly_sym_interpolate as poly_sym
import argparse
import os

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

# Some tuning switches
do_preprocess = True


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
    chip_avg_train_data=None

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
                data = pickle.load(pickle_file)
                self.psf_data = data['psf_data']
                self.chip_avg_train_data = data['chip_avg_train_data']
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
            #     TODO: This is a try to pre-process PSF.
            # avg_PSF = <48*48 pixel-wise mean of **train data**>
            # stdev_PSF = <48*48 pixel-wise mean of **train data**>
            # For all train/validate data:
            # <processed PSF> = <origin PSF> - avg_PSF
            if do_preprocess:
                chip_avg_train_datas = []
                for i in range(36):
                    chip_train_data = self.psf_data[i]['chip_train_data']
                    chip_avg_train_data = np.mean(np.array([data[4] for data in chip_train_data]), axis=0)
                    chip_avg_train_datas.append(chip_avg_train_data)
                chip_avg_train_data = np.mean(np.array(chip_avg_train_datas), axis=0)
                print("write chip avg")
                self.chip_avg_train_data = chip_avg_train_data
                for i in range(36):
                    chip_info = self.psf_data[i]
                    chip_train_data = chip_info['chip_train_data']
                    chip_validate_data = chip_info['chip_validate_data']
                    chip_info['chip_train_data'] = [data[:4] + [data[4] - chip_avg_train_data] for data in chip_train_data]
                    chip_info['chip_validate_data'] = [data[:4] + [data[4] - chip_avg_train_data] for data in chip_validate_data]
                    # print(len(chip_train_data[0]))
                    # chip_validate_data = [psf_data - chip_avg_train_data for psf_data in chip_info['chip_validate_data']]
                    # if i == 0:
                    #     print("Min and max for chip 1 train data.")
                    #     psf_min = np.min(np.array([data[4] for data in chip_train_data]), axis=0)
                    #     psf_max = np.max(np.array([data[4] for data in chip_train_data]), axis=0)
                    #     print(psf_min)
                    #     utils.plot_stamp(psf_min)
                    #     print(psf_max)
                    #     utils.plot_stamp(psf_max)
                    #     utils.plot_stamp(self.psf_data[i]['chip_train_data'][0][4])
            # save psf data to cache
            pickle.dump({'psf_data': self.psf_data, 'chip_avg_train_data': self.chip_avg_train_data},
                        open('assets/cache/{}_{}/psf_data.p'.format(self.region, self.exp_num), 'wb'))

    def examine(self, method='tf_psfwise', part='train', hidden1=36, hidden2=144, learning_rate=0.1, max_steps=4000, batch_size=100):
        part_name = part
        part = part_map[part]

        color = []
        if method.startswith('poly') and (not method.startswith('poly_sym')):
            order = int(method[4:])
            if order == 1:
                method_path = 'poly/' + method
            elif order > 1:
                method_path = 'poly/poly_{}'.format(str(order))
            method_disp = method.title()
        elif method == 'tf_psfwise':
            method_path = 'tf_psfwise/l2_lr{}_ms{}_h1.{}_h2.{}_bs{}'.format(learning_rate, max_steps, hidden1,
                                                                            hidden2, batch_size)
            method_disp = 'tf_psfwise_l2_lr{}_ms{}_h1.{}_h2.{}_bs{}'.format(learning_rate, max_steps, hidden1,
                                                                            hidden2, batch_size).title()
        elif method.startswith('poly_sym'):
            order = int(method[8:])
            method_path = 'poly_sym/poly_sym{}'.format(str(order))
            method_disp = method.title()
        path_prefix = 'assets/predictions/{}_{}/{}/'.format(self.region, self.exp_num, method_path)
        info_file_path = path_prefix + 'info.dat'
        fits_file_path = path_prefix + 'predictions.fits'
        met_pred = []
        met_coord = []
        with open(info_file_path, 'r') as info_file:
            info_file.readline()
            met_coord = [info.split()[2:4] for info in info_file.readlines()]
            psf_count = len(met_coord)
        with fits.open(fits_file_path) as fits_file:
            psf_power_data = fits_file[0].data
            # print(type(psf_power_data))
            # exit()
            for k in range(psf_count):
                met_pred.append(psf_power_data[((k // 15) * 48):((k // 15 + 1) * 48),
                                ((k % 15) * 48):((k % 15 + 1) * 48)].copy())

        psf_num = 0
        for i in range(36):
            temp_data = self.psf_data[i][part]
            color = [get_ellp(data[4]) for data in temp_data] if not do_preprocess else [
                get_ellp(data[4] + self.chip_avg_train_data) for data in temp_data]
            color = np.array(color)
            origin_max_num = np.argmax(color)
            print(temp_data[origin_max_num][2:4])
            print(met_coord[origin_max_num+psf_num])
            utils.plot_stamp_comparison(stamp_data_1=temp_data[origin_max_num][4],
                                        stamp_data_2=met_pred[origin_max_num+psf_num]-self.chip_avg_train_data,
                                        title_1='Origin train pre-processed PSF on chip{}'.format(i+1),
                                        title_2=method_disp)
            # utils.plot_stamp(temp_data[origin_max_num][4]+self.chip_avg_train_data)
            # exit()
            psf_num += len(color)


    def collect_origin_data(self, tag='train'):
        origin_psf = []
        fits_info = []
        part_name = 'chip_{}_data'.format(tag)
        for chip_psf_data in self.psf_data:
            origin_psf += [data[4].ravel() for data in chip_psf_data[part_name]] if not do_preprocess else [data[4].ravel()+self.chip_avg_train_data.ravel() for data in chip_psf_data[part_name]]
            fits_info += [data[0:4] for data in chip_psf_data[part_name]]
        origin_psf = np.array(origin_psf)
        result_dir = 'assets/predictions/{}_{}/origin/{}/'.format(self.region, self.exp_num, tag)
        utils.write_predictions(result_dir, origin_psf, fits_info, method=tag)

    def plot_ellipticities(self, tag='exposure', method='tf_psfwise', part='train', cache=False, hidden1=36,
                           hidden2=144, learning_rate=0.1, max_steps=4000, batch_size=100):
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
                    ellip += [get_ellipticity(data[4]) for data in temp_data] if not do_preprocess else [get_ellipticity(data[4]+self.chip_avg_train_data) for data in temp_data]
                    color += [get_ellp(data[4]) for data in temp_data] if not do_preprocess else [get_ellp(data[4]+self.chip_avg_train_data) for data in temp_data]

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
                if method.startswith('poly') and (not method.startswith('poly_sym')):
                    order = int(method[4:])
                    if order == 1:
                        method_path = 'poly/' + method
                    elif order > 1:
                        method_path = 'poly/poly_{}'.format(str(order))
                elif method.startswith('poly_sym'):
                    order = int(method[8:])
                    method_path = 'poly_sym/poly_sym{}'.format(str(order))
                elif method == 'tf_psfwise':
                    method_path = 'tf_psfwise/l2_lr{}_ms{}_h1.{}_h2.{}_bs{}'.format(learning_rate, max_steps, hidden1,
                                                                                     hidden2, batch_size)
                path_prefix = 'assets/predictions/{}_{}/{}/'.format(self.region, self.exp_num, method_path)
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
            plt.title('Original {} PSF Ellip-dist.'.format(part_name.title()))
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
            if method == 'tf_psfwise' or method == 'tf_pixelwise':
                ellip_dist_dir = 'assets/predictions/ellip_dist/{}/'.format(method)
                if not os.path.exists(ellip_dist_dir):
                    os.makedirs(ellip_dist_dir)
            plt.savefig('assets/predictions/ellip_dist/{}.png'.format(method_path))
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

    def interpolate(self, method='poly1', **kwargs):
        if method == 'poly1':
            poly1.poly1_interpolation(self)
            return
        if method == 'tf_psfwise':
            tf_psfwise.tf_psfwise_interpolation(self, **kwargs)
        if method == 'tf_pixelwise':
            tf_pixelwise.tf_pixelwise_interpolation(self, **kwargs)
        if method.startswith('poly_sym'):
            poly_sym.poly_sym_interpolate(self, int(method[8:]))
            return
        if method.startswith('poly'):
            poly.poly_interpolation(self, int(method[4:]))


    def predict(self, method='poly1', **kwargs):
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
            tf_psfwise.predict(self, coord, fits_info, **kwargs)
        elif method == 'tf_pixelwise':
            tf_pixelwise.predict(self, coord, fits_info)
        elif method.startswith('poly_sym'):
            poly_sym.predict(self, coord, fits_info, int(method[8:]))
        elif method.startswith('poly'):
            poly.predict(self, coord, fits_info, int(method[4:]))
        else:
            return None


if __name__ == '__main__':
    my_psf = PSF_interpolation()

    # my_psf.interpolate(method='poly1')
    # my_psf.interpolate(method='poly_sym1')
    # my_psf.interpolate(method='poly_sym6')
    # print('poly_inter with scipy')
    # for i in range(2, 10):
    #     t0=time.time()
    #     my_psf.predict(method='poly{}'.format(i))
    #     t1=time.time()
    #     my_psf.plot_ellipticities(method='poly{}'.format(i))
    #     print('time:{} order:{}\n'.format(t1-t0, i))
        # my_psf.predict(method='poly_sym{}'.format(i))

    # my_psf.interpolate(method='poly3')
    # my_psf.predict(method='poly3')

    my_psf.examine('poly5')


    # for i in range(10, 15):
    #     my_psf.plot_ellipticities(method='poly_sym{}'.format(i))


    # my_psf.interpolate(method='tf_pixelwise', learning_rate=0.01, hidden1=3, hidden2=6, pixel_num=1152)

    # for hidden1, hidden2, learning_rate in ((36, 144, 1), (36, 144, 0.1), (36, 144, 0.01)):
    # for hidden1, hidden2, learning_rate in ((64, 256, 1), (128, 512, 0.1)):
    #     my_psf.interpolate(method='tf_psfwise', hidden1=hidden1, hidden2=hidden2, learning_rate=learning_rate,
    #                        max_steps=4000, batch_size=100)
    #     my_psf.predict(method='tf_psfwise', hidden1=hidden1, hidden2=hidden2, learning_rate=learning_rate,
    #                    max_steps=4000, batch_size=100)
    #     my_psf.plot_ellipticities(method='tf_psfwise', hidden1=hidden1, hidden2=hidden2, learning_rate=learning_rate,
    #                               max_steps=4000, batch_size=100)

    # for tag in ('train', 'validate'):
    #     my_psf.collect_origin_data(tag=tag)


    # for i in range(2, 15):
    #     my_psf.predict(method='poly'+str(i))
    # for i in range(4, 15):
    #     my_psf.plot_ellipticities(tag='exposure', method='poly'+str(i), part='train', cache=False)
    # my_psf.predict('tf_psfwise')
    # my_psf.interpolate(method='tf_psfwise', hidden1=32, hidden2=144, learning_rate=10)
    # my_psf.interpolate(method='poly1')
    # for hidden1, hidden2, learning_rate in (
    #         (32, 128, 10), (32, 128, 1), (32, 128, 0.1),
    #         (36, 144, 10), (36, 144, 1), (36, 144, 0.1),
    #         (144, 36, 10), (144, 36, 1), (144, 36, 0.1)):
    #     my_psf.tf_psfwise_interpolation(hidden1=hidden1, hidden2=hidden2, learning_rate=learning_rate)
