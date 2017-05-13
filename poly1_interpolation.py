import numpy as np
import pickle
import psf_interpolation_utils as utils

def poly1_interpolation(self):
    '''
    apply linear interpolation
    save matrix a, b, c to cal_info['poly1']
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
    to_interpolate = False
    try:
        with open('assets/cache/{}_{}/cal_info.p'.format(self.region, self.exp_num),
                  'rb') as pickle_file:
            self.cal_info = pickle.load(pickle_file)
            if not ("poly1" in self.cal_info):
                # TODO: repeat the algor. below
                to_interpolate = True
    except FileNotFoundError:
        # get train/validate set coordinates
        to_interpolate = True
    if to_interpolate:
        train_coord = []
        t_z = []
        for chip_psf_data in self.psf_data:
            train_coord += [data[2:4] for data in chip_psf_data['chip_train_data']]
            t_z += [data[4].ravel() for data in chip_psf_data['chip_train_data']]
        train_coord = np.array(train_coord)
        t_z = np.array(t_z)
        # calculate linear interpolation coefficient matrices on train psf data
        t_x = train_coord[:, 0]
        t_y = train_coord[:, 1]
        coef_x_x = np.sum(t_x ** 2)
        coef_y_y = np.sum(t_y ** 2)
        coef_x_y = np.sum(t_x * t_y)
        coef_y_z, coef_z_x = np.zeros(t_z[0].shape), np.zeros(t_z[0].shape)
        for i in range(len(t_x)):
            coef_y_z += t_y[i] * t_z[i]
            coef_z_x += t_x[i] * t_z[i]
        coef_x = np.sum(t_x)
        coef_y = np.sum(t_y)
        coef_z = np.sum(t_z, axis=0)
        inv_coef = np.linalg.inv(np.array([[coef_x_x, coef_x_y, coef_x],
                                           [coef_x_y, coef_y_y, coef_y],
                                           [coef_x, coef_y, len(t_x)]]))
        opt_A, opt_B, opt_C = np.zeros(t_z[0].shape), np.zeros(t_z[0].shape), np.zeros(t_z[0].shape)
        coef_rhs = (coef_z_x, coef_y_z, coef_z)
        for i in range(3):
            opt_A += inv_coef[0, i] * coef_rhs[i]
            opt_B += inv_coef[1, i] * coef_rhs[i]
            opt_C += inv_coef[2, i] * coef_rhs[i]
        self.cal_info['poly1'] = [opt_A, opt_B, opt_C]
        # Cache plot data to file
        pickle.dump(self.cal_info, open('assets/cache/{}_{}/cal_info.p'.
                                        format(self.region, self.exp_num), 'wb'))
    else:
        opt_A, opt_B, opt_C = self.cal_info['poly1']
    # Calculate mse on train/validate set
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
        psf_predictions = np.array([the_coord[0] * opt_A + the_coord[1] * opt_B + opt_C for the_coord in coord])
        loss = np.sum((psf_labels - psf_predictions) ** 2)
        data_sets[tag] = [coord, psf_labels]
        print('{} Data Eval:'.format(tag.title()))
        print('  Num examples: %d  Total loss: %0.09f  Mean loss @ 1: %0.09f' %
              (len(coord), np.asscalar(loss), np.asscalar(loss / len(coord))))


def predict(self, coord, fits_info):
    if not ("poly1" in self.cal_info):
        poly1_interpolation(self)
    opt_A, opt_B, opt_C = self.cal_info['poly1']
    psf_predictions = np.array([the_coord[0] * opt_A + the_coord[1] * opt_B + opt_C for the_coord in coord])
    result_dir = 'assets/predictions/poly1/'
    utils.write_predictions(result_dir, psf_predictions, fits_info)



