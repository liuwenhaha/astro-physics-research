import numpy as np
import pickle
import psf_interpolation_utils as utils
import time

def poly_interpolation(self, order):
    '''
    apply polynomial <order>_th interpolation exposure-wise
    save coefficient matrix to cal_info['poly_<order>']
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
    poly_name = 'poly_{}'.format(str(order))
    try:
        with open('assets/cache/{}_{}/cal_info.p'.format(self.region, self.exp_num),
                  'rb') as pickle_file:
            self.cal_info = pickle.load(pickle_file)
            if not (poly_name in self.cal_info):
                # TODO: Change back to True
                to_interpolate = False
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
        # calculate polynomial interpolation coefficient matrices on train psf data
        t_x = train_coord[:, 0]
        t_y = train_coord[:, 1]
        r_tot = 0
        PIXEL_NUM = t_z.shape[1]
        TERM_NUM = int((order+1)*(order+2)/2)
        coeffs = np.zeros((PIXEL_NUM, TERM_NUM))
        print('order: {}'.format(order))
        for i in range(PIXEL_NUM):
            print('\r{}%'.format(i/2304*100), end="")
            coeff_term, r_sub = utils.poly_scipy_fit(t_x, t_y, t_z[:, i], order)
            # coeff_term, r_sub = utils.poly_fit(t_x, t_y, t_z[:, i], order)
            coeffs[i] = coeff_term
            r_tot += r_sub

        coeffs = coeffs.T.copy()
        # for i in range(TERM_NUM):
        #     utils.plot_stamp(coeffs[i].reshape((48,48)))
        self.cal_info[poly_name] = coeffs
        # Cache plot data to file
        # pickle.dump(self.cal_info, open('assets/cache/{}_{}/cal_info.p'.
        #                                 format(self.region, self.exp_num), 'wb'))
    else:
        coeffs = self.cal_info[poly_name]
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
        psf_predictions = np.array([utils.poly_val_all(the_coord[0], the_coord[1], coeffs, order) for the_coord in coord])
        loss = np.sum((psf_labels - psf_predictions) ** 2)
        data_sets[tag] = [coord, psf_labels]
        print('{} Data Eval:'.format(tag.title()))
        print('  Num examples: %d  Total loss: %0.09f  Mean loss @ 1: %0.09f' %
              (len(coord), np.asscalar(loss), np.asscalar(loss / len(coord))))


def predict(self, coord, fits_info, order):
    poly_name = 'poly_{}'.format(str(order))
    if not (poly_name in self.cal_info):
        poly_interpolation(self, order)
    coeffs = self.cal_info[poly_name]
    psf_predictions = np.array([utils.poly_val_all(the_coord[0], the_coord[1], coeffs, order) for the_coord in coord])
    result_dir = 'assets/predictions/{}_{}/poly/{}/'.format(self.region, self.exp_num, poly_name)
    utils.write_predictions(result_dir, psf_predictions, fits_info, method=poly_name)



