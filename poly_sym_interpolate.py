from sympy import *
import numpy as np
import psf_interpolation_utils as utils


# Some tuning switches
do_preprocess = True


x, y, z, n, c0 = symbols('x, y, z, n, c0')


def poly_sym_fun_sub(order):
    #     coeff_sub is to keep track for coefficient subscripts
    coeff_sub = int((order + 1) * order / 2)
    result = "c{}*x**{} + ".format(coeff_sub, order)
    coeff_sub += 1
    for i in range(1, order):
        x_coef = order - i
        y_coef = i
        x_term = "x**{}".format(x_coef)
        y_term = "y**{}".format(y_coef)
        if x_coef == 1:
            x_term = "x"
        if y_coef == 1:
            y_term = "y"
        result += "c{}*{}*{} + ".format(coeff_sub, x_term, y_term)
        coeff_sub += 1
    result += "c{}*y**{}".format(coeff_sub, order)
    return result


def poly_sym_maker(order):
    diff_eq = 'c0 + c1*x + c2*y'
    for i in range(2, order + 1):
        sub_poly_sym_eq = poly_sym_fun_sub(i)
        diff_eq += ' + {}'.format(sub_poly_sym_eq)
    diff_eq += ' - z'
    TERM_NUM = int((order + 2) * (order + 1) / 2)
    init = 'x, y, z, n'
    for i in range(TERM_NUM):
        init += ', c{}'.format(i)
    init = init + ' = ' + 'symbols(\'{}\')'.format(init)
    #     print(init)
    exec(init, globals())

    coeff_mat = []
    coeff_vect = []
    for i in range(TERM_NUM):
        sub_coeffs = []
        temp_eq = eval('expand(diff(({})**2, c{}, 1)/2)'.format(diff_eq, i))
        if i == 0:
            temp_eq = (temp_eq - c0) + c0 * n
        for i in range(TERM_NUM):
            exec('sub_coeffs.append(temp_eq.coeff(c{}))'.format(i))
        # print(temp_eq)
        coeff_mat.append(sub_coeffs)
        coeff_vect.append(-temp_eq.coeff(z))
    return coeff_mat, coeff_vect


def poly_sym_interpolate(self, order):
    TERM_NUM = int((order + 2) * (order + 1) / 2)
    train_coord = []
    t_z = []
    for chip_psf_data in self.psf_data:
        train_coord += [data[2:4] for data in chip_psf_data['chip_train_data']]
        t_z += [data[4].ravel() for data in chip_psf_data['chip_train_data']]

    # for dev
    # N = 20
    # PIXEL_NUM = 2
    # train_coord = [[i, i + 0.2] for i in range(N)]
    # t_z = [np.ones((PIXEL_NUM, PIXEL_NUM)).ravel() * i for i in range(N)]

    train_coord = np.array(train_coord)
    t_z = np.array(t_z)
    # get analytic solution from symbolic approach
    coeff_mat, coeff_vect = poly_sym_maker(order)
    coeff_mat_eval, coeff_vect_eval = np.zeros((TERM_NUM, TERM_NUM)), []
    # calculate linear interpolation coefficient matrices on train psf data
    t_x = train_coord[:, 0]
    t_y = train_coord[:, 1]
    t_n = np.ones(t_x.shape)
    r_tot = 0
    PIXEL_NUM = t_z.shape[1]

    for i in range(TERM_NUM):
        for j in range(TERM_NUM):
            term_func = lambdify((x, y, n), coeff_mat[i][j], 'numpy')
            coeff_mat_eval[i, j] = np.sum(term_func(t_x, t_y, t_n))

    for i in range(TERM_NUM):
        term_func = lambdify((x, y), coeff_vect[i], 'numpy')
        term_x_y = term_func(t_x, t_y)
        if isinstance(term_x_y, int):
            coeff_vect_eval.append(np.sum(t_z * term_x_y, axis=0))
        else:
            coeff_vect_eval.append(np.sum(t_z * term_x_y[:, np.newaxis], axis=0))
    coeff_vect_eval = np.array(coeff_vect_eval)
    inv_coeff_mat_eval = np.linalg.inv(coeff_mat_eval)
    coeff_vect_result = np.dot(inv_coeff_mat_eval, coeff_vect_eval)
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
        psf_predictions = np.array(
            [utils.poly_val_all(the_coord[0], the_coord[1], coeff_vect_result, order) for the_coord in coord])
        loss = np.sum((psf_labels - psf_predictions) ** 2)
        data_sets[tag] = [coord, psf_labels]
        print('{} Data Eval:'.format(tag.title()))
        print('  Num examples: %d  Total loss: %0.09f  Mean loss @ 1: %0.09f' %
              (len(coord), np.asscalar(loss), np.asscalar(loss / len(coord))))


def predict(self, coord, fits_info, order):
    poly_name = 'poly_{}'.format(str(order))
    if not (poly_name in self.cal_info):
        poly_sym_interpolate(self, order)
    coeffs = self.cal_info[poly_name]
    psf_predictions = np.array([utils.poly_val_all(the_coord[0], the_coord[1], coeffs, order) for the_coord in coord])
    result_dir = 'assets/predictions/{}_{}/poly_sym/{}/'.format(self.region, self.exp_num, poly_name)
    utils.write_predictions(result_dir, psf_predictions, fits_info, method=poly_name)
