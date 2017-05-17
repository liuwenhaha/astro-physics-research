from astropy.io import fits
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import least_squares


[psf_mesh_x, psf_mesh_y] = np.mgrid[0:48, 0:48]
psf_mesh_x = psf_mesh_x - 24
psf_mesh_y = psf_mesh_y - 24

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


def write_predictions(result_dir, psf_predictions, fits_info):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    info_file_path = result_dir + 'info.dat'
    fits_file_path = result_dir + 'predictions.fits'
    with open(info_file_path, 'w') as info_file:
        info_file.write('x y RA Dec\n')
        info_file.writelines("%s\n" % "     ".join(map(str, l)) for l in fits_info)
        info_file.close()
    num_pred = psf_predictions.shape[0]
    fits_w = 15
    fits_h = math.ceil(num_pred/fits_w)
    fits_image = np.zeros((fits_h*48, fits_w*48))

    for k in range(num_pred):
        fits_image[((k // 15) * 48):((k // 15 + 1) * 48),
                   ((k % 15) * 48):((k % 15 + 1) * 48)] = psf_predictions[k].reshape((48,48))
    hdu = fits.PrimaryHDU(fits_image)
    hdu.writeto(fits_file_path, overwrite=True)
    print('poly1 predictions saved to ' + fits_file_path)


def get_ellipticity(PSF):
    q11 = np.sum(PSF * psf_mesh_x * psf_mesh_x)
    q12 = np.sum(PSF * psf_mesh_x * psf_mesh_y)
    q22 = np.sum(PSF * psf_mesh_y * psf_mesh_y)
    epsl1 = (q11 - q22) / (q11 + q22)
    epsl2 = 2 * q12 / (q11 + q22)
    epsl = math.sqrt(epsl1 * epsl1 + epsl2 * epsl2)
    # print("No.{} e {}".format(tag, epsl))
    # print("No.{} e1 {} e2 {}".format(tag, epsl1, epsl2))
    # print("No.{} x1 {} x2 {}".format(tag, epsl1/(1+math.sqrt(1-epsl*epsl)), epsl2/(1+math.sqrt(1-epsl*epsl))))
    return [epsl1 / epsl, epsl2 / epsl]


def get_ellp(PSF):
    q11 = np.sum(PSF * psf_mesh_x * psf_mesh_x)
    q12 = np.sum(PSF * psf_mesh_x * psf_mesh_y)
    q22 = np.sum(PSF * psf_mesh_y * psf_mesh_y)
    epsl1 = (q11 - q22) / (q11 + q22)
    epsl2 = 2 * q12 / (q11 + q22)
    epsl = math.sqrt(epsl1 * epsl1 + epsl2 * epsl2)
    return epsl


def plot_stamp(stamp_data, plot_axes_extend=(0, 48, 0, 48)):
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


def plot_ellipticities(coord, ellip):
    plt.figure(figsize=(4.2, 9.4), dpi=80)
    plt.xlim(0, 2100)
    plt.ylim(0, 4700)
    plt.xticks(np.arange(0, 2100, 500))
    plt.yticks(np.arange(0, 4700, 500))
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
    all_ellipticities = []
    for j in range(36):
        i = j + 1
        image_file = 'assets/explosure_{0}_{1}/star_power{0}_{2}.fits'.format(exp_num, region, '{0:02d}'.format(i))
        label_file = 'assets/explosure_{0}_{1}/star_shape{0}_{2}.dat'.format(exp_num, region, '{0:02d}'.format(i))
        label_f = open(label_file, 'r')
        labels = label_f.readlines()[1:]
        label_f.close()
        labels = [x.split()[1:3] for x in labels]
        labels = [[eval(x[0]), eval(x[1])] for x in labels]
        star_number = len(labels)
        star_power = fits.open(image_file)
        star_power_data = star_power[0].data
        # t0=time.time()
        all_ellipticities += [
            get_ellp(star_power_data[((i // 15) * 48):((i // 15 + 1) * 48), ((i % 15) * 48):((i % 15 + 1) * 48)])
            for i in range(star_number)]
    print('Exp No.{0} ellip range is {1} to {2}'.format(exp_num, min(all_ellipticities), max(all_ellipticities)))
    print('Number of psf is {}'.format(len(all_ellipticities)))


def poly_val_sub(x, y, coeff, order):
    result = 0
    for i in range(0, order+1):
        x_ord = order - i
        y_ord = i
        result += (coeff[i]*(x**x_ord)*(y**y_ord))
    return result


def poly_val_all(x, y, coeff, order):
    result = coeff[0]
    if order == 0:
        return result
    for i in range(1, order+1):
        result += poly_val_sub(x, y, coeff[int((i+1)*i/2): int((i+2)*(i+1)/2)], i)
    return result


def poly_term_sub(x, y, order):
    result = []
    for i in range(0, order+1):
        x_ord = order - i
        y_ord = i
        result.append((x**x_ord)*(y**y_ord))
    return result


def poly_term_all(x, y, order):
    result = [x*0+1]
    if order == 0:
        return np.array(result)
    for i in range(1, order+1):
        result += poly_term_sub(x, y, i)
    return np.array(result)


def poly_fit(x, y, z, order):
    A = poly_term_all(x, y, order).T
    coeff, r, rank, s = np.linalg.lstsq(A, z)
    return coeff, r


def poly_fun_sub(order):
    start = int((order+1)*order/2)
    result = "x[{}]*t[0]**{} + ".format(start, order)
    start += 1
    for i in range(1, order):
        x_coef = order - i
        y_coef = i
        x_term = "t[0]**{}".format(x_coef)
        y_term = "t[1]**{}".format(y_coef)
        if x_coef == 1:
            x_term = "t[0]"
        if y_coef == 1:
            y_term = "t[1]"
        result += "x[{}]*{}*{} + ".format(start, x_term, y_term)
        start += 1
    result += "x[{}]*t[1]**{}".format(start, order)
    return result


def poly_maker(order):
    result = 'lambda x, t, y: x[0] + x[1]*t[0] + x[2]*t[1]'
    if order == 1:
        return eval(result)
    for i in range(2, order+1):
        result += ' + {}'.format(poly_fun_sub(i))
    return eval(result)


def poly_scipy_fit(x, y, z, order):
    t_train = np.array([x, y])
    fun = poly_maker(order)
    TERM_NUM = int((order + 2) * (order + 1) / 2)
    x0 = np.random.rand(TERM_NUM)
    res_lsq = least_squares(fun, x0, args=(t_train, z))
    coeff, cost = res_lsq.x, res_lsq.cost
    return coeff, cost


if __name__ == '__main__':
    # label_f = open(label_file, 'r')
    # labels = label_f.readlines()[1:]
    # label_f.close()
    # labels = [x.split()[1:3] for x in labels]
    # labels = [[eval(x[0]), eval(x[1])] for x in labels]
    # star_number = len(labels)
    # no. x y
    # x 0 - 2043
    # y 0 - 4606
    # star_power = fits.open(image_file)
    # star_power_data = star_power[0].data
    # ellipticities = [
    #     get_ellipticity(star_power_data[((i // 15) * 48):((i // 15 + 1) * 48), ((i % 15) * 48):((i % 15 + 1) * 48)], i)
    #     for i in range(star_number)]
    # i=0
    # plot_stamp(star_power_data[((i // 15) * 48):((i // 15 + 1) * 48), ((i % 15) * 48):((i % 15 + 1) * 48)])
    # plot_ellipticities(labels, ellipticities)
    # print(ellipticities)
    # explosure_ellipticity_range()

    # t0 = time.time()
    # t1 = time.time()
    # print('Time is {}'.format(t1 - t0))
    pass
