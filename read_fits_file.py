# read star PSF from .fits file

from astropy.io import fits
import numpy as np
import math
import matplotlib.pyplot as plt

image_file = 'assets/star_power831555_01.fits'
label_file = 'assets/star_info831555_01.dat'


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

# 48 is not included!
def get_ellipticity(PSF):
    # assume PSF is 48*48, center of light (23.5, 23.5)
    PSF.shape
    normalization = PSF.sum()
    q11 = 0
    q12 = 0
    q21 = 0
    q22 = 0
    for k in range(48):
        for j in range(48):
            q11 += PSF[k, j] * (k - 24) * (k - 24)
            q12 += PSF[k, j] * (k - 24) * (j - 24)
            q21 += PSF[k, j] * (j - 24) * (k - 24)
            q22 += PSF[k, j] * (j - 24) * (j - 24)
    # q11 /= normalization
    # q12 /= normalization
    # q21 /= normalization
    # q21 /= normalization
    chi1 = (q11 - q22) / (q11 + q22)
    chi2 = 2 * q12 / (q11 + q22)
    chi = math.sqrt(chi1 * chi1 + chi2 * chi2)
    return [chi1 / chi, chi2 / chi]


def plot_ellipticities(coord, ellip):
    num = len(coord)
    ellip_vector = [np.array([[coord[n][0] + 15 * ellip[n][0], coord[n][1] + 15 * ellip[n][1]],
                              [coord[n][0] - 15 * ellip[n][0], coord[n][1] - 15 * ellip[n][1]]])
                    for n in range(num)]
    for i in range(num):
        vertices = ellip_vector[i]
        color = 'r'
        plt.plot(vertices[:,0], vertices[:,1], color=color)

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
    ellipticities = [
        get_ellipticity(star_power_data[((i / 15) * 48):((i / 15 + 1) * 48), ((i % 15) * 48):((i % 15 + 1) * 48)])
        for i in range(star_number)]
    plot_ellipticities(labels, ellipticities)
    print ellipticities
