# read star PSF from .fits file

from astropy.io import fits
import numpy as np

image_file = 'assets/star_power831555_01.fits'
label_file = 'assets/star_info831555_01.dat'
label_f = open(label_file, 'r')
labels = label_f.readlines()[1:]
label_f.close()
labels = [x.split()[0:2] for x in labels]

star_power = fits.open(image_file)
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
star_power_data = star_power[0].data
testPSF = star_power_data[0:48, 0:48]
# 48 is not included!
def get_ellipticity(PSF):
    # assume PSF is 48*48, center of light (23.5, 23.5)
    normalization = PSF.sum()
    q11 = 0
    q12 = 0
    q21 = 0
    q22 = 0
    for i in range(48):
        for j in range(48):
            q11 += PSF[i, j]*(i-23.5)*(i-23.5)
            q12 += PSF[i, j]*(i-23.5)*(j-23.5)
            q21 += PSF[i, j]*(j-23.5)*(i-23.5)
            q22 += PSF[i, j]*(j-23.5)*(j-23.5)
    q11 /= normalization
    q12 /= normalization
    q21 /= normalization
    q21 /= normalization
    chi1 = (q11 - q22) / (q11 + q22)
    chi2 = q12 / (q11 + q22)
    return
