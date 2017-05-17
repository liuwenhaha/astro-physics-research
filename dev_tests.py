import numpy as np
from scipy.optimize import least_squares
import psf_interpolation_utils as utils

# x = np.linspace(0, 1, 20)
# y = np.linspace(0, 1, 20)
# X, Y = np.meshgrid(x, y, copy=False)
# Z = X**2 + Y**2 + np.random.rand(*X.shape)*0.01
# Z = X**2 + Y**2

# X = X.flatten()
# Y = Y.flatten()
# Z = Z.flatten()

# A = np.array([X*0+1, X, Y, X**2, X*Y, Y**2]).T
# B = Z.flatten()
#
# coeff, r, rank, s = np.linalg.lstsq(A, B)
# for order in range(2,15):
#     coeff, r = utils.poly_fit(X, Y, Z, order=order)
#     if order == 2:
#         print(coeff)
#     print('Order: {}, Residual: {}, x**2 {}, y**2 {}'.format(order, r, coeff[3], coeff[5]))
# [ 0.00322333  0.00269038  0.00641761  0.99971714  0.00693432 -0.00915853 0.99413242  0.01365575 -0.01370687]


x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x, y, copy=False)
Z = X**2 + Y**2 + np.random.rand(*X.shape)*0.01
Z = X**2 + Y**2

X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

t_train = np.array([X, Y])
# t_train = t_train.T.copy()
y_train = Z

def poly2(x, t, y):
    return x[0] + x[1]*t[0] + x[2]*t[1] + x[3]*t[0]**2 + x[4]*t[0]*t[1] + x[5]*t[1]**2 - y

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


order = 4
fun = poly_maker(4)
TERM_NUM = int((order+2)*(order+1)/2)
x0=np.random.rand(TERM_NUM)
# x0=np.random.rand(6)
# fun = poly_maker(2)
res_lsq = least_squares(fun, x0, args=(t_train, y_train))
coeff, cost = res_lsq.x, res_lsq.cost
print(coeff)
print(cost)
# print()
# print(res_lsq.cost)
coeff, r = utils.poly_fit(X, Y, Z, order)
print(coeff)
print(r)