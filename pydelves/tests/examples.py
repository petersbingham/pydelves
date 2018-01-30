import os
import sys
basedir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,basedir+'/../..')

from pydelves import *

import numpy.testing as testing
import numpy as np
import numpy.linalg as la
from scipy.integrate import ode
import scipy.constants as consts

import matplotlib.pyplot as plt
import time

def trig(mode=default_mode):
    '''
    Make a square of length just under 5*pi. Find the roots of sine.
    '''
    N=5000
    f = lambda z: np.sin(z)
    fp = lambda z: np.cos(z)
    x_cent = 0.
    y_cent = 0.
    width = 5.*np.pi+1e-5
    height = 5.*np.pi+1e-5

    return droots(f,fp,x_cent,y_cent,width,height,mode=mode)

def get_root_bounds(roots):
    x_lmt = [None,None]
    y_lmt = [None,None]
    for root in roots:
        if x_lmt[0] is None or x_lmt[0]>root.real:
            x_lmt[0] = root.real
        if x_lmt[1] is None or x_lmt[1]<root.real:
            x_lmt[1] = root.real
        if y_lmt[0] is None or y_lmt[0]>root.imag:
            y_lmt[0] = root.imag
        if y_lmt[1] is None or y_lmt[1]<root.imag:
            y_lmt[1] = root.imag
    return x_lmt, y_lmt 

def get_coeff(polyOrder):
    coeff = []
    for n in range(polyOrder):
        coeff.append((n+1)*1.0+(n+1)*1.0j)
    return np.poly1d(coeff)

def get_poly_fun(polyOrder):
    poly = np.poly1d(get_coeff(polyOrder))
    return lambda z: poly(z)

def poly_roots(polyOrder,N=default_N,max_steps=default_max_steps,
               mode=default_mode,printPolys=False):
    '''
    Find roots of polynomials with increasing powers.
    Compares with roots returned from np.roots.
    '''
    coeff = get_coeff(polyOrder)
    poly = np.poly1d(coeff)
    poly_diff = np.polyder(poly)

    roots_numpy = np.roots(coeff)
    bnds = get_root_bounds(roots_numpy)

    if printPolys:
        print poly
        print poly_diff

    f = lambda z: poly(z)
    fp = lambda z: poly_diff(z)
    width = (bnds[0][1]-bnds[0][0])/2.
    height = (bnds[1][1]-bnds[1][0])/2.
    x_cent = bnds[0][0] + width
    y_cent = bnds[1][0] + height
    width += 0.1
    height += 0.1

    status,roots_delves=droots(f,fp,x_cent,y_cent,width,height,N,max_steps,mode)
    return status,roots_delves,roots_numpy

def wilk_f(x):
    val = 1
    for i in range(1,21):
        val *= (x-i)
    return val

def wilk_fp(x):
    val = 20.*x**19.-\
    3990.*x**18.+\
    371070.*x**17.-\
    21366450.*x**16.+\
    853247136.*x**15.-\
    25084212300.*x**14.+\
    562404802820.*x**13.-\
    9829445398500.*x**12.+\
    135723323944572.*x**11.-\
    1491437011894830.*x**10.+\
    13075350105403950.*x**9.-\
    91280698789603050.*x**8.+\
    504246496794359168.*x**7.-\
    2179335502129734480.*x**6.+\
    7239886822682240160.*x**5.-\
    17999897589738036000.*x**4.+\
    32151247290580207104.*x**3.-\
    38612793735452966400.*x**2.+\
    27607519507281408000.*x-\
    8752948036761600000.
    return val

def wilkinson(N=default_N,max_steps=default_max_steps,mode=default_mode):
    '''
    Find roots of Wilkinson polynomial. 
    '''
    import sys
    sys.setrecursionlimit(10000)

    width = 12.
    height = 4.
    x_cent = 10
    y_cent = 0

    return droots(wilk_f,wilk_fp,x_cent,y_cent,width,height,N,max_steps,mode)

def boundary_root(N=default_N,max_steps=default_max_steps,mode=default_mode):
    '''
    Example where the root and boundary point coincide.
    '''
    f = lambda z: pow(z,2.)
    fp = lambda z: 2.*z
    x_cent = 1.
    y_cent = 0.
    width = 1.
    height = 1.

    return droots(f,fp,x_cent,y_cent,width,height,N,max_steps,mode)
