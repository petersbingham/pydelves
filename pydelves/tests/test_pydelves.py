from gilroots import *

import numpy.testing as testing
import numpy as np
import numpy.linalg as la
from scipy.integrate import ode
import scipy.constants as consts

import matplotlib.pyplot as plt
import time

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

def almost_equal(el1,el2,eps=1e-7):
    if abs(el1 - el2) < eps:
        return True
    else: return False  

def two_sets_almost_equal(S1,S2,eps=1e-7):
    '''
    Tests if two iterables have the same elements up to some tolerance eps.

    Args:
        S1,S2 (lists): two lists
        eps (optional[float]): precision for testing each elements

    Returns:
        True if the two sets are equal up to eps, false otherwise
    '''
    if len(S1) != len(S2):
        return False

    ran2 = range(len(S2))
    for i in range(len(S1)):
        found_match = False
        for j in ran2:
            if almost_equal(S1[i],S2[j],eps):
                found_match = True
                ran2.remove(j)
                break
        if not found_match:
            return False
    return True


def test_Roots():
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

    all_fnd,retroots = droots(f,fp,x_cent,y_cent,width,height,N)
    roots = np.asarray(retroots)
    print two_sets_almost_equal(roots/np.pi,
                                [-5.,-4.,-3.,-2.,-1.,-0.,1.,2.,3.,4.,5.] )


def _print_params(x_cent,y_cent,width,height,N,outlier_coeff,max_steps,
                  max_order,mul_fzltol,mul_fzhtol,mul_N,mul_off,dist_eps,
                  lmt_N,lmt_eps,bnd_thres,conj_min_i,I0_tol):
    print "x_cent:" + str(x_cent)
    print "y_cent:" + str(y_cent)
    print "width:" + str(width)
    print "height:" + str(height)
    print "N:" + str(N)
    print "outlier_coeff:" + str(outlier_coeff)
    print "max_steps:" + str(max_steps)
    print "max_order:" + str(max_order)
    print "mul_fzltol:" + str(mul_fzltol)
    print "mul_fzhtol:" + str(mul_fzhtol)
    print "mul_N:" + str(mul_N)
    print "mul_off:" + str(mul_off)
    print "dist_eps:" + str(dist_eps)
    print "lmt_N:" + str(lmt_N)
    print "lmt_eps:" + str(lmt_eps)
    print "bnd_thres:" + str(bnd_thres)
    print "conj_min_i:" + str(conj_min_i)
    print "I0_tol:" + str(I0_tol)    

def truncateFloat(dps, val):
    return float(("{0:."+str(dps)+"f}").format(val))
def truncateComplex(dps, val):
    return truncateFloat(dps,val.real) + truncateFloat(dps,val.imag)*1.0j

def test_Wilkinson(printRoots=False, printPolys=False, printParams=False, 
                   doubleOnWarning=False):
    '''
    Find roots of wilkonson polynomial. 
    '''
    def f(x):
        val = 1
        for i in range(1,21):
            val *= (x-i)
        #return truncateComplex(21,val)
        return val

    def fp(x):
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
        #return truncateComplex(21,val)
        return val

    import sys
    sys.setrecursionlimit(10000)

    width = 12.
    height = 4.
    x_cent = 10
    y_cent = 0

    N = 100
    max_steps = 5
    
    outlier_coeff = 100.
    max_order = 10
    I0_tol = 5e-3

    mul_N = 400
    mul_fzltol = 1e-12
    mul_fzhtol = 1e-12
    mul_off = 1e-5

    mul_ztol = 1e-3
    conj_min_i = 1e-8

    dist_eps = 1e-7
    lmt_N = 10
    lmt_eps = 1e-3
    bnd_thres = .1

    logmode = mode_off    
    logmode |= mode_log_summary
    #logmode |= mode_log_recursive
    #logmode |= mode_log_debug
    #logmode |= mode_log_verbose

    calcmode = mode_off
    calcmode |= mode_recurse_on_inaccurate_roche | mode_recurse_on_not_all_interior_found
    #calcmode |= mode_accept_int_muller_close_to_good_roche | mode_use_stripped_subtraction

    mode = logmode | calcmode

    if printPolys:
        print poly
        print poly_diff

    if printParams:
        print_params(x_cent,y_cent,width,height,N,outlier_coeff,max_steps,
                     max_order,mul_fzltol,mul_fzhtol,mul_N,mul_off,dist_eps,
                     lmt_N,lmt_eps,bnd_thres,conj_min_i,I0_tol)

    set_delves_routine_parameters(outlier_coeff,max_order,I0_tol)
    set_muller_parameters(mul_N,mul_fzltol,mul_fzhtol,mul_off)
    set_mode_parameters(mul_ztol,conj_min_i)
    set_advanced_parameters(dist_eps,lmt_N,lmt_eps,bnd_thres)
    state,roots=droots(f,fp,x_cent,y_cent,width,height,N,max_steps,mode)
    print "warn: " + str(state & 0x7)

    if printRoots:
        for root in sorted(roots):
          print str(root) + "  \t" + str(f(root))

def test_Poly_Roots_N(N, printRoots=False, printPolys=False, printParams=False, doubleOnWarning=False):
    '''
    Find roots of polynomials with increasing powers. Compares with roots returned from np.roots.
    '''
    print "\nN=" + str(N)

    coeff = []
    for n in range(N):
      coeff.append((n+1)*1.0+(n+1)*1.0j)
    roots_numpy = np.roots(coeff)
    bnds = get_root_bounds(roots_numpy)

    poly = np.poly1d(coeff)
    poly_diff = np.polyder(poly)

    f = lambda z: poly(z)
    fp = lambda z: poly_diff(z)
    width = (bnds[0][1]-bnds[0][0])/2.
    height = (bnds[1][1]-bnds[1][0])/2.
    x_cent = bnds[0][0] + width
    y_cent = bnds[1][0] + height
    width += 0.1
    height += 0.1

    N = 1000
    max_steps = 5
    
    outlier_coeff = 100.
    max_order = 10
    I0_tol = 5e-3

    mul_N = 400
    mul_fzltol = 1e-12
    mul_fzhtol = 1e-12
    mul_off = 1e-5

    mul_ztol = 1e-3
    conj_min_i = 1e-8

    dist_eps = 1e-7
    lmt_N = 10
    lmt_eps = 1e-3
    bnd_thres = .1

    logmode = mode_off    
    logmode |= mode_log_summary
    #logmode |= mode_log_recursive
    #logmode |= mode_log_debug
    #logmode |= mode_log_verbose

    calcmode = mode_off
    calcmode |= mode_recurse_on_inaccurate_roche | mode_recurse_on_not_all_interior_found
    #calcmode |= mode_accept_int_muller_close_to_good_roche | mode_use_stripped_subtraction

    mode = logmode | calcmode

    if printPolys:
        print poly
        print poly_diff

    if printParams:
        print_params(x_cent,y_cent,width,height,N,outlier_coeff,max_steps,
                     max_order,mul_fzltol,mul_fzhtol,mul_N,mul_off,dist_eps,
                     lmt_N,lmt_eps,bnd_thres,conj_min_i,I0_tol)

    set_delves_routine_parameters(outlier_coeff,max_order,I0_tol)
    set_muller_parameters(mul_N,mul_fzltol,mul_fzhtol,mul_off)
    set_mode_parameters(mul_ztol,conj_min_i)
    set_advanced_parameters(dist_eps,lmt_N,lmt_eps,bnd_thres)
    state,roots=droots(f,fp,x_cent,y_cent,width,height,N,max_steps,mode)
    print "warn: " + str(state & 0x7)

    print "Comparison with numpy:"
    print "\t" + str(len(roots_numpy)) + " numpy roots"
    print "\t" + str(len(roots)) + " gil roots"
    common = 0
    for root_numpy in roots_numpy:
        for root_gil in roots:
            if almost_equal(root_numpy, root_gil,eps=1e-5):
                common += 1
                break
    print "\t" + str(common) + " common roots"

    if printRoots:
        for root in sorted(roots_numpy):
          print str(root) + "  \t" + str(f(root))
        print
        for root in sorted(roots):
          print str(root) + "  \t" + str(f(root))

def test_Poly_Roots(printRoots=False, printPolys=False, printParams=False):
    for N in range(2,41):
        test_Poly_Roots_N(N,printRoots,printPolys,printParams)

if __name__ == "__main__":
    #test_Roots()
    #test_Poly_Roots()
    test_Poly_Roots_N(31, printRoots=False, printPolys=False, printParams=False)
    #test_Wilkinson(printRoots=True, printPolys=False, printParams=False)
