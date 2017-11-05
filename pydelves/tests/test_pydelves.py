import gilroots as Roots

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


def test_Roots_1():
    '''
    Make a square of length just under 5*pi. Find the roots of sine.
    '''
    N=5000
    f = lambda z: np.sin(z)
    fp = lambda z: np.cos(z)
    x_cent = 0.
    y_cent = 0.
    width = 5.*np.pi-1e-5
    height = 5.*np.pi-1e-5

    roots_gil, warn, numregions = Roots.get_roots_rect(f,fp,x_cent,y_cent,width,height,N)
    roots = np.asarray(retRoots)
    roots_inside_boundary = Roots.inside_boundary(roots,x_cent,y_cent,width,height)
    print two_sets_almost_equal(np.asarray(roots_inside_boundary)/np.pi,
        [-4.,-3.,-2.,-1.,-0.,1.,2.,3.,4.] )

def test_Roots_2_fun(f, fp):
    '''
    Make a square of length just over 5*pi. Find the roots of sine.
    '''
    N=5000
    x_cent = 0.
    y_cent = 0.
    width = 5.*np.pi+1e-5
    height = 5.*np.pi+1e-5

    roots_gil, warn, numregions = Roots.get_roots_rect(f,fp,x_cent,y_cent,width,height,N,summary=True)
    roots = np.asarray(retRoots)
    roots_inside_boundary = Roots.inside_boundary(roots,x_cent,y_cent,width,height)
    print two_sets_almost_equal(np.asarray(roots_inside_boundary)/np.pi,
        [-5.,-4.,-3.,-2.,-1.,-0.,1.,2.,3.,4.,5.] )

def test_Roots_2():
    f = lambda z: np.sin(z)
    fp = lambda z: np.cos(z)
    test_Roots_2_fun(f, fp)

def test_Poly_Roots(N, printRoots=False, printPolys=False, printParams=False, doubleOnWarning=False):
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

    N = 10
    outlier_coeff = 100.
    max_steps = 5
    max_order=10
    
    mul_tol = 1e-12
    mul_N = 400
    mul_off = 1e-5
    
    dist_eps = 1e-7
    lmt_N = 10
    lmt_eps = 1e-3
    min_i = 1e-8
    min_i = None

    if printPolys:
        print poly
        print poly_diff

    ret = -1
    while ret==-1 or (doubleOnWarning and ret!=0):
        # Doubling is for test purposes.
        if ret & Roots.warn_imprecise_roots:
            N *= 2
        elif ret & Roots.warn_max_steps_exceeded:
            max_steps *= 2
        elif ret & Roots.warn_no_muller_root:
            mul_N *= 2
        if printParams:
            print "x_cent:" + str(x_cent)
            print "y_cent:" + str(y_cent)
            print "width:" + str(width)
            print "height:" + str(height)
            print "N:" + str(N)
            print "outlier_coeff:" + str(outlier_coeff)
            print "max_steps:" + str(max_steps)
            print "mul_tol:" + str(mul_tol)
            print "mul_N:" + str(mul_N)
        ret = Roots.get_roots_rect(f,fp,x_cent,y_cent,width,height,N,
                                   outlier_coeff,max_steps,max_order,mul_tol,
                                   mul_N,mul_off,dist_eps,lmt_N,lmt_eps,
                                   min_i,
                                   log=Roots.log_summary)
                                   #log=Roots.log_summary|Roots.log_recursive)
                                   #log=Roots.log_summary|Roots.log_debug)
                                   #log=Roots.log_summary|Roots.log_debug|Roots.log_recursive)
        roots_gil, warn, numregions = ret
        roots_gil = np.asarray(roots_gil)
        roots_gil = Roots.inside_boundary(roots_gil,x_cent,y_cent,width,height)

        print "\nComparison with numpy:"
        print "\t" + str(len(roots_numpy)) + " numpy roots"
        print "\t" + str(len(roots_gil)) + " gil roots"
        common = 0
        for root_numpy in roots_numpy:
            for root_gil in roots_gil:
                if almost_equal(root_numpy, root_gil,eps=1e-5):
                    common += 1
                    break
        print "\t" + str(common) + " common roots"

        if printRoots:
            for root in sorted(roots_numpy):
              print str(root) + "  \t" + str(f(root))
            print
            for root in sorted(roots_gil):
              print str(root) + "  \t" + str(f(root))

def test_Roots_3(printRoots=False, printPolys=False, printParams=False, doubleOnWarning=False):
    for N in range(2,41):
        test_Poly_Roots(N,printRoots,printPolys,printParams,doubleOnWarning)

if __name__ == "__main__":
    #test_Roots_1()
    #test_Roots_2()
    test_Roots_3()
    #test_Poly_Roots(20, printRoots=False, printPolys=False, printParams=False, doubleOnWarning=False)
