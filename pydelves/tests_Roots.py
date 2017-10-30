from .. import Potapov
from .. import Roots
from .. import Time_Delay_Network
from .. import functions
from .. import Time_Sims_nonlin
from .. import Hamiltonian
from .. import phase_matching

import numpy.testing as testing
import numpy as np
import numpy.linalg as la
from scipy.integrate import ode
import scipy.constants as consts

import matplotlib.pyplot as plt
import time

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

    def almost_equal(el1,el2):
        if abs(el1 - el2) < eps:
            return True
        else: return False

    ran2 = range(len(S2))
    for i in range(len(S1)):
        found_match = False
        for j in ran2:
            if almost_equal(S1[i],S2[j]):
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

    roots = np.asarray(Roots.get_roots_rect(f,fp,x_cent,y_cent,width,height,N))
    roots_inside_boundary = Roots.inside_boundary(roots,x_cent,y_cent,width,height)
    two_sets_almost_equal(np.asarray(roots_inside_boundary)/np.pi,
        [-4.,-3.,-2.,-1.,-0.,1.,2.,3.,4.] )

def test_Roots_2():
    '''
    Make a square of length just over 5*pi. Find the roots of sine.
    '''
    N=5000
    f = lambda z: np.sin(z)
    fp = lambda z: np.cos(z)
    x_cent = 0.
    y_cent = 0.
    width = 5.*np.pi+1e-5
    height = 5.*np.pi+1e-5

    roots = np.asarray(Roots.get_roots_rect(f,fp,x_cent,y_cent,width,height,N))
    roots_inside_boundary = Roots.inside_boundary(roots,x_cent,y_cent,width,height)
    two_sets_almost_equal(np.asarray(roots_inside_boundary)/np.pi,
        [-5.,-4.,-3.,-2.,-1.,-0.,1.,2.,3.,4.,5.] )



if __name__ == "__main__":
    test_Roots_1()
    test_Roots_2()
