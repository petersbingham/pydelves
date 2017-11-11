# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 20:15:35 2015

@author: gil
@title: Rootfinder

Functions moved from file containing Delves routine

Modified by P Bingham October-November 2017

"""
import numpy as np
from scipy import integrate
import math
import cmath as cm

def muller(x1,x2,x3,f,N=400,ltol=1e-12,htol=1e-12):
    '''
    A method that works well for finding roots locally in the complex plane.
    Uses three points for initial guess, x1,x2,x3.

    Args:
        x1,x2,x3 (complex numbers): initial points for the algorithm.

        f (function): complex valued function for which to find roots.

        N(optional[int]): maximum number of iterations.

        ltol (optional[float]): low tolerance limit.

        htol (optional[float]): high tolerance limit.

    Returns:
        estimated root of the function f.
    '''
    n = 0
    x = x3

    if x1 == x2:
        raise Exception("Muller needs x1 and x2 different!!!")

    if x2 == x3:
        raise Exception("Muller needs x2 and x3 different!!!")

    if x1 == x3:
        raise Exception("Muller needs x1 and x3 different!!!")

    while n < N and abs(f(x3))>ltol:
        n+=1
        q = (x3 - x2) / (x2 - x1)
        A = q * f(x3) - q*(1.+q)*f(x2)+q**2.*f(x1)
        B = (2.*q+1.)*f(x3)-(1.+q)**2.*f(x2)+q**2.*f(x1)
        C = (1.+q)*f(x3)

        D1 = B+cm.sqrt(B**2-4.*A*C)
        D2 = B-cm.sqrt(B**2-4.*A*C)
        if abs(D1) > abs(D2):
            D = D1
        elif D1 == D2 == 0:
            x = x3
            break
        else: D = D2

        x = x3 - (x3-x2)*2.*C / D

        x1 = x2
        x2 = x3
        x3 = x
        #print x

    return x, abs(f(x))<=htol

def residues(f_frac,roots,lmt_N=10,lmt_eps=1e-3):
    '''
    Finds the resides of :math:`f_{frac} = f'/f` given the location of some roots of f.
    The roots of f are the poles of f_frac.

    Args:
        f_frac (function): a complex.

        roots (a list of complex numbers): the roots of f; poles of f_frac.

        lmt_N (int): parameter to he limit function.

        lmt_eps (optional[float]): parameter to he limit function.

    Returns:
        A list of residues of f_frac.

    '''
    return [limit(lambda z: (z-root)*f_frac(z),root,lmt_N,lmt_eps) for root in roots]

def limit(f,z0,N=10,eps=1e-3):
    '''
    Takes possibly matrix-valued function f and its simple pole z0 and returns
    limit_{z \to val} f(z). Estimates the value based on N surrounding
    points at a distance eps.

    Args:
        f (function): the function for which the limit will be found.

        z0 (complex number): The value at which the limit is evaluated.

        N (int): number of points used in the estimate.

        eps (optional[float]):
            distance from z0 at which estimating points are placed.

    Returns:
        Limit value (complex):
            The estimated value of :math:`limit_{z -> z_0} f(z)`.

    '''
    t=np.linspace(0.,2.*np.pi*(N-1.)/N,num=N)
    c=np.exp(1j*t)*eps
    s=sum(f(z0 + c_el) for c_el in c)/float(N)
    return s

def new_f_frac(f_frac,z0,residues,roots,val=None):
    '''
    Functions that evaluate the f_frac after some roots and their residues are subtracted.
    This function does NOT check to see if there is division by zero of if the
    values become too large.

    We assume here that the poles are of order 1.

    Args:
        f_frac (function): function for which roots will be subtracted.

        z0 (complex number): point where new_f_frac is evaluated.

        residues (list of complex numbers): The corresponding residues to subtract.

        roots (list of complex numbers): The corresponding roots to subtract.

        val (optional[complex number]): We can impose a value f_frac(z0) if we wish.

    Returns:
        The new value of f_frac(z0) once the chosen poles have been subtracted.
    '''
    if val == None:
        val = f_frac(z0)
    for res,root in zip(residues,roots):
        val -= res/(z0-root)
    return val

def new_f_frac_safe(f_frac,z0,residues,roots,max_ok,val=None,lmt_N=10,lmt_eps=1e-3):
    '''
    Functions that evaluate the f_frac after some roots and their residues are subtracted.
    The safe version checks for large values and division by zero.
    If the value of f_frac(z0) is too large, subtracting the roots of f becomes
    numerically unstable. In this case, we approximate the new function f_frac
    by using the limit function.

    We assume here that the poles are of order 1.

    Args:
        f_frac (function): function for which roots will be subtracted.

        z0 (complex number): point where new_f_frac is evaluated.

        residues (list of complex numbers): The corresponding residues to
            subtract.

        roots (list of complex numbers): The corresponding roots to subtract.

        val (optional[complex number]): We can impose a value f_frac(z0) if
            we wish.

        max_ok (float) Maximum absolute value of f_frac(z0 to use).

    Returns:
        The new value of f_frac(z0) once the chosen poles have been subtracted.
    '''
    try:
        if val == None:
            val = f_frac(z0)
        if abs(val) < max_ok:
            return new_f_frac(f_frac,z0,residues,roots,val), True
        else:
            return limit(lambda z: new_f_frac(f_frac,z,residues,roots),
                         z0,lmt_N,lmt_eps), True
    except ZeroDivisionError:
        return limit(lambda z: new_f_frac(f_frac,z,residues,roots),
                     z0,lmt_N,lmt_eps), False

def locate_poly_roots(y_smooth,c,num_roots_to_find):
    '''
    given the values y_smooth, locations c, and the number to go up to,
    find the roots using the polynomial trick.

    Args:
        y_smooth (list of complex numbers): points along smoothed-out boundary.
    '''
    p=[0]  ##placeholder
    for i in xrange(1,num_roots_to_find+1):
        p.append(integrate.trapz([el*z**i for el,z in zip(y_smooth,c)],c) )
    e = [1.]
    for k in xrange(1,num_roots_to_find+1):
        s = 0.
        for i in xrange(1,k+1):
            s += (-1.)**(i-1)*e[k-i]*p[i]
        e.append(s / k)
    coeff = [e[k]*(-1.)**(2.*num_roots_to_find-k)
        for k in xrange(0,num_roots_to_find+1)]
    return np.roots(coeff)

def purge(lst,eps=1e-5):
    '''
    Get rid of redundant elements in a list. There is a precision cutoff eps.

    Args:
        lst (list): elements.

        eps (optional[float]): precision cutoff.

    Returns:
        A list without redundant elements.

    '''
    if len(lst) == 0:
        return []
    for el in lst[:-1]:
        if abs(el-lst[-1]) < eps:
            return purge(lst[:-1],eps)
    return purge(lst[:-1],eps) + [lst[-1]]

def get_unique(lst1,lst2,eps=1e-5):
    newlst = []
    for el1 in lst1:
        fnd = False
        for el2 in lst2:
            if abs(el1-el2) < eps:
                fnd = True
                break
        if not fnd:
            newlst.append(el1)
    return newlst

def linspace(c1,c2,num=50):
    '''
    make a linespace method for complex numbers.

    Args:
        c1,c2 (complex numbers): The two points along which to draw a line.

        num (optional [int]): number of points along the line.

    Returns:
        a list of num points starting at c1 and going to c2.
    '''
    x1 = c1.real
    y1 = c1.imag
    x2 = c2.real*(num-1.)/num+x1*(1.)/num
    y2 = c2.imag*(num-1.)/num+y1*(1.)/num
    return [real+imag*1j for real,imag in zip(np.linspace(x1,x2,num=num),
                                              np.linspace(y1,y2,num=num)) ]

def get_boundary(rx,ry,rw,rh,N):
    '''
    Make a rectangle centered at rx,ry. Find points along this rectangle.
    I use the convention that rw/rh make up half the dimensions of the rectangle.

    Args:
        rx,ry (floats): the coordinates of the center of the rectangle.

        rw,rh (floats): The (half) width and height of the rectangle.

        N (int): number of points to use along each edge.

    Returns:
        A list of points along the edge of the rectangle in the complex plane.
    '''
    c1 = rx-rw+(ry-rh)*1j
    c2 = rx+rw+(ry-rh)*1j
    c3 = rx+rw+(ry+rh)*1j
    c4 = rx-rw+(ry+rh)*1j
    return  linspace(c1,c2,num=N)+\
            linspace(c2,c3,num=N)+\
            linspace(c3,c4,num=N)+\
            linspace(c4,c1,num=N)


def inside_boundary(roots_near_boundary,rx,ry,rw,rh):
    '''
    Takes roots and the specification of a rectangular region
    returns the roots in the interior (and ON the boundary) of the region.

    Args:
        roots_near_boundary (list of complex numbers): roots near the boundary.

        rx,ry (floats): coordinates of the center of the region.

        rw,rh (floats): The (half) width and height of the rectangle.

    Returns:
        Roots in the interior and on the boundary of the rectangle.
    '''
    return [root for root in roots_near_boundary if
            rx - rw <= root.real <= rx + rw and \
            ry - rh <= root.imag <= ry + rh]

def get_max(y):
    '''
    return the :math:`IQR + median` to determine a maximum permissible value to use
    in the numerically safe function new_f_frac_safe.

    '''
    q75, q50, q25 = np.percentile(y, [75 , 50, 25])
    IQR = q75-q25
    return q50+IQR

def find_maxes(y):
    '''
    Given a list of numbers, find the indices where local maxima happen.

    Args:
        y(list of floats).

    Returns:
        list of indices where maxima occur.

    '''
    maxes = []
    for i in xrange(-2,len(y)-2):
        if y[i-1] < y[i] > y[i+1]:
            maxes.append(i)
    return maxes
