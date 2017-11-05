# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 20:15:35 2015

@author: gil
@title: Rootfinder

Find the roots of a function f in the complex plane inside of a rectangular region.
We implement the method in the following paper:

 Delves, L. M., and J. N. Lyness. "A numerical method for locating the zeros of
 an analytic function." Mathematics of computation 21.100 (1967): 543-560.

Alternative code using a similar method can be found here:

 http://cpc.cs.qub.ac.uk/summaries/ADKW_v1_0.html

The main idea is to compute contour integrals of functions of the form
:math:`z^k f'/f` around the contour, for integer values of k. Here :math:`f'` denotes the
derivative of f. The resulting values of the contour integrals are proportional
to :math:`\sum_i z_i^k`, where i is the index of the roots.

Throughout we denote :math:`f_{frac} = f'/f`.

I have also tried several optimizations and strategies for numerical stability.

"""
import numpy as np
from itertools import chain
from scipy import integrate
import math
import cmath as cm

def Muller(x1,x2,x3,f,tol=1e-12,N=400):
    '''
    A method that works well for finding roots locally in the complex plane.
    Uses three points for initial guess, x1,x2,x3.

    Args:
        x1,x2,x3 (complex numbers): initial points for the algorithm.

        f (function): complex valued function for which to find roots.

        tol (optional[float]): tolerance.

        N(optional[int]): maximum number of iterations.

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

    while n < N and abs(f(x3))>tol:
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

    return x, abs(f(x))<=tol

def residues(f_frac,roots,lmt_N=10,lmt_eps=1e-3):
    '''
    Finds the resides of :math:`f_{frac} = f'/f` given the location of some roots of f.
    The roots of f are the poles of f_frac.

    Args:
        f_frac (function): a complex.

        roots (a list of complex numbers): the roots of f; poles of f_frac.

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

def get_boundary(x_cent,y_cent,width,height,N):
    '''
    Make a rectangle centered at x_cent,y_cent. Find points along this rectangle.
    I use the convention that width/height make up half the dimensions of the rectangle.

    Args:
        x_cent,y_cent (floats): the coordinates of the center of the rectangle.

        width,height (float): The (half) width and height of the rectangle.

        N (int): number of points to use along each edge.

    Returns:
        A list of points along the edge of the rectangle in the complex plane.
    '''
    c1 = x_cent-width+(y_cent-height)*1j
    c2 = x_cent+width+(y_cent-height)*1j
    c3 = x_cent+width+(y_cent+height)*1j
    c4 = x_cent-width+(y_cent+height)*1j
    return  linspace(c1,c2,num=N)+\
            linspace(c2,c3,num=N)+\
            linspace(c3,c4,num=N)+\
            linspace(c4,c1,num=N)


def inside_boundary(roots_near_boundary,x_cent,y_cent,width,height):
    '''
    Takes roots and the specification of a rectangular region
    returns the roots in the interior (and ON the boundary) of the region.

    Args:
        roots_near_boundary (list of complex numbers): roots near the boundary.

        x_cent,y_cent (floats): coordinates of the center of the region.

        width,height (floats): The (half) width of height of the rectangle.

    Returns:
        Roots in the interior and on the boundary of the rectangle.
    '''
    return [root for root in roots_near_boundary if
            x_cent - width <= root.real <= x_cent + width and \
            y_cent - height <= root.imag <= y_cent + height]

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

def root_purge(lst,eps=1e-7,min_i=1e-10):
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
        if abs(el-lst[-1]) < eps and \
        (el.imag/lst[-1].imag>=0 or abs(el.imag)<min_i):
            return root_purge(lst[:-1],eps,min_i)
    return root_purge(lst[:-1],eps,min_i) + [lst[-1]]

def add_missing_conjugates(lst,eps=1e-7,min_i=1e-10):
    new_lst = []
    for el in lst:
        new_lst.append(el)
        new_lst.append(el.conjugate())
    return root_purge(new_lst,eps,min_i)

def handle_warning(warn,verbose,lvl_cnt):
    s = " "*lvl_cnt
    imprecise_roots = s+"Warning!! Number of roots may be imprecise for " + \
                      "this N. Increase N for greater precision."
    max_steps_exceeded = s+"Warning!! max_steps exceeded. Some interior " + \
                         "roots might be missing."
    no_bnd_muller_root = s+"Warning!! Boundary Muller failed to converge." 
    bnd_muller_exception = s+"Warning!! Exception during boundary Muller." 
    no_int_muller_root = s+"Warning!! Interior Muller failed to converge." 
    int_muller_exception = s+"Warning!! Exception during interior Muller." 
    not_all_interior_fnd = s+"Warning!! Not all predicted interior roots " + \
                           "found."
    root_subtraction_division_by_zero = s+"Warning!! Division by zero " + \
                                        "during root subtraction."
    
    if verbose:
        if warn == warn_imprecise_roots:
            print imprecise_roots
        elif warn == warn_max_steps_exceeded:
            print max_steps_exceeded
        elif warn == warn_no_bnd_muller_root:
            print no_bnd_muller_root
        elif warn == warn_bnd_muller_exception:
            print bnd_muller_exception
        elif warn == warn_no_int_muller_root:
            print no_int_muller_root
        elif warn == warn_int_muller_exception:
            print int_muller_exception
        elif warn == warn_not_all_interior_fnd:
            print not_all_interior_fnd
        elif warn == warn_root_subtraction_division_by_zero:
            print root_subtraction_division_by_zero
    return warn

def print_roots_rect_summary(warn,num_final_roots,num_added_conj_roots,roots_near_boundary,
                           I0,num_interior_roots_fnd,num_sub_roots_fnd,num_known_roots,
                           x_cent,y_cent,width,height,lvl_cnt,dist_eps,verbose):
    '''
    Return final roots and optionally prints summary of get_roots_rect.

    Args:
        warn (int): warnings generated during get_roots_rect.
        
        num_final_roots (floats): total roots within region found by the routine.
        
        num_added_conj_roots (floats): number of conjugate roots added.
        
        roots_near_boundary (floats): roots found during the smoothing.
        
        num_known_roots (floats): number of roots previously discovered.
        
        I0 (float): number of roots predicted using Roche.
        
        x_cent,y_cent (floats): coordinates of the center of the region.

        width,height (floats): The (half) width of height of the rectangle.

        verbose (optional[boolean]): print all warnings.
    '''
    s = " "*lvl_cnt
    d = "-"*lvl_cnt
    if verbose:
        if warn == 0:
            print d+"Calculations complete. No warnings."
        else:
            print d+"Calculations completed with following warnings occurring at least once:"
            if warn & warn_imprecise_roots:
                print s+"  -Imprecise number of roots in region."
            if warn & warn_max_steps_exceeded:
                print s+"  -Number of region steps exceeded."
            if warn & warn_no_bnd_muller_root:
                print s+"  -No boundary muller root found with specified parameters."
            if warn & warn_bnd_muller_exception:
                print s+"  -Exception during Muller routine."
            if warn & warn_no_int_muller_root:
                print s+"  -No interior muller root found with specified parameters."
            if warn & warn_not_all_interior_fnd:
                print s+"  -Not all predicted interior roots found."
            if warn & warn_root_subtraction_division_by_zero:
                print s+"  -Division by zero when subtracting roots."

        roots_within_boundary = inside_boundary(
                                    purge(roots_near_boundary,dist_eps),
                                    x_cent,y_cent,width,height)
        num_roots_found = num_interior_roots_fnd+num_sub_roots_fnd+num_added_conj_roots
        if num_known_roots != 0:
            print s + str(num_known_roots) + " known roots."
        print s+"Total of " + str(num_final_roots) + " new roots found."
        print s+"  " + str(len(roots_within_boundary)) + " from Boundary Muller."
        print (s+"  Internal: {:.5f}".format(abs(I0)) + " Roche predicted. " + 
               str(num_roots_found) + " located:")
        print s+"    " + str(num_interior_roots_fnd) + " from Poly Muller."
        print s+"    " + str(num_sub_roots_fnd) + " from subregions."
        if num_added_conj_roots != 0:
            print s+"    " + str(num_added_conj_roots) + " added conjugates."

def print_roots(roots_near_boundary_all,roots_near_boundary,roots_subtracted,
                roots_rough,roots_interior_mull_all,roots_interior_mull,
                roots_interior_mull_unique,roots_interior_all_subs,roots_all,
                roots_final,roots_new,lvl_cnt,verbose):
    s = " "*lvl_cnt
    if verbose:
        print "\n"+s+"All:\n" + str(np.array(roots_near_boundary_all))
        print s+"Purged:\n" + str(np.array(roots_near_boundary))
        print s+"Subtracted:\n" + str(np.array(roots_subtracted))
        print ""
        print s+"Rough:\n" + str(roots_rough)
        print s+"Interior:\n" + str(np.array(roots_interior_mull_all))
        print s+"Purged:\n" + str(np.array(roots_interior_mull))
        print s+"Unique:\n" + str(np.array(roots_interior_mull_unique))
        print ""
        print s+"Subs:\n" + str(np.array(roots_interior_all_subs))
        print ""
        print s+"All:\n" + str(np.array(roots_all))
        print s+"Final:\n" + str(np.array(roots_final))
        print s+"New:\n" + str(np.array(roots_new))

def locate_muller_root(x1,x2,x3,f,mul_tol,mul_N,roots,log,lvl_cnt):
    warn = 0
    try:          
        mull_root,ret = Muller(x1,x2,x3,f,mul_tol,mul_N)
        if ret:
            roots.append(mull_root)
        else:
            warn |= handle_warning(warn_no_bnd_muller_root,log&log_all_warn,lvl_cnt)
    except:
        warn |= handle_warning(warn_bnd_muller_exception,log&log_all_warn,lvl_cnt)
    return warn

def correct_roots(roots,x_cent,y_cent,width,height,min_i):
    roots_inside = inside_boundary(roots,x_cent,y_cent,width,height)
    conjs_added = 0
    if min_i:
        roots_final = inside_boundary(add_missing_conjugates(roots_inside),
                                      x_cent,y_cent,width,height)
        conjs_added = len(roots_final)-len(roots_inside)
    else:
        roots_final = roots_inside
    return roots_final, conjs_added

log_off = 0
log_recursive = 1
log_summary = 2
log_all_warn = 4
log_debug = 8

warn_imprecise_roots = 1
warn_max_steps_exceeded = 2
warn_no_bnd_muller_root = 4
warn_bnd_muller_exception = 8
warn_no_int_muller_root = 16
warn_int_muller_exception = 32
warn_not_all_interior_fnd = 64
warn_root_subtraction_division_by_zero = 128
def get_roots_rect(f,fp,x_cent,y_cent,width,height,N=10,outlier_coeff=100.,
    max_steps=5,max_order=10,mul_tol=1e-12,mul_N=400,mul_off=1e-5,
    dist_eps=1e-7,lmt_N=10,lmt_eps=1e-3,min_i=None,log=log_off,
    roots_known=[],lvl_cnt=0):
    '''
    I assume f is analytic with simple (i.e. order one) zeros.

    TODO:
    save values along edges if iterating to a smaller rectangle
    extend to other kinds of functions, e.g. function with non-simple zeros.

    Args:
        f (function): the function for which the roots (i.e. zeros) will be
            found.

        fp (function): the derivative of f.

        x_cent,y_cent (floats): The center of the rectangle in the complex
            plane.

        width,height (floats): half the width and height of the rectangular
            region.

        N (optional[int]): Number of points to sample per edge

        outlier_coeff (optional[float]): multiplier for coefficient used when 
            subtracting poles to improve numerical stability. 
            See new_f_frac_safe.

        max_step (optional[int]): Number of iterations allowed for algorithm to
            repeat on smaller rectangles.

        mul_tol (optional[float]): muller tolerance.

        mul_N (optional[int]): maximum number of iterations for muller.

        mul_off (optional[float]): muller point offset .

        known roots (optional[list of complex numbers]): Roots of f that are
            already known.

        min_i (optional[boolean]): If function is a polynomial then
            roots will occur in a+ib, a-ib pairs. This options takes this mode
            into account when purging roots that are close to the real axis.
            calculation.

    Returns:
        A list of roots for the function f inside the rectangle determined by
            the values x_cent,y_cent,width, and height. Also a warning and
            number of regions in calculation.
    '''

    warn = 0
    num_regions = 1

    s = "-"*lvl_cnt
    if log&log_recursive:
        print ("\n"+s+"Region(x,y,w,h): "+str(x_cent)+" "+str(y_cent)+" "
               +str(width)+" "+str(height))


    c = get_boundary(x_cent,y_cent,width,height,N)
    f_frac = lambda z: fp(z)/(2j*np.pi*f(z))
    y = [f_frac(z) for z in c]

    outliers = find_maxes(map(abs,y))
    roots_near_boundary_all = []
    for outlier_index in outliers:
        warn |= locate_muller_root(c[outlier_index-2],c[outlier_index+2],
                                   c[outlier_index]/2,f,mul_tol,mul_N,
                                   roots_near_boundary_all,log,lvl_cnt)
    roots_near_boundary = purge(roots_near_boundary_all,dist_eps)
    roots_subtracted = purge(roots_near_boundary+roots_known,dist_eps)
    # We don't need the roots far outside the boundary
    roots_subtracted = inside_boundary(roots_subtracted,x_cent,y_cent,
                                       width+2.,height+2.)
    max_ok = abs(outlier_coeff*get_max(y))
    residues_subtracted = residues(f_frac,roots_subtracted,lmt_N,lmt_eps)

    
    y_smooth = []
    for y_el,z_el in zip(y,c):
        val, ret = new_f_frac_safe(f_frac,z_el,residues_subtracted,
                                   roots_subtracted,max_ok,y_el,lmt_N,lmt_eps)
        y_smooth.append(val)
        if not ret:
            warn |= handle_warning(warn_root_subtraction_division_by_zero,
                                   log&log_all_warn,lvl_cnt)
    I0 = integrate.trapz(y_smooth, c)  # Approx number of roots not subtracted
    tot_num_interior_pred = int(math.ceil(abs(I0)-0.005))
    
    roots_rough = []
    roots_interior_mull_all = []
    roots_interior_mull = []
    if I0 < max_order:
        # If there's only a few roots, find them.
        if tot_num_interior_pred == 0:
            roots_final,conjs_added = correct_roots(roots_subtracted,x_cent,
                                                   y_cent,width,height,
                                                   min_i)
            roots_new = get_unique(roots_final,roots_known,dist_eps)
            print_roots_rect_summary(warn,len(roots_final),conjs_added,
                                     roots_near_boundary,I0,0,0,
                                     len(roots_known),x_cent,y_cent,width,
                                     height,lvl_cnt,dist_eps,log&log_summary)
            print_roots(roots_near_boundary_all,roots_near_boundary,roots_subtracted,
                        [],[],[],[],[],[],roots_final,roots_new,lvl_cnt,log&log_debug)
            return roots_new,warn,num_regions
        if abs(tot_num_interior_pred-I0)>0.005:
            warn |= handle_warning(warn_imprecise_roots,log&log_all_warn,
                                   lvl_cnt)

        roots_rough = locate_poly_roots(y_smooth,c,tot_num_interior_pred)
        ##TODO: best way to pick points for Muller method below
        for root in roots_rough:
            warn |= locate_muller_root(root-mul_off,root+mul_off,root,f,mul_tol,
                                       mul_N,roots_interior_mull_all,log,
                                       lvl_cnt)
        roots_interior_mull = purge(roots_interior_mull_all,dist_eps)
    roots_interior_mull_unique = inside_boundary(get_unique(roots_interior_mull,
                                     roots_near_boundary,dist_eps),
                                     x_cent,y_cent,width,height)

    roots_interior_mull_final,conjs_added = correct_roots(roots_interior_mull_unique,
                                                          x_cent,y_cent,width,height,
                                                          min_i)
    roots_all = purge(roots_near_boundary+roots_interior_mull_final,dist_eps)
    roots_interior_all_subs = []
    # Don't count the added conjs at thie stage, just pass them to the subregions.
    # This is because the Roche sometimes does not locate both paired roots.
    all_interior_found = len(roots_interior_mull_unique) >= tot_num_interior_pred
    roche_accurate = abs(tot_num_interior_pred-I0)<0.005
    was_subs = False
    # If some interior roots are missed or if there were many roots,
    # subdivide the rectangle and search recursively.
    if (I0>=max_order or not all_interior_found or not roche_accurate) and max_steps!=0:
        was_subs = True
        x_list = [x_cent - width / 2.,x_cent - width / 2.,
                  x_cent + width / 2.,x_cent + width / 2.]
        y_list = [y_cent - height / 2.,y_cent + height / 2.,
                  y_cent - height / 2.,y_cent + height / 2.]
        if log&log_summary:
            print s+"Contains " +str(len(x_list)) + " subregions:"
        for x,y in zip(x_list,y_list):
            new_log = log if log&log_recursive else log_off
            roots_from_subrectangle,newWarn,new_regions = get_roots_rect(f,fp,
                x,y,width/2.,height/2.,N,outlier_coeff,max_steps-1,max_order,
                mul_tol,mul_N,mul_off,dist_eps,lmt_N,lmt_eps,min_i,
                new_log,roots_all,lvl_cnt+1)
            warn |= newWarn
            num_regions += new_regions
            roots_interior_all_subs.extend(roots_from_subrectangle)
    elif max_steps == 0:
        warn |= handle_warning(warn_max_steps_exceeded,log&log_all_warn,lvl_cnt)
    tot_num_interior_found = len(roots_interior_mull_final+roots_interior_all_subs)
    if tot_num_interior_found != tot_num_interior_pred:
        warn |= handle_warning(warn_not_all_interior_fnd,log&log_all_warn,lvl_cnt)
    roots_all = purge(roots_all+roots_interior_all_subs,dist_eps)


    roots_final,conjs_added = correct_roots(roots_all,x_cent,y_cent,width,
                                           height,min_i)
    roots_new = get_unique(roots_final,roots_known,dist_eps)
    if was_subs and log&log_summary: print
    print_roots_rect_summary(warn,len(roots_final),conjs_added,
                             roots_near_boundary,I0,len(roots_interior_mull_unique),
                             len(roots_interior_all_subs),len(roots_known),
                             x_cent,y_cent,width,height,lvl_cnt,dist_eps,
                             log&log_summary)
    print_roots(roots_near_boundary_all,roots_near_boundary,roots_subtracted,
                roots_rough,roots_interior_mull_all,roots_interior_mull,
                roots_interior_mull_unique, roots_interior_all_subs,roots_all,
                roots_final,roots_new,lvl_cnt,log&log_debug)
    return roots_new,warn,num_regions
