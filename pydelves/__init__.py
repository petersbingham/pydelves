# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 20:15:35 2015

@author: gil
@title: Rootfinder
Modified by P Bingham October-November 2017

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
from scipy import integrate
import math
from functions import *


mode_off = 0
#If function is a polynomial then roots will occur in a+ib, a-ib pairs.
#If mode==mode_add_conjs then routine will take advantage of this by
#automatically adding missing partners.
mode_add_conjs = 0x1

mode_log_recursive = 0x200
mode_log_summary = 0x400
mode_log_notes = 0x800
mode_log_all_notes = 0x1000
mode_log_debug = 0x2000

#Used for switching the log off on recursion:
mode_log_switch = 0x1FF

warn_max_steps_reached = 1
note_imprecise_roots = 2
note_no_bnd_muller_root = 4
note_bnd_muller_exception = 8
note_no_int_muller_root = 16
note_int_muller_exception = 32
note_not_all_interior_fnd = 64
note_root_sub_div_by_zero = 128

def root_purge(lst,eps=1e-7,conj_min_i=1e-8):
    if len(lst) == 0:
        return []
    for el in lst[:-1]:
        if abs(el-lst[-1]) < eps and \
        (el.imag/lst[-1].imag>=0 or abs(el.imag)<conj_min_i):
            return root_purge(lst[:-1],eps,conj_min_i)
    return root_purge(lst[:-1],eps,conj_min_i) + [lst[-1]]

def add_miss_conjs(lst,eps=1e-7,conj_min_i=1e-8):
    new_lst = []
    for el in lst:
        new_lst.append(el)
        new_lst.append(el.conjugate())
    return root_purge(new_lst,eps,conj_min_i)

def locate_muller_root(lp,x1,x2,x3,roots,failed_roots):
    state = 0
    try:          
        mull_root,ret = muller(x1,x2,x3,gp.f,gp.mul_N,gp.mul_ltol,gp.mul_htol)
        if ret:
            roots.append(mull_root)
        else:
            failed_roots.append(mull_root)
            state |= gp.handle_state_change(lp,note_no_bnd_muller_root)
    except:
        state |= gp.handle_state_change(lp,note_bnd_muller_exception)
    return state

def correct_roots(lp,b,roots):
    roots_inside = inside_boundary(roots,b.rx,b.ry,b.rw,b.rh)
    conjs_added = 0
    if lp.mode & mode_add_conjs:
        roots_miss_conj = add_miss_conjs(roots_inside,gp.dist_eps,gp.conj_min_i)
        roots_final = inside_boundary(roots_missing_conjs,b.rx,b.ry,b.rw,b.rh)
        conjs_added = len(roots_final)-len(roots_inside)
    else:
        roots_final = roots_inside
    return roots_final, conjs_added

class root_container:
    def __init__(self, known):
        self.known = known

        self.boundary_all = []
        self.boundary_failed_mulls = []  
        self.boundary_purged = [] 
        self.boundary_new = []     
        self.boundary_within = []
        self.boundary_and_known = []
        self.residues_subtracted = []

        self.subtracted = []

        self.interior_rough = []
        self.interior_mull_all = []
        self.interior_failed_mulls = []
        self.interior_mull_purged = []
        self.interior_mull_within = []
        self.interior_mull_new = []

        self.region = []
        self.region_corrected = []
        self.num_added_conjs = 0
        
        self.interior_all_subs = []
          
        self.end = []
        self.end_purged = []
        self.end_unique = []

    def log_region(self,lp,b,I0,isSubregions):
        if lp.mode & mode_log_summary:
            self._log_region(lp,b,I0,isSubregions)

    def _log_region(self,lp,b,I0,isSubregions):
        s = "."*lp.lvl_cnt
        gp.print_region_string(lp,b)
        num_known_roots = len(self.known)
        num_interior_roots_fnd = len(self.interior_mull_new)
        if num_known_roots != 0:
            print s + str(num_known_roots) + " known."
        print s+str(len(self.boundary_within))+" from Boundary Muller."
        print s+"{:.5f}".format(abs(I0))+" Roche predicted. "
        a=""
        print s+"  " + str(num_interior_roots_fnd) + " from Poly Muller."+a
        if not isSubregions:
            print s+"  There were no subregions."
        elif lp.mode & mode_log_recursive:
            print s+"  Subregions:"

    def log_sub_region(self,lp,num_regions):
        if lp.mode & mode_log_summary:
            self._log_sub_region(lp,num_regions)

    def _log_sub_region(self,lp,num_regions):
        s = "."*lp.lvl_cnt
        num_sub_roots_fnd = len(self.interior_all_subs)
        if num_regions>0 and lp.mode & mode_log_recursive:
            print ""
        print(s+"  "+str(num_sub_roots_fnd)+" from "+str(num_regions)+\
              " subregions.")

    def log_totals(self,lp):
        if lp.mode & mode_log_summary:
            self._log_totals(lp)

    def _log_totals(self,lp):
        s = "."*lp.lvl_cnt
        num_interior_roots_fnd = len(self.interior_mull_new)
        num_sub_roots_fnd = len(self.interior_all_subs)
        num_roots_found = num_interior_roots_fnd+\
                          num_sub_roots_fnd+\
                          self.num_added_conjs
        if self.num_added_conjs != 0:
            print s+"    "+str(self.num_added_conjs)+" added conjugates."
        msg = s+str(len(self.end_unique))+" total in this region. "
        if self.num_added_conjs != 0:
            msg += "Including "+str(self.num_added_conjs)+" added conjugates. "
            if lp.lvl_cnt>0:
                msg += str(len(self.end_unique+self.known))+\
                       " accumulated so far."
        print msg

    def log_notes(self,lp,state):
        if lp.mode & mode_log_summary:
            self._log_notes(lp,state)

    def _log_notes(self,lp,state):
        s = "."*lp.lvl_cnt
        if state & warn_max_steps_reached:
            print s+"WARNING: max_steps limit hit."
        else:
            print s+"All roots found for max_steps and I0_tol parameters."

        if lp.mode & mode_log_notes:
            if state == 0:
                print s+"No notes."
            else:
                print s+"Following notes occurred at least once:"
                if state & note_imprecise_roots:
                    print s+"  -Imprecise number of roots in region."
                if state & note_no_bnd_muller_root:
                    print (s+"  -No boundary muller root found with " +
                            "specified parameters.")
                if state & note_bnd_muller_exception:
                    print s+"  -Exception during Muller routine."
                if state & note_no_int_muller_root:
                    print (s+"  -No interior muller root found with " +
                            "specified parameters.")
                if state & note_not_all_interior_fnd:
                    print s+"  -Not all predicted interior roots found."
                if state & note_root_sub_div_by_zero:
                    print s+"  -Division by zero when subtracting roots."

    def log_roots(self,lp):
        if lp.mode & mode_log_debug:
            self._log_roots(lp)

    def _log_roots(self,lp):
        s = "."*lp.lvl_cnt
        print s+"\nBOUNDARY:"
        print s+"Muller:\n"+str(np.array(self.boundary_all))
        print (s+"New:\n"+str(np.array(self.boundary_new)))
        print s+"Within:\n"+str(np.array(self.boundary_within))
        print s+"Subtracted:\n"+str(np.array(self.subtracted))
        print s+"Failed:\n"+str(np.array(self.boundary_failed_mulls))
        print "\nINTERIOR:"
        print s+"Rough Poly:\n"+str(self.interior_rough)
        print s+"Muller:\n"+str(np.array(self.interior_mull_all))
        print s+"Purged:\n"+str(np.array(self.interior_mull_purged))
        print s+"New:\n"+str(np.array(self.interior_mull_new))
        print s+"Failed:\n"+str(np.array(self.interior_failed_mulls))
        print "\nSUBREGIONS:"
        print s+"New:\n"+str(np.array(self.interior_all_subs))
        print "\nFINAL:"
        print s+"New:\n"+str(np.array(self.end_unique))

    def log_close_region(self,lp):
        if lp.mode & mode_log_summary:
            print "-"*lp.lvl_cnt

    def at_boundary(self,lp,b):
        outliers = find_maxes(map(abs,b.y))
        state = 0
        for index in outliers:
            state |= locate_muller_root(lp,b.c[index-2],b.c[index+2],
                                        b.c[index]/2,self.boundary_all,
                                        self.boundary_failed_mulls)
        self.boundary_purged = purge(self.boundary_all,gp.dist_eps)
        self.boundary_new = get_unique(self.boundary_purged,self.known,
                                          gp.dist_eps)

        self.boundary_and_known = purge(self.boundary_new+self.known,
                                        gp.dist_eps)

        self.boundary_within = inside_boundary(self.boundary_purged,
                                               b.rx,b.ry,b.rw,b.rh)

        # We don't need the roots far outside the boundary
        self.subtracted = inside_boundary(self.boundary_and_known,
                                          b.rx,b.ry,b.rw+gp.bnd_thres,
                                          b.rh+gp.bnd_thres)
        self.residues_subtracted = \
            residues(b.f_frac,self.subtracted,gp.lmt_N,gp.lmt_eps)
        return state

    def is_polysolve_required(self,num_pred_roots):
        return num_pred_roots <= gp.max_order and num_pred_roots >= 1

    def from_polysolve(self,lp,b,num_pred_roots):
        self.interior_rough = locate_poly_roots(b.y_smooth,b.c,num_pred_roots)
        state = 0
        ##TODO: best way to pick points for Muller method below
        for root in self.interior_rough:
            state |= locate_muller_root(lp,root-gp.mul_off,root+gp.mul_off,root,
                                        self.interior_mull_all,
                                        self.interior_failed_mulls)
        self.interior_mull_purged = purge(self.interior_mull_all,gp.dist_eps)

        self.interior_mull_within = inside_boundary(self.interior_mull_purged,
                                                    b.rx,b.ry,b.rw,b.rh)
        self.interior_mull_new = get_unique(self.interior_mull_within,
                                               self.boundary_new,gp.dist_eps)
        return state

    def finialise_region_roots(self,lp,b,I0,sub_required):
        # This should be the only place where conjugate addition is required
        # Only the newly found roots here. Already know can be added later.
        self.region = \
            purge(self.boundary_new+self.interior_mull_new,gp.dist_eps)
        self.region_corrected,self.num_added_conjs = correct_roots(lp,b,self.region)
        self.log_region(lp,b,I0,sub_required)

    def finialise_end_roots(self,lp,state):
        # Only return new roots. They'll be added to known roots in the parent.
        self.end = self.region_corrected+self.interior_all_subs
        self.end_purged = purge(self.end,gp.dist_eps)
        self.end_unique = get_unique(self.end_purged,self.known,gp.dist_eps)
        self.log_totals(lp)
        self.log_notes(lp,state)
        self.log_roots(lp)
        self.log_close_region(lp)

class boundary:
    def __init__(self,rx,ry,rw,rh):
        self.rx = rx
        self.ry = ry
        self.rw = rw
        self.rh = rh
        self.c = get_boundary(rx,ry,rw,rh,gp.N)
        self.f_frac = lambda z: gp.fp(z)/(2j*np.pi*gp.f(z))
        self.y = [self.f_frac(z) for z in self.c]
        self.max_ok = abs(gp.outlier_coeff*get_max(self.y))

    def smoothed(self,lp,roots):
        self.y_smooth = []
        state = 0
        for y_el,z_el in zip(self.y,self.c):
            val, ret = new_f_frac_safe(self.f_frac,z_el,
                                       roots.residues_subtracted,
                                       roots.subtracted,self.max_ok,y_el,
                                       gp.lmt_N,gp.lmt_eps)
            self.y_smooth.append(val)
            if not ret:
                state |= gp.handle_state_change(lp,note_root_sub_div_by_zero)
        return state


def is_subdivision_required(roots,I0,num_pred_roots):
    # Don't count the added conjs at this stage, just pass them to the 
    # subregions. Otherwise will confuse the routine.
    all_int_fnd = len(roots.interior_mull_new)>=num_pred_roots
    roche_accurate = gp.is_roche_accurate(I0,num_pred_roots)
    if num_pred_roots>gp.max_order:
        return True
    if not roche_accurate:
        return True
    if not all_int_fnd:
        return True
    return False

def calculate_for_subregions(lp,b,roots):
    x_list = [b.rx - b.rw / 2.,b.rx - b.rw / 2., 
              b.rx + b.rw / 2.,b.rx + b.rw / 2.]
    y_list = [b.ry - b.rh / 2.,b.ry + b.rh / 2.,
              b.ry - b.rh / 2.,b.ry + b.rh / 2.]
    num_regions = len(x_list)
    if lp.mode & mode_log_recursive:
        new_mode = lp.mode
    else:
        new_mode = lp.mode & mode_log_switch
    state = 0
    known_roots = roots.region + roots.known
    for x,y in zip(x_list,y_list):
        ret,sub_roots = droots(gp.f,gp.fp,x,y,b.rw/2.,b.rh/2.,gp.N,
                               lp.max_steps-1,new_mode,known_roots,lp.lvl_cnt+1)
        if not ret:
            state |= warn_max_steps_reached
        roots.interior_all_subs.extend(sub_roots)
    return state,num_regions

class global_parameters:
    def __init__(self): 
        self.f = None
        self.fp = None

        self.N = 10
        self.outlier_coeff = 100.
        self.max_order = 10
        self.I0_tol = 5e-3

        self.mul_N = 400
        self.mul_ltol = 1e-12
        self.mul_htol = 1e-12
        self.mul_off = 1e-5

        self.conj_min_i = 1e-8

        self.dist_eps = 1e-7
        self.lmt_N = 10
        self.lmt_eps = 1e-3
        self.bnd_thres = 2.

    def set_delves_routine_parameters(self,outlier_coeff,max_order,I0_tol):
        self.outlier_coeff = outlier_coeff
        self.max_order = max_order
        self.I0_tol = I0_tol

    def set_muller_parameters(self,mul_N,mul_ltol,mul_htol,mul_off):
        self.mul_N = mul_N
        self.mul_ltol = mul_ltol
        self.mul_htol = mul_htol
        self.mul_off = mul_off

    def set_mode_parameters(self, conj_min_i):
        self.conj_min_i = conj_min_i

    def set_advanced_parameters(self,dist_eps,lmt_N,lmt_eps,bnd_thres):
        self.dist_eps = dist_eps
        self.lmt_N = lmt_N
        self.lmt_eps = lmt_eps
        self.bnd_thres = bnd_thres

    def set_calc_parameters(self,f,fp,N):
        self.f = f
        self.fp = fp
        self.N = N

    def print_region_string(self,lp,b):
        if lp.mode & mode_log_summary:
            s = "-"*lp.lvl_cnt
            print ("\n"+s+"Region(rx,ry,rw,rh): "+str(b.rx)+" "+str(b.ry)+\
                   " "+str(b.rw)+" "+str(b.rh))

    def handle_state_change(self,lp,state):
        s = "."*lp.lvl_cnt
        imprecise_roots = s+"Note!! Number of roots may be imprecise for " + \
                          "this N. Increase N for greater precision."
        max_steps_exceeded = s+"Note!! max_steps exceeded. Some interior " + \
                             "roots might be missing."
        no_bnd_muller_root = s+"Note!! Boundary Muller failed to converge." 
        bnd_muller_exception = s+"Note!! Exception during boundary Muller." 
        no_int_muller_root = s+"Note!! Interior Muller failed to converge." 
        int_muller_exception = s+"Note!! Exception during interior Muller." 
        not_all_interior_fnd = s+"Note!! Not all predicted interior roots " + \
                               "found."
        root_subtraction_division_by_zero = s+"Note!! Division by zero " + \
                                            "during root subtraction."

        if lp.mode & mode_log_all_notes:
            if state == note_imprecise_roots:
                print imprecise_roots
            elif state == warn_max_steps_reached:
                print max_steps_exceeded
            elif state == note_no_bnd_muller_root:
                print no_bnd_muller_root
            elif state == note_bnd_muller_exception:
                print bnd_muller_exception
            elif state == note_no_int_muller_root:
                print no_int_muller_root
            elif state == note_int_muller_exception:
                print int_muller_exception
            elif state == note_not_all_interior_fnd:
                print not_all_interior_fnd
            elif state == note_root_sub_div_by_zero:
                print root_subtraction_division_by_zero
        return state

    def is_roche_accurate(self,I0,num_pred_roots):
        return abs(num_pred_roots-I0)<self.I0_tol

class local_parameters:
    def __init__(self,max_steps,mode,lvl_cnt):
        self.max_steps = max_steps
        self.mode = mode
        self.lvl_cnt = lvl_cnt

gp = global_parameters()

def set_delves_routine_parameters(outlier_coeff=100.,max_order=10,I0_tol=5e-3):
    '''
    Set primary routine arguments

    Args:
        outlier_coeff (optional[float]): multiplier for coefficient used when 
            subtracting poles to improve numerical stability. 
            See new_f_frac_safe.

        max_order (optional[int]): Max power of polynomial to be solved,
            otherwise routine will recurse.

        I0_tol (optional[float]): The Roche should return an integer number of 
            roots if the calculation has been performed accurately. This 
            parameter is the decimal deviation from integer required to render
            the calculation inaccurate. When inaccurate and the number of 
            iterations is less than max_step the routine will recurse.
    '''
    gp.set_delves_routine_parameters(outlier_coeff,max_order,I0_tol)

def set_muller_parameters(mul_N=400,mul_ltol=1e-12,mul_htol=1e-12,mul_off=1e-5):
    '''
    Set arguments related to the muller routine

    Args:
        mul_N (optional[int]): maximum number of iterations for muller.

        mul_ltol (optional[float]): muller low (strict) tolerance.

        mul_htol (optional[float]): muller high (relaxed) tolerance.

        mul_off (optional[float]): muller point offset.

    '''
    gp.set_muller_parameters(mul_N,mul_ltol,mul_htol,mul_off)

def set_mode_parameters(conj_min_i=1e-8):
    '''
    Set advanced arguments

    Args:
        conj_min_i (optional[float]): If mode==mode_add_conjs then this parameter
            determines the minimum distance from the real axis a root must lie
            before being considered as having a conjugate partner.

    '''
    gp.set_mode_parameters(conj_min_i)
    
def set_advanced_parameters(dist_eps=1e-7,lmt_N=10,lmt_eps=1e-3,bnd_thres=2.):
    '''
    Set advanced arguments

    Args:
        dist_eps (optional[float]): epsilon used when distinguishing roots
            using absolute values. Within this they are judged the same root.

        lmt_N (int): number of points used in the estimate when calculaing the
            residues.

        lmt_eps (optional[float]): distance from z0 at which estimating points 
            are placed for the subtraction.

        bnd_thres (optional[float]): The perpendicular distance outwards from 
            the region boundary within which roots must lie to be considered
            boundary.

    '''
    gp.set_advanced_parameters(dist_eps,lmt_N,lmt_eps,bnd_thres)

def droots(f,fp,rx,ry,rw,rh,N=10,max_steps=5,mode=mode_off,
           known_roots=[],lvl_cnt=0):
    '''
    I assume f is analytic with simple (i.e. order one) zeros.

    TODO:
    save values along edges if iterating to a smaller rectangle
    extend to other kinds of functions, e.g. function with non-simple zeros.

    Args:
        f (function): the function for which the roots (i.e. zeros) will be
            found.

        fp (function): the derivative of f.

        rx,ry (floats): The center of the rectangle in the complex
            plane.

        rw,rh (floats): half the width and height of the rectangular
            region.

        N (optional[int]): Number of points to sample per edge

        max_steps (optional[int]): Number of iterations allowed for algorithm to
            repeat on smaller rectangles.

        mode (optional[int]): This sets the calculation mode. See top of file
            for available modes.

        known_roots (internal[list of complex numbers]): Roots of f that are
            already known. Used when recursing

        lvl_cnt (internal[int]): number of times the routine has recursed.

    Returns:h
        A boolean indicating if all roots found for the supplied max_steps and 
            I0_tol parameters. A list of roots for the function f inside the 
            rectangle determined by the values rx,ry,rw and rh.
    '''
    lp = local_parameters(max_steps,mode,lvl_cnt)
    gp.set_calc_parameters(f,fp,N)
    roots = root_container(known_roots)
    state = 0
    num_regions = 0

    b = boundary(rx,ry,rw,rh)
    state |= roots.at_boundary(lp,b)

    state |= b.smoothed(lp,roots)
    I0 = integrate.trapz(b.y_smooth, b.c)  # Approx num of roots not subtracted
    num_pred_roots = int(math.ceil(abs(I0)-gp.I0_tol))

    if abs(num_pred_roots-I0)>gp.I0_tol:
        state |= gp.handle_state_change(lp,note_imprecise_roots)

    if roots.is_polysolve_required(num_pred_roots):
        state |= roots.from_polysolve(lp,b,num_pred_roots)

    sub_required = is_subdivision_required(roots,I0,num_pred_roots)
    sub_possible = lp.max_steps!=0
    roots.finialise_region_roots(lp,b,I0,sub_required and sub_possible)
    if sub_required:
        if sub_possible:
            new_notes,num_regions = calculate_for_subregions(lp,b,roots)
            state |= (new_notes&warn_max_steps_reached)
            roots.log_sub_region(lp,num_regions)
        else:
            state |= gp.handle_state_change(lp,warn_max_steps_reached)

    tot_interior_found = len(roots.region_corrected+roots.interior_all_subs)
    if tot_interior_found != num_pred_roots:
        state |= gp.handle_state_change(lp,note_not_all_interior_fnd)
    
    roots.finialise_end_roots(lp,state)

    return not state&warn_max_steps_reached,roots.end_unique
