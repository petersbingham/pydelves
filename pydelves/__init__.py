# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 20:15:35 2015

@author: gil
@title: Rootfinder
Modified by P Bingham October-December 2017

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

#Since muller uses f(x)<tol then if differential is very high small displacement
#from root can give much large f(x). Since we know the rough value of the roche
#root we can compare to that instead (or as well). Two options here. If the 
#good_roche variant is selected then will only accept if the routine had a good
#I0.
mode_accept_int_muller_close_to_good_roche = 0x1
mode_accept_int_muller_close_to_any_roche = 0x2
#Similarly, for the boundary roots. Note that boundary roots have a greater
#tendency to wander from their starting than the roche roots. Recommended to
#keep the following mode off and use mode_use_stripped_subtraction instead.
mode_accept_bnd_muller_close_to_start = 0x4
#Or the following option to accept all mullers, without any check. Good when
#roots can be missing and and some external check is applied.
mode_accept_all_mullers = 0x8

#Due to tendency to wander it's harder to verify the boundary roots. The 
#following mode extends the supplied region by twice bnd_thres and carries out 
#the bounadary root calculation and roche integration at the midline between 
#the supplied and the extended regions. Only boundary roots between the 
#supplied region and the extended region are used. Only roche roots inside the
#supplied region are returned to the user.
mode_use_stripped_subtraction = 0x10

#Recursion is expensive. If the routine is selected to not recurse on either of
#the two conditions below then a warning will be returned from the routine, as
#can't guarantee that all of the roots have been found.
mode_dont_recurse_on_inaccurate_roche = 0x20
mode_dont_recurse_on_not_all_interior_found = 0x40

#If function is a polynomial with real coefficients then roots will occur in 
#a+ib, a-ib pairs. If mode==mode_add_conjs then routine will take advantage of 
#this by automatically adding missing partners before recursing.
mode_add_conjs = 0x80

#Modes for the different log modes.
mode_log_recursive = 0x200
mode_log_summary = 0x400
mode_log_verbose = 0x800
mode_log_debug = 0x1000

#Used for switching the log off on recursion when not int mode_log_recursive:
mode_log_switch = 0x1FF

warn_root_check_disabled = 0x1
warn_inaccurate_roche = 0x2
warn_could_not_locate_roche_root = 0x4

note_inaccurate_roche = 0x8
note_could_not_locate_roche_root = 0x10
note_muller_fail_1st = 0x20
note_muller_fail_2nd = 0x40
note_muller_exception = 0x80
note_root_sub_div_by_zero = 0x100

#Used for switching out notes from warnings:
mode_warn_switch = 0x7

default_N = 500 
default_max_steps = 5
default_mode = mode_off 
default_outlier_coeff = 100.
default_max_order = 10
default_I0_tol = 5e-3
default_mul_N = 400
default_mul_fzltol = 1e-12
default_mul_fzhtol = 1e-12
default_mul_off = 1e-5
default_mul_ztol = 1e-4
default_conj_min_i = 1e-8
default_dist_eps = 1e-7
default_lmt_N = 10
default_lmt_eps = 1e-3
default_bnd_thres = 2.

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

def locate_muller_root(lp,x1,x2,x3,found_roots,failed_roots):
    status = 0
    try:          
        mull_root,ret = muller(x1,x2,x3,gp.f,gp.mul_N,gp.mul_fzltol,gp.mul_fzhtol)
        if lp.mode & mode_accept_all_mullers or ret:
            found_roots.append(mull_root)
        else:
            failed_roots.append((x3,mull_root))
            status |= note_muller_fail_1st
    except:
        status |= note_muller_exception
    return status

def check_against_start_roots(failed_roots,reaccepted_roots):
    status = 0
    for rough_root, mull_root in failed_roots:
        if abs(rough_root-mull_root) < gp.mul_ztol:
            reaccepted_roots.append(mull_root)
        else:
            status |= note_muller_fail_2nd
    return status

def check_against_rough_roots(lp,I0,failed_roots,reaccepted_roots):
    check = False
    if lp.mode & mode_accept_int_muller_close_to_any_roche:
        check = True
    elif lp.mode & mode_accept_int_muller_close_to_good_roche:
        if gp.is_roche_accurate(I0):
            check = True
    if check:
        return check_against_start_roots(failed_roots,reaccepted_roots)
    return 0

def check_against_bnd_start_roots(lp,failed_roots,reaccepted_roots):
    check = False
    if lp.mode & mode_accept_bnd_muller_close_to_start:
        return check_against_start_roots(failed_roots,reaccepted_roots)
    return 0

def correct_roots(lp,b,roots):
    if lp.lvl_cnt == 0:
        roots_mod = inside_boundary(roots,*b.reg_i())
    else:
        roots_mod = roots
    conjs_added = 0
    if lp.mode & mode_add_conjs:
        roots_miss_conj = add_miss_conjs(roots_mod,gp.dist_eps,gp.conj_min_i)
        if lp.lvl_cnt == 0:
            roots_final = inside_boundary(roots_missing_conjs,*b.reg_i())
        else:
            roots_final = roots_missing_conjs
        conjs_added = len(roots_final)-len(roots_mod)
    else:
        roots_final = list(roots_mod)
    return roots_final, conjs_added

class root_container:
    def __init__(self, known):
        self.known = known

        self.boundary_outliers = []
        self.boundary_passed_fz = []
        self.boundary_failed_fz = []  
        self.boundary_passed_z = []
        self.boundary_all = []
        self.boundary_purged = [] 
        self.boundary_new = []     
        self.boundary_within = []
        self.boundary_and_known = []
        self.residues_subtracted = []

        self.interior_mull_all = []
        self.interior_failed_mulls = []
        self.interior_reaccepted = []
        
        self.interior_rough = []
        self.interior_passed_fz = []
        self.interior_failed_fz = []  
        self.interior_passed_z = []
        self.interior_all = []        
        self.interior_purged = []
        self.interior_within = []
        self.interior_new = []

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
        b.print_region_string(lp)
        num_known_roots = len(self.known)
        num_interior_roots_fnd = len(self.interior_new)
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
        num_interior_roots_fnd = len(self.interior_new)
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

    def log_notes(self,lp,status):
        if lp.mode & mode_log_summary:
            self._log_notes(lp,status)

    def _log_notes(self,lp,status):
        s = "."*lp.lvl_cnt
        print_status(status, s)

    def log_roots(self,lp):
        if lp.mode & mode_log_debug:
            self._log_roots(lp)

    def _log_roots(self,lp):
        s = "."*lp.lvl_cnt
        
        print s+"\nKNOWN:"
        print s+"Known:\n"+str(np.array(self.known))

        print s+"\nBOUNDARY:"
        print s+"Outliers:\n"+str(np.array(self.boundary_outliers))
        print s+"Passed fz:\n"+str(np.array(self.boundary_passed_fz))
        if len(self.boundary_failed_fz) > 0:
            print s+"Failed fz rough:\n"+str(np.array(zip(*self.boundary_failed_fz)[0]))
            print s+"Failed fz mull:\n"+str(np.array(zip(*self.boundary_failed_fz)[1]))
        else:
            print s+"Failed fz:\n[]"
        k = "YES" if len(self.boundary_passed_z) > 0 else "NO"
        print s+"Passed z (" + k +")"+":\n" + str(np.array(self.boundary_passed_z))
        print s+"All Passed:\n"+str(np.array(self.boundary_all))
        print s+"New:\n"+str(np.array(self.boundary_new))
        print s+"Within:\n"+str(np.array(self.boundary_within))
        print s+"Subtracted:\n"+str(np.array(self.boundary_and_known))
        
        print "\nINTERIOR:"
        print s+"Rough:\n"+str(np.array(self.interior_rough))
        print s+"Passed fz:\n"+str(np.array(self.interior_passed_fz))
        if len(self.interior_failed_fz) > 0:
            print s+"Failed fz rough:\n"+str(np.array(zip(*self.interior_failed_fz)[0]))
            print s+"Failed fz mull:\n"+str(np.array(zip(*self.interior_failed_fz)[1]))
        else:
            print s+"Failed fz:\n[]"
        k = "YES" if len(self.interior_passed_z) > 0 else "NO"
        print s+"Passed z (" + k +")"+":\n" + str(np.array(self.interior_passed_z))
        print s+"All Passed:\n"+str(np.array(self.interior_all))
        print s+"Purged:\n"+str(np.array(self.interior_purged))
        print s+"Within:\n"+str(np.array(self.interior_within))
        print s+"New:\n"+str(np.array(self.interior_new))
            
        print "\nSUBREGIONS:"
        print s+"New:\n"+str(np.array(self.interior_all_subs))
        print "\nFINAL:"
        print s+"New:\n"+str(np.array(self.end_unique))

    def log_close_region(self,lp):
        if lp.mode & mode_log_summary:
            print "-"*lp.lvl_cnt

    def at_boundary(self,lp,b):
        outlier_indices = find_maxes(map(abs,b.y))
        status = 0
        self.boundary_outliers = []
        for i in outlier_indices:
            self.boundary_outliers.append(b.c[i])
            status |= locate_muller_root(lp,b.c[i-1],b.c[i+1],
                                        b.c[i],self.boundary_passed_fz,
                                        self.boundary_failed_fz)
        status |= check_against_bnd_start_roots(lp,self.boundary_failed_fz,
                                               self.boundary_passed_z)
        self.boundary_all = self.boundary_passed_fz + self.boundary_passed_z
        self.boundary_purged = purge(self.boundary_all,gp.dist_eps)
        
        # We don't need the roots far outside the boundary.
        self.boundary_useful = inside_boundary(self.boundary_purged,
                                          *b.reg_sub(lp))
        if lp.mode & mode_use_stripped_subtraction:
            self.boundary_useful = outside_boundary(self.boundary_useful,
                                                    *b.reg_i())        

        self.boundary_new = get_unique(self.boundary_useful,self.known,
                                          gp.dist_eps)
        #For stripped mode known is only ever roche roots. Since the roche root
        #will have passed a test we can use for subtraction in the internal 
        #region:
        self.boundary_and_known = purge(self.boundary_new+self.known,
                                        gp.dist_eps)

        self.residues_subtracted = \
            residues(b.f_frac,self.boundary_and_known,gp.lmt_N,gp.lmt_eps)

        #For debug info only:
        self.boundary_within = inside_boundary(self.boundary_purged,*b.reg_i())
        return status

    def is_polysolve_required(self,lp,I0,num_pred_roots):
        roche_accurate = gp.is_roche_accurate(I0)
        if not roche_accurate and\
           not lp.mode & mode_dont_recurse_on_inaccurate_roche:
            return False
        return num_pred_roots <= gp.max_order and num_pred_roots >= 1

    def from_polysolve(self,lp,b,num_pred_roots,I0):
        self.interior_rough = locate_poly_roots(b.y_smooth,b.c,num_pred_roots)
        status = 0
        ##TODO: best way to pick points for Muller method below
        for rough_root in self.interior_rough:
            status |= locate_muller_root(lp,rough_root-gp.mul_off,
                                        rough_root+gp.mul_off,rough_root,
                                        self.interior_passed_fz,
                                        self.interior_failed_fz)
        status |= check_against_rough_roots(lp,I0,self.interior_failed_fz,
                                            self.interior_passed_z)
        self.interior_all = self.interior_passed_fz + self.interior_passed_z

        self.interior_purged = purge(self.interior_all,gp.dist_eps)

        #It's possible the muller wandered outside the region:
        if not lp.mode & mode_use_stripped_subtraction:
            self.interior_within = inside_boundary(self.interior_purged,
                                                   *b.reg_i())
        else:
            #Use all when stripped incase we find one that happens to be in an
            #adjacent region (since overlapping when subregions) Any roots
            #outside the specified region will be removed in correct_roots.
            self.interior_within = inside_boundary(self.interior_purged,
                                                   *b.reg_m())

        #boundary_and_known are the subtracted roots, so we should not find
        #then again:
        self.interior_new = get_unique(self.interior_within,
                                       self.boundary_and_known,gp.dist_eps)
        return status

    def finialise_region_roots(self,lp,b,I0,sub_required):
        # This should be the only place where conjugate addition is required
        # Only the newly found roots here. Already known will be added later.
        root_set = list(self.interior_new)
        if not lp.mode & mode_use_stripped_subtraction:
            root_set.extend(self.boundary_new)
        self.region = purge(root_set,gp.dist_eps)
        self.region_corrected,self.num_added_conjs = correct_roots(lp,b,
                                                                   self.region)
        self.log_region(lp,b,I0,sub_required)

    def finialise_end_roots(self,lp,status):
        # Only return new roots. They'll be added to known roots in the parent.
        self.end = self.region_corrected+self.interior_all_subs
        self.end_purged = purge(self.end,gp.dist_eps)
        self.end_unique = get_unique(self.end_purged,self.known,gp.dist_eps)
        self.log_totals(lp)
        self.log_notes(lp,status)
        self.log_roots(lp)
        self.log_close_region(lp)

class boundary:
    def __init__(self,lp,rx,ry,rw,rh):
        self.rx = rx
        self.ry = ry
        
        self.rw_i = rw
        self.rh_i = rh
        
        self.rw_m = rw+gp.bnd_thres
        self.rh_m = rh+gp.bnd_thres
        
        self.rw_o = rw+2.*gp.bnd_thres
        self.rh_o = rh+2.*gp.bnd_thres

        self.c = get_boundary(gp.N,*self.reg_calcs(lp))
        self.f_frac = lambda z: gp.fp(z)/(2j*np.pi*gp.f(z))
        self.y = [self.f_frac(z) for z in self.c]
        self.max_ok = abs(gp.outlier_coeff*get_max(self.y))

    def reg_i(self):
        return self.rx, self.ry, self.rw_i, self.rh_i

    def reg_m(self):
        return self.rx, self.ry, self.rw_m, self.rh_m

    def reg_o(self):
        return self.rx, self.ry, self.rw_o, self.rh_o

    def reg_calcs(self,lp):
        if lp.mode & mode_use_stripped_subtraction:
            return self.reg_m()
        else:
            return self.reg_i()

    def reg_sub(self,lp):
        if lp.mode & mode_use_stripped_subtraction:
            return self.reg_o()
        else:
            return self.reg_m()

    def smoothed(self,roots):
        self.y_smooth = []
        status = 0
        for y_el,z_el in zip(self.y,self.c):
            val, ret = new_f_frac_safe(self.f_frac,z_el,
                                       roots.residues_subtracted,
                                       roots.boundary_and_known,
                                       self.max_ok,y_el,
                                       gp.lmt_N,gp.lmt_eps)
            self.y_smooth.append(val)
            if not ret:
                status |= note_root_sub_div_by_zero
        return status

    def get_subregions(self):
        x_list = [self.rx - self.rw_i / 2., self.rx - self.rw_i / 2., 
                  self.rx + self.rw_i / 2., self.rx + self.rw_i / 2.]
        y_list = [self.ry - self.rh_i / 2., self.ry + self.rh_i / 2.,
                  self.ry - self.rh_i / 2., self.ry + self.rh_i / 2.]
        return x_list, y_list

    def print_region_string(self,lp):
        if lp.mode & mode_log_summary:
            s = "-"*lp.lvl_cnt
            print ("\n"+s+"Region(rx,ry,rw,rh): "+\
                   str(self.rx)+" "+str(self.ry)+\
                   " "+str(self.rw_i)+" "+str(self.rh_i))


def all_interior_found(roots,num_pred_roots):
    return len(roots.interior_new)>=num_pred_roots

def do_subcalculation(lp,roots,I0,num_pred_roots):
    # Don't count the added conjs at this stage, just pass them to the 
    # subregions. Otherwise will confuse the routine.
    all_int_fnd = all_interior_found(roots,num_pred_roots)
    roche_accurate = gp.is_roche_accurate(I0)
    ret = False
    if num_pred_roots>gp.max_order:
        ret = True
    if not roche_accurate and\
       not lp.mode & mode_dont_recurse_on_inaccurate_roche:
        ret = True
    if not all_int_fnd and\
       not lp.mode & mode_dont_recurse_on_not_all_interior_found:
        ret = True
    return ret and lp.max_steps!=0

def calculate_for_subregions(lp,b,roots):
    if lp.mode & mode_log_recursive:
        new_mode = lp.mode
    else:
        new_mode = lp.mode & mode_log_switch
    status = 0
    known_roots = roots.region + roots.known
    num_regions = 0
    for x,y in zip(*b.get_subregions()):
        new_state,sub_roots = droots(gp.f,gp.fp,x,y,b.rw_i/2.,b.rh_i/2.,gp.N,
                               lp.max_steps-1,new_mode,known_roots,lp.lvl_cnt+1)
        status |= new_state
        roots.interior_all_subs.extend(sub_roots)
        num_regions += 1
    if lp.lvl_cnt == 0:
        #Need this here since subregion roots added after correct_roots.
        roots.interior_all_subs = inside_boundary(roots.interior_all_subs,
                                                  *b.reg_i())
    return status,num_regions

class global_parameters:
    def __init__(self): 
        self.f = None
        self.fp = None
        self.set_defaults()

    def set_defaults(self):
        self.N = default_N
        self.outlier_coeff = default_outlier_coeff
        self.max_order = default_max_order
        self.I0_tol = default_I0_tol

        self.mul_N = default_mul_N
        self.mul_fzltol = default_mul_fzltol
        self.mul_fzhtol = default_mul_fzhtol
        self.mul_off = default_mul_off

        self.conj_min_i = default_conj_min_i
        self.mul_ztol = default_mul_ztol

        self.dist_eps = default_dist_eps
        self.lmt_N = default_lmt_N
        self.lmt_eps = default_lmt_eps
        self.bnd_thres = default_bnd_thres

    def set_delves_routine_parameters(self,outlier_coeff,max_order,I0_tol):
        self.outlier_coeff = outlier_coeff
        self.max_order = max_order
        self.I0_tol = I0_tol

    def set_muller_parameters(self,mul_N,mul_fzltol,mul_fzhtol,mul_off):
        self.mul_N = mul_N
        self.mul_fzltol = mul_fzltol
        self.mul_fzhtol = mul_fzhtol
        self.mul_off = mul_off

    def set_mode_parameters(self, mul_ztol, conj_min_i):
        self.conj_min_i = conj_min_i
        self.mul_ztol = mul_ztol

    def set_advanced_parameters(self,dist_eps,lmt_N,lmt_eps,bnd_thres):
        self.dist_eps = dist_eps
        self.lmt_N = lmt_N
        self.lmt_eps = lmt_eps
        self.bnd_thres = bnd_thres

    def set_calc_parameters(self,f,fp,N):
        self.f = f
        self.fp = fp
        self.N = N

    def handle_state(self,lp,status):
        if lp.mode & mode_log_verbose:
            s = "."*lp.lvl_cnt
            print s+"State appended: " +str(status)
        return status

    def is_roche_accurate(self,I0):
        return abs(abs(I0)-round(abs(I0)))<self.I0_tol

    def update_state_for_subcalc(self,lp,do_subcalc,roots,num_pred_roots,I0):
        status = 0
        if not gp.is_roche_accurate(I0):
            if do_subcalc:
                status = gp.handle_state(lp,note_inaccurate_roche)
            else:
                status = gp.handle_state(lp,warn_inaccurate_roche)
        elif not all_interior_found(roots,num_pred_roots):
            if do_subcalc:
                status = gp.handle_state(lp,note_could_not_locate_roche_root)
            else:
                status = gp.handle_state(lp,warn_could_not_locate_roche_root)
        return status

class local_parameters:
    def __init__(self,max_steps,mode,lvl_cnt):
        self.max_steps = max_steps
        self.mode = mode
        self.lvl_cnt = lvl_cnt
        
    def init_state(self):
        status = 0
        if self.mode & mode_accept_all_mullers:
            status = warn_root_check_disabled
        return status

gp = global_parameters()

def set_delves_routine_parameters(outlier_coeff=default_outlier_coeff,
                                  max_order=default_max_order,
                                  I0_tol=default_I0_tol):
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

def set_muller_parameters(mul_N=default_mul_N,mul_fzltol=default_mul_fzltol,
                          mul_fzhtol=default_mul_fzhtol,
                          mul_off=default_mul_off):
    '''
    Set arguments related to the muller routine

    Args:
        mul_N (optional[int]): maximum number of iterations for muller.

        mul_fzltol (optional[float]): muller low (strict) tolerance.

        mul_fzhtol (optional[float]): muller high (relaxed) tolerance.

        mul_off (optional[float]): muller point offset.

    '''
    gp.set_muller_parameters(mul_N,mul_fzltol,mul_fzhtol,mul_off)

def set_mode_parameters(mul_ztol=default_mul_ztol,
                        conj_min_i=default_conj_min_i):
    '''
    These parameters are only relevant if the related mode is set.

    Args:
        conj_min_i (optional[float]): If mode is:
            mode_add_conjs
            then this parameter determines the minimum distance from the real 
            axis a root must lie before being considered as having a conjugate 
            partner.

        mul_ztol (optional[float]): If mode is: 
            mode_accept_bnd_muller_close_to_start
            mode_accept_int_muller_close_to_good_roche
            mode_accept_int_muller_close_to_any_roche 
            then this parameter determines the distance from the rough that the 
            returned (failed muller) can be within to be re-accepted as a good 
            root.
    '''
    gp.set_mode_parameters(mul_ztol, conj_min_i)
    
def set_advanced_parameters(dist_eps=default_dist_eps,lmt_N=default_lmt_N,
                            lmt_eps=default_lmt_eps,
                            bnd_thres=default_bnd_thres):
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

def set_default_parameters():
    gp.set_defaults()

def was_warning(status):
    return bool(status&mode_warn_switch)
    
def print_status(status, s=""):
    if status & warn_inaccurate_roche:
        print s+"WARNING: inaccurate final roche."
    if status & warn_could_not_locate_roche_root:
        print s+"WARNING: could not locate roche root."
    if status & warn_root_check_disabled:
        print s+"WARNING: root checks disabled."
    if not status&mode_warn_switch:
        print s+"All roots found for max_steps and I0_tol parameters."
    print "Status Num: " + str(status)
    
def droots(f,fp,rx,ry,rw,rh,N=default_N,max_steps=default_max_steps,
           mode=default_mode,known_roots=[],lvl_cnt=0):
    '''
    I assume f is analytic with simple (i.e. order one) zeros.

    TODO:
    save values along edges if iterating to a smaller rectangle
    extend to other kinds of functions, e.g. function with non-simple zeros.

    Args:
        f (function): the function for which the roots (i.e. zeros) will be
            found. It is highly recommended (especially if the f is expensive)
            to cache values.

        fp (function): the derivative of f. It is highly recommended 
            (especially if the fp is expensive) to cache values.

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
    status = lp.init_state()
    num_regions = 0

    b = boundary(lp,rx,ry,rw,rh)
    status |= gp.handle_state(lp,roots.at_boundary(lp,b))

    status |= gp.handle_state(lp,b.smoothed(roots))
    I0 = integrate.trapz(b.y_smooth,b.c)  # Approx num of roots not subtracted
    num_pred_roots = int(math.ceil(abs(I0)-gp.I0_tol))

    if roots.is_polysolve_required(lp,I0,num_pred_roots):
        new_state = roots.from_polysolve(lp,b,num_pred_roots,I0)
        status |= gp.handle_state(lp,new_state)

    do_subcalc = do_subcalculation(lp,roots,I0,num_pred_roots)
    roots.finialise_region_roots(lp,b,I0,do_subcalc)
    if do_subcalc:
        new_state,num_regions = calculate_for_subregions(lp,b,roots)
        status |= new_state
        roots.log_sub_region(lp,num_regions)
    new_state = gp.update_state_for_subcalc(lp,do_subcalc,roots,num_pred_roots,
                                            I0)
    status |= gp.handle_state(lp,new_state)
    
    roots.finialise_end_roots(lp,status)

    return status,roots.end_unique
