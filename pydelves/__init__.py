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
import cmath
from functions import *

#This is the default. Optimal for simple functions.
mode_off = 0

################################################################################
########################### Recursion Modes ####################################
################################################################################
#Recursion is expensive. If the routine is selected to not recurse on either of
#the three conditions below then a warning will be returned from the routine, as
#can't guarantee that all of the roots have been found.
mode_dont_recurse_on_bad_roche = 0x1
mode_dont_recurse_on_inaccurate_roche = 0x2
mode_dont_recurse_on_not_all_interior_found = 0x4

################################################################################
######################## Root identifying Modes ################################
################################################################################
#Since muller uses f(x)<tol then if differential is very high small displacement
#from root can give much larger f(x). Since we know the rough value of the roche
#root we can compare to that instead (or as well). Two options here. If the 
#good_roche variant is selected then will only accept if the routine had a good
#I0.
mode_accept_int_muller_close_to_good_roche = 0x100
mode_accept_int_muller_close_to_any_roche = 0x200
#Or the following option to accept all mullers, without any check. Good when
#roots can be missing and and some external check is applied.
mode_accept_all_mullers = 0x400

#When recursing the routine doesn't bother trying to find the roche roots if
#the IO is outside I0_tol from integer; it will be trying in the subregion.
#However, if the roche never becomes good at any level of recursion an attempt
#will never be made. This mode tells the routine to both attempt to find the
#roche roots and to recurse with the I0 is bad.
mode_attempt_polysolve_on_bad_roche = 0x800

mode_always_attempt_polysolve_on_final_step = 0x1000

################################################################################
############################# Boundary Modes ###################################
################################################################################
#Poles close enough to boundary will deteriorate the values obtained from the
#roche integrals. These modes will change the boundary until a good roche is
#obtained.
mode_boundary_change_off = 0x10000

#Use this mode to attempt to remove poles close to contour by searching for 
#turning points along the contour. These are then used as starting points for 
#mullers
mode_boundary_search_on = 0x20000

#Use this mode to turn off boundary smoothing and root subtraction
mode_boundary_smoothing_off = 0x40000

#Due to tendency to wander it's harder to verify the boundary roots. The 
#following mode extends the supplied region by twice bnd_thres and carries out 
#the boundary root calculation and roche integration at the midline between 
#the supplied and the extended regions. Only boundary roots between the 
#supplied region and the extended region are used. Only roche roots inside the
#supplied region are returned to the user.
mode_use_stripped_subtraction = 0x80000

#Similar to mode_accept_int_muller_close_to_good_roche. Note that boundary roots
#have a greater tendency to wander from their starting than the roche roots. 
#Recommended to keep the following mode off and use 
#mode_use_stripped_subtraction instead.
mode_accept_bnd_muller_close_to_start = 0x100000

#When searching for roots around the boundary default is to check for turning
#points in the absolute values. This can sometimes identify spurious outliers.
#Following mode checks the components separately. This mode has been observed
#to add ~50% to the run time, so is off by default. Spurious outliers that are 
#identified when not in this mode will usually be removed with the purge.
mode_strict_boundary_search = 0x200000

################################################################################
############################## Other Modes #####################################
################################################################################
#If function is a polynomial with real coefficients then roots will occur in 
#a+ib, a-ib pairs. If mode==mode_add_conjs then routine will take advantage of 
#this by automatically adding missing partners before recursing.
mode_add_conjs = 0x1000000



################################################################################
################################ Logging #######################################
################################################################################
log_off = 0
log_recursive = 0x1
log_summary = 0x2
log_verbose = 0x4
log_debug = 0x8


################################################################################
################################ Status ########################################
################################################################################
ok = 0

warn_root_check_disabled = 0x1
warn_inaccurate_roche = 0x2
warn_bad_roche = 0x4
warn_could_not_locate_roche_root = 0x8

note_inaccurate_roche = 0x10
note_bad_roche = 0x20
note_could_not_locate_roche_root = 0x40
note_muller_fail_1st = 0x80
note_muller_fail_2nd = 0x100
note_muller_exception = 0x200
note_root_sub_div_by_zero = 0x400
note_boundary_stuck = 0x800
note_boundary_cnt_exceeded = 0x1000

#Used for switching out notes from warnings:
mode_warn_switch = 0xF


################################################################################
############################### Defaults #######################################
################################################################################

default_N = 500 
default_max_steps = 5
default_mode = mode_off 
default_log = log_off
default_outlier_coeff = 100.
default_max_order = 10
default_I0_tol = 5e-3
default_fun_multiplier = 1.
default_mul_N = 400
default_mul_fzltol = 1e-12
default_mul_fzhtol = 1e-12
default_mul_off = 1e-5
default_mul_ztol = 1e-4
default_bnd_change = 0.01
default_bnd_limit = None
default_bnd_max_tries = 5
default_conj_min_i = 1e-8
default_dist_eps = 1e-7
default_lmt_N = 10
default_lmt_eps = 1e-3
default_bnd_thres = 2.


################################################################################
############################ Test Variables ####################################
################################################################################

test_region_changes = 0
def reset_delves_test():
    global test_region_changes
    test_region_changes = 0
def get_test_region_changes():
    global test_region_changes
    return test_region_changes

################################################################################
################################ Routine #######################################
################################################################################

def _root_purge(lst,eps=1e-7,conj_min_i=1e-8):
    if len(lst) == 0:
        return []
    for el in lst[:-1]:
        if abs(el-lst[-1]) < eps and \
        (el.imag/lst[-1].imag>=0 or abs(el.imag)<conj_min_i):
            return _root_purge(lst[:-1],eps,conj_min_i)
    return _root_purge(lst[:-1],eps,conj_min_i) + [lst[-1]]

def _add_miss_conjs(lst,eps=1e-7,conj_min_i=1e-8):
    new_lst = []
    for el in lst:
        new_lst.append(el)
        new_lst.append(el.conjugate())
    return _root_purge(new_lst,eps,conj_min_i)

def _locate_muller_root(lp,x1,x2,x3,found_roots,failed_roots):
    status = ok
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

def _check_against_start_roots(failed_roots,reaccepted_roots):
    status = ok
    for rough_root, mull_root in failed_roots:
        if abs(rough_root-mull_root) < gp.mul_ztol:
            reaccepted_roots.append(mull_root)
        else:
            status |= note_muller_fail_2nd
    return status

def _check_against_rough_roots(lp,I0,failed_roots,reaccepted_roots):
    check = False
    if lp.mode & mode_accept_int_muller_close_to_any_roche:
        check = True
    elif lp.mode & mode_accept_int_muller_close_to_good_roche:
        if gp.is_roche_accurate(I0):
            check = True
    if check:
        return _check_against_start_roots(failed_roots,reaccepted_roots)
    return 0

def _check_against_bnd_start_roots(lp,failed_roots,reaccepted_roots):
    if lp.mode & mode_accept_bnd_muller_close_to_start:
        return _check_against_start_roots(failed_roots,reaccepted_roots)
    return 0

def _correct_roots(lp,b,roots):
    if lp.lvl_cnt == 0:
        roots_mod = inside_boundary(roots,*b.reg_start())
    else:
        roots_mod = roots
    conjs_added = 0
    if lp.mode & mode_add_conjs:
        roots_missing_conjs = _add_miss_conjs(roots_mod,gp.dist_eps,gp.conj_min_i)
        if lp.lvl_cnt == 0:
            roots_final = inside_boundary(roots_missing_conjs,*b.reg_start())
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
        self.boundary_and_known = self.known
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
        self.end_within = []

    def log_region(self,lp,b,I0,isSubregions):
        if lp.log & log_summary:
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
        print s+"  " + str(num_interior_roots_fnd) + " new from Poly Muller."+a
        if not isSubregions:
            print s+"  There were no subregions."
        elif lp.log & log_recursive:
            print s+"  Subregions:"

    def log_sub_region(self,lp,num_regions):
        if lp.log & log_summary:
            self._log_sub_region(lp,num_regions)

    def _log_sub_region(self,lp,num_regions):
        s = "."*lp.lvl_cnt
        num_sub_roots_fnd = len(self.interior_all_subs)
        if num_regions>0 and lp.log & log_recursive:
            print ""
        print(s+"  "+str(num_sub_roots_fnd)+" from "+str(num_regions)+\
              " subregions.")

    def log_totals(self,lp):
        if lp.log & log_summary:
            self._log_totals(lp)

    def _log_totals(self,lp):
        s = "."*lp.lvl_cnt
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
        if lp.log & log_summary:
            self._log_notes(lp,status)

    def _log_notes(self,lp,status):
        s = "."*lp.lvl_cnt
        print_status(status, s)

    def log_roots(self,lp):
        if lp.log & log_debug:
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
        print s+"Within:\n"+str(np.array(self.end_within))

    def log_close_region(self,lp):
        if lp.log & log_summary:
            print "-"*lp.lvl_cnt

    def at_boundary(self,lp,b):
        if lp.mode & mode_strict_boundary_search:
            outlier_indices = find_maxes_complex(b.y)
        else:
            outlier_indices = find_maxes(map(abs,b.y))
        status = ok
        self.boundary_outliers = []
        for i in outlier_indices:
            self.boundary_outliers.append(b.c[i])
            status |= _locate_muller_root(lp,b.c[i-1],b.c[i+1],
                                        b.c[i],self.boundary_passed_fz,
                                        self.boundary_failed_fz)
        status |= _check_against_bnd_start_roots(lp,self.boundary_failed_fz,
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

    def is_polysolve_required(self,lp,num_pred_roots,I0):
        if lp.max_steps==0 and\
           lp.mode & mode_always_attempt_polysolve_on_final_step:
            return True
        roche_accurate = gp.is_roche_accurate(I0)
        if (cmath.isinf(I0) or cmath.isnan(I0)) and\
           not lp.mode & mode_dont_recurse_on_bad_roche:
            return False
        if not roche_accurate and\
           not lp.mode & mode_attempt_polysolve_on_bad_roche and\
           not lp.mode & mode_dont_recurse_on_inaccurate_roche:
            return False
        return num_pred_roots <= gp.max_order and num_pred_roots >= 1

    def from_polysolve(self,lp,b,num_pred_roots,I0):
        if not lp.mode & mode_boundary_smoothing_off:
            self.interior_rough = locate_poly_roots(b.y_smooth,b.c,
                                                    num_pred_roots)
        else:
            self.interior_rough = locate_poly_roots(b.y,b.c,num_pred_roots)

        status = ok
        ##TODO: best way to pick points for Muller method below
        for rough_root in self.interior_rough:
            status |= _locate_muller_root(lp,rough_root-gp.mul_off,
                                        rough_root+gp.mul_off,rough_root,
                                        self.interior_passed_fz,
                                        self.interior_failed_fz)
        status |= _check_against_rough_roots(lp,I0,self.interior_failed_fz,
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
            #outside the specified region will be removed in _correct_roots.
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
        self.region_corrected,self.num_added_conjs = _correct_roots(lp,b,
                                                                   self.region)
        self.log_region(lp,b,I0,sub_required)

    def finialise_end_roots(self,lp,b,status):
        # Only return new roots. They'll be added to known roots in the parent.
        self.end = self.region_corrected+self.interior_all_subs
        self.end_purged = purge(self.end,gp.dist_eps)
        self.end_unique = get_unique(self.end_purged,self.known,gp.dist_eps)
        self.end_within = inside_boundary(self.end_unique, *b.reg_i())
        self.log_totals(lp)
        self.log_notes(lp,status)
        self.log_roots(lp)
        self.log_close_region(lp)

class boundary:
    def __init__(self,rx,ry,rw,rh):
        self.rx_start = rx
        self.ry_start = ry
        self.rw_start = rw
        self.rh_start = rh
        self.cnt = 0

    def set(self,lp,I0,calculate=True):
        self.stuck = False
        rx,ry,rw,rh = self._calculate_mod_region_parameters(lp,I0)

        self.rx = rx
        self.ry = ry

        self.rw_i = rw
        self.rh_i = rh

        self.rw_m = rw+gp.bnd_thres
        self.rh_m = rh+gp.bnd_thres

        self.rw_o = rw+2.*gp.bnd_thres
        self.rh_o = rh+2.*gp.bnd_thres

        if calculate:
            self.c = get_boundary(gp.N,*self.reg_calcs(lp))
            self.f_frac = lambda z: gp.fp(z)/(2j*np.pi*gp.f(z))
            self.y = [self.f_frac(z) for z in self.c]
            self.max_ok = abs(gp.outlier_coeff*get_max(self.y))

        self.cnt+=1

    def _get_side_position_dec(self,start,last,chg,lmt):
        if chg is not None:
            new = start - chg*self.cnt
            if lmt is None or new > lmt:
                return new,new!=last
            else:
                return lmt,lmt!=last
        return start,False

    def _get_side_position_inc(self,start,last,chg,lmt):
        if chg is not None:
            new = start + chg*self.cnt
            if lmt is None or new < lmt:
                return new,new!=last
            else:
                return lmt,lmt!=last
        return start,False

    def _calculate_mod_region_parameters(self,lp,I0):
        if self.cnt != 0:
            left_start = self.rx_start - self.rw_start
            right_start = self.rx_start + self.rw_start
            bot_start = self.ry_start - self.rh_start
            top_start = self.ry_start + self.rh_start
            
            left_last = self.rx - self.rw_i
            right_last = self.rx + self.rw_i
            bot_last = self.ry - self.rh_i
            top_last = self.ry + self.rh_i
            
            s = "."*lp.lvl_cnt
            if lp.log & log_summary:
                print s+"Attempting to change region, Bad Roche: "+str(abs(I0))

            left,a = self._get_side_position_dec(left_start,left_last,
                                                 gp.change_left,
                                                 gp.limit_left)
            right,b = self._get_side_position_inc(right_start,right_last,
                                                  gp.change_right,
                                                  gp.limit_right)
            bot,c = self._get_side_position_dec(bot_start,bot_last,
                                                gp.change_bottom,
                                                gp.limit_bottom)
            top,d = self._get_side_position_inc(top_start,top_last,
                                                gp.change_top,
                                                gp.limit_top)

            rx = (left + right) / 2.
            ry = (top + bot) / 2.
            rw = abs((left - right) / 2.)
            rh = abs((top - bot) / 2.)

            if a or b or c or d:
                global test_region_changes
                test_region_changes += 1
                if lp.log & log_summary:
                    print (s+"  Region changed(rx,ry,rw,rh): "+str(rx)+" "+\
                           str(ry)+" "+str(rw)+" "+str(rh))
            else:
                if lp.log & log_summary:
                    print s+"  Region stuck."
                self.stuck = True
        else:
            rx = self.rx_start
            ry = self.ry_start
            rw = self.rw_start
            rh = self.rh_start

        return rx,ry,rw,rh

    def is_stuck(self):
        return self.stuck

    def exceeded_tries(self):
        return self.cnt>gp.max_tries

    def get_corner_indices(self):
        return [0, gp.N, 2*gp.N, 3*gp.N]

    def reg_start(self):
        return self.rx_start, self.ry_start, self.rw_start, self.rh_start

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
        status = ok
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
        if lp.log & log_summary:
            s = "-"*lp.lvl_cnt
            print ("\n"+s+"Region(rx,ry,rw,rh): "+\
                   str(self.rx)+" "+str(self.ry)+\
                   " "+str(self.rw_i)+" "+str(self.rh_i))

def _boundary_status(b,I0):
    if not gp.is_roche_accurate(I0):
        if b.is_stuck():
            return note_boundary_stuck
        if b.exceeded_tries():
            return note_boundary_cnt_exceeded
    return ok

def _all_interior_found(roots,num_pred_roots):
    return len(roots.interior_new)>=num_pred_roots

def _do_subcalculation(lp,roots,I0,num_pred_roots):
    ret = False
    if (cmath.isinf(I0) or cmath.isnan(I0)) and\
           not lp.mode & mode_dont_recurse_on_bad_roche:
        ret = True
    else:
        # Don't count the added conjs at this stage, just pass them to the 
        # subregions. Otherwise will confuse the routine.
        all_int_fnd = _all_interior_found(roots,num_pred_roots)
        if num_pred_roots > gp.max_order:
            ret = True
        elif not gp.is_roche_accurate(I0) and\
           not lp.mode & mode_dont_recurse_on_inaccurate_roche:
            ret = True
        elif not all_int_fnd and\
           not lp.mode & mode_dont_recurse_on_not_all_interior_found:
            ret = True
    return ret and lp.max_steps!=0

def _change_boundary(lp,b,I0):
    if I0 is None:
        return True
    return not lp.mode & mode_boundary_change_off and not b.is_stuck() and\
           not gp.is_roche_accurate(I0) and not b.exceeded_tries()

def _calculate_for_subregions(lp,b,roots):
    if lp.log & log_recursive:
        new_log = lp.log
    else:
        new_log = log_off
    status = ok
    known_roots = roots.region + roots.known
    num_regions = 0
    for x,y in zip(*b.get_subregions()):
        new_state,sub_roots = droots(gp.f,gp.fp,x,y,b.rw_i/2.,b.rh_i/2.,gp.N,
                               lp.max_steps-1,lp.mode,new_log,known_roots,
                               lp.lvl_cnt+1)
        status |= new_state
        roots.interior_all_subs.extend(sub_roots)
        num_regions += 1
    if lp.lvl_cnt == 0:
        #Need this here since subregion roots added after _correct_roots.
        roots.interior_all_subs = inside_boundary(roots.interior_all_subs,
                                                  *b.reg_start())
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
        self.fun_multiplier = default_fun_multiplier

        self.mul_N = default_mul_N
        self.mul_fzltol = default_mul_fzltol
        self.mul_fzhtol = default_mul_fzhtol
        self.mul_off = default_mul_off

        self.change_left = default_bnd_change
        self.change_right = default_bnd_change
        self.change_bottom = default_bnd_change
        self.change_top = default_bnd_change
        self.limit_left = default_bnd_limit
        self.limit_right = default_bnd_limit
        self.limit_bottom = default_bnd_limit
        self.limit_top = default_bnd_limit
        self.max_tries = default_bnd_max_tries

        self.conj_min_i = default_conj_min_i
        self.mul_ztol = default_mul_ztol

        self.dist_eps = default_dist_eps
        self.lmt_N = default_lmt_N
        self.lmt_eps = default_lmt_eps
        self.bnd_thres = default_bnd_thres

        self.left = None

    def set_delves_routine_parameters(self,outlier_coeff,max_order,I0_tol):
        self.outlier_coeff = outlier_coeff
        self.max_order = max_order
        self.I0_tol = I0_tol

    def set_muller_parameters(self,mul_N,mul_fzltol,mul_fzhtol,mul_off):
        self.mul_N = mul_N
        self.mul_fzltol = mul_fzltol
        self.mul_fzhtol = mul_fzhtol
        self.mul_off = mul_off

    def set_changing_region_parameters(self,change_left,change_right, 
                                       change_bottom,change_top,
                                       limit_left,limit_right,
                                       limit_bottom, limit_top,
                                       max_tries):
        self.change_left = change_left
        self.change_right = change_right
        self.change_bottom = change_bottom
        self.change_top = change_top
        self.limit_left = limit_left
        self.limit_right = limit_right
        self.limit_bottom = limit_bottom
        self.limit_top = limit_top
        self.max_tries = max_tries

    def set_mode_parameters(self,mul_ztol,conj_min_i):
        self.conj_min_i = conj_min_i
        self.mul_ztol = mul_ztol

    def set_advanced_parameters(self,dist_eps,lmt_N,lmt_eps,bnd_thres,
                                fun_multiplier):
        self.dist_eps = dist_eps
        self.lmt_N = lmt_N
        self.lmt_eps = lmt_eps
        self.bnd_thres = bnd_thres
        self.fun_multiplier = fun_multiplier

    def set_calc_parameters(self,f,fp,N):
        if self.fun_multiplier == 1.:
            self.f = f
            self.fp = fp
        else:
            self.f = lambda x: self.fun_multiplier*f(x)
            self.fp = lambda x: self.fun_multiplier*fp(x)
        self.N = N

    def set_limits(self,left):
        self.left = left

    def handle_state(self,lp,status):
        if status!=ok and lp.log & log_verbose:
            s = "."*lp.lvl_cnt
            print s+"State appended: " +str(status)
        return status

    def is_roche_accurate(self,I0):
        return abs(abs(I0)-round(abs(I0)))<self.I0_tol

    def update_state_for_subcalc(self,lp,do_subcalc,roots,num_pred_roots,I0):
        status = ok
        if cmath.isinf(I0) or cmath.isnan(I0):
            if do_subcalc:
                status = gp.handle_state(lp,note_bad_roche)
            else:
                status = gp.handle_state(lp,warn_bad_roche)
        elif not gp.is_roche_accurate(I0):
            if do_subcalc:
                status = gp.handle_state(lp,note_inaccurate_roche)
            else:
                status = gp.handle_state(lp,warn_inaccurate_roche)
        elif not _all_interior_found(roots,num_pred_roots):
            if do_subcalc:
                status = gp.handle_state(lp,note_could_not_locate_roche_root)
            else:
                status = gp.handle_state(lp,warn_could_not_locate_roche_root)
        return status

class local_parameters:
    def __init__(self,max_steps,mode,log,lvl_cnt):
        self.max_steps = max_steps
        self.mode = mode
        self.log = log
        self.lvl_cnt = lvl_cnt
        
    def init_state(self):
        status = ok
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

def set_changing_region_parameters(change_left=default_bnd_change,
                                   change_right=default_bnd_change,
                                   change_bottom=default_bnd_change,
                                   change_top=default_bnd_change,
                                   limit_left=default_bnd_limit,
                                   limit_right=default_bnd_limit,
                                   limit_bottom=default_bnd_limit,
                                   limit_top=default_bnd_limit,
                                   max_tries=default_bnd_max_tries):
    '''
    These parameters are only relevant if the related mode is set.

    Args:
        change_left (optional[float]): If mode is not
            mode_boundary_change_off then this value will be applied when 
            changing region due to bad roche. Applies to left of region.

        change_right (optional[float]): See change_left. Applies to
            right of region.

        change_bottom (optional[float]): See change_left. Applies to
            bottom of region.

        change_top (optional[float]): See change_left. Applies to
            top of region.

        limit_left (optional[float]): If mode is not mode_boundary_limit_off
            then the changing region will be limited by this value. Applies to
            left of region.

        limit_right (optional[float]): See limit_left. Applies to right
            of region.

        limit_bottom (optional[float]): See limit_left. Applies to 
            bottom of region.

        limit_top (optional[float]): See limit_left. Applies to top of 
            region.

        max_tries (optional[int]): If mode is not
            mode_boundary_change_off then this value will be the maximum number
            of attempts at finding a good roche for different regions sizes.
    '''
    gp.set_changing_region_parameters(change_left,change_right,
                                      change_bottom,change_top,
                                      limit_left,limit_right,
                                      limit_bottom,limit_top,max_tries)

def set_mode_parameters(mul_ztol=default_mul_ztol,
                        conj_min_i=default_conj_min_i):
    '''
    These parameters are only relevant if the related mode is set.

    Args:
        mul_ztol (optional[float]): If mode is: 
            mode_accept_bnd_muller_close_to_start
            mode_accept_int_muller_close_to_good_roche
            mode_accept_int_muller_close_to_any_roche 
            then this parameter determines the distance from the rough that the 
            returned (failed muller) can be within to be re-accepted as a good 
            root.

        conj_min_i (optional[float]): If mode is mode_add_conjs:
            then this parameter determines the minimum distance from the real 
            axis a root must lie before being considered as having a conjugate 
            partner.
    '''
    gp.set_mode_parameters(mul_ztol, conj_min_i)
    
def set_advanced_parameters(dist_eps=default_dist_eps,lmt_N=default_lmt_N,
                            lmt_eps=default_lmt_eps,bnd_thres=default_bnd_thres,
                            fun_multiplier=default_fun_multiplier):
    '''
    Set advanced arguments

    Args:
        dist_eps (optional[float]): epsilon used when distinguishing roots
            using absolute values. Within this they are judged the same root.

        lmt_N (int): number of points used in the estimate when calculating the
            residues.

        lmt_eps (optional[float]): distance from z0 at which estimating points 
            are placed for the subtraction.

        bnd_thres (optional[float]): The perpendicular distance outwards from 
            the region boundary within which roots must lie to be considered
            boundary.
            
        fun_multiplier (optional[float]): Value to multiple the function and 
            derivative by. Useful if values very low.
    '''
    gp.set_advanced_parameters(dist_eps,lmt_N,lmt_eps,bnd_thres,fun_multiplier)

def set_default_parameters():
    gp.set_defaults()

def was_warning(status):
    return bool(status&mode_warn_switch)
    
def print_status(status, s=""):
    if status & warn_bad_roche:
        print s+"WARNING: bad final roche."
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
           mode=default_mode,log=default_log,known_roots=[],lvl_cnt=0):
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

        log (optional[int]): This sets the log mode. See top of file for
            available modes.

        known_roots (internal[list of complex numbers]): Roots of f that are
            already known. Used when recursing

        lvl_cnt (internal[int]): number of times the routine has recursed.

    Returns:h
        A boolean indicating if all roots found for the supplied max_steps and 
            I0_tol parameters. A list of roots for the function f inside the 
            rectangle determined by the values rx,ry,rw and rh.
    '''
    lp = local_parameters(max_steps,mode,log,lvl_cnt)
    gp.set_calc_parameters(f,fp,N)
    if lvl_cnt == 0:
        gp.set_limits(rx-rw)
    roots = root_container(known_roots)
    status = lp.init_state()
    num_regions = 0

    I0 = None
    b = boundary(rx,ry,rw,rh)
    while _change_boundary(lp,b,I0):
        b.set(lp,I0)

        if lp.mode & mode_boundary_search_on:
            status |= gp.handle_state(lp,roots.at_boundary(lp,b))

        if not lp.mode & mode_boundary_smoothing_off:
            status |= gp.handle_state(lp,b.smoothed(roots))
            I0 = integrate.trapz(b.y_smooth,b.c)
        else:
            I0 = integrate.trapz(b.y,b.c)
    status |= gp.handle_state(lp,_boundary_status(b,I0))

    num_pred_roots = 0
    if not cmath.isinf(I0) and not cmath.isnan(I0):
        num_pred_roots = int(math.ceil(abs(I0)-gp.I0_tol))

    if roots.is_polysolve_required(lp,num_pred_roots,I0):
        new_state = roots.from_polysolve(lp,b,num_pred_roots,I0)
        status |= gp.handle_state(lp,new_state)

    do_subcalc = _do_subcalculation(lp,roots,I0,num_pred_roots)
    roots.finialise_region_roots(lp,b,I0,do_subcalc)
    if do_subcalc:
        new_state,num_regions = _calculate_for_subregions(lp,b,roots)
        status |= new_state
        roots.log_sub_region(lp,num_regions)

    new_state = gp.update_state_for_subcalc(lp,do_subcalc,roots,num_pred_roots,
                                            I0)
    status |= gp.handle_state(lp,new_state)

    roots.finialise_end_roots(lp,b,status)

    return status,roots.end_unique
