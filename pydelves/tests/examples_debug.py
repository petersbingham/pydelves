from examples import *
from functions import *

def run_poly(polyOrder,printRoots,printPolys):
    N = default_N
    max_steps = default_max_steps

    outlier_coeff = default_outlier_coeff
    max_order = default_max_order
    I0_tol = default_I0_tol

    mul_N = default_mul_N
    mul_fzltol = default_mul_fzltol
    mul_fzhtol = default_mul_fzhtol
    mul_off = default_mul_off

    mul_ztol = default_mul_ztol
    conj_min_i = default_conj_min_i

    dist_eps = default_dist_eps
    lmt_N = default_lmt_N
    lmt_eps = default_lmt_eps
    bnd_thres = default_bnd_thres

    logmode = mode_off    
    #logmode |= mode_log_summary
    #logmode |= mode_log_recursive
    #logmode |= mode_log_debug
    #logmode |= mode_log_verbose

    calcmode = mode_off
    #calcmode |= mode_dont_recurse_on_inaccurate_roche
    #calcmode |= mode_dont_recurse_on_not_all_interior_found
    #calcmode |= mode_accept_int_muller_close_to_good_roche
    #calcmode |= mode_use_stripped_subtraction
    mode = logmode | calcmode

    set_delves_routine_parameters(outlier_coeff,max_order,I0_tol)
    set_muller_parameters(mul_N,mul_fzltol,mul_fzhtol,mul_off)
    set_mode_parameters(mul_ztol,conj_min_i)
    set_advanced_parameters(dist_eps,lmt_N,lmt_eps,bnd_thres)

    print "\npolyOrder = " + str(polyOrder)

    status,roots_delves,roots_numpy=\
        poly_roots(polyOrder,N,max_steps,mode,printPolys)
    print_warnings(status)

    print "Comparison with numpy:"
    print "\t" + str(len(roots_numpy)) + " numpy roots"
    print "\t" + str(len(roots_delves)) + " delves roots"
    common = 0
    for root_numpy in roots_numpy:
        for root_delves in roots_delves:
            if almost_equal(root_numpy, root_delves,eps=1e-5):
                common += 1
                break
    print "\t" + str(common) + " common roots_delves"

    if printRoots:
        for root in sorted(roots_numpy):
          print str(root) + "  \t" + str(get_poly_fun(polyOrder)(root))
        print
        for root in sorted(roots_delves):
          print str(root) + "  \t" + str(get_poly_fun(polyOrder)(root))

def run_poly_range(printRoots,printPolys):
    for polyOrder in range(2,41):
        run_poly(polyOrder,printRoots,printPolys)

def run_wilkinson(printRoots):
    N = 100
    max_steps = default_max_steps

    outlier_coeff = default_outlier_coeff
    max_order = default_max_order
    I0_tol = default_I0_tol

    mul_N = default_mul_N
    mul_fzltol = default_mul_fzltol
    mul_fzhtol = default_mul_fzhtol
    mul_off = default_mul_off

    mul_ztol = default_mul_ztol
    conj_min_i = default_conj_min_i

    dist_eps = default_dist_eps
    lmt_N = default_lmt_N
    lmt_eps = default_lmt_eps
    bnd_thres = default_bnd_thres

    logmode = mode_off    
    #logmode |= mode_log_summary
    #logmode |= mode_log_recursive
    #logmode |= mode_log_debug
    #logmode |= mode_log_verbose

    calcmode = mode_off
    #calcmode |= mode_dont_recurse_on_inaccurate_roche
    #calcmode |= mode_dont_recurse_on_not_all_interior_found
    #calcmode |= mode_accept_int_muller_close_to_good_roche
    #calcmode |= mode_use_stripped_subtraction
    mode = logmode | calcmode

    set_delves_routine_parameters(outlier_coeff,max_order,I0_tol)
    set_muller_parameters(mul_N,mul_fzltol,mul_fzhtol,mul_off)
    set_mode_parameters(mul_ztol,conj_min_i)
    set_advanced_parameters(dist_eps,lmt_N,lmt_eps,bnd_thres)

    status,roots=wilkinson(N,max_steps,mode)
    print_warnings(status)

    if printRoots:
        for root in sorted(roots):
          print str(root) + "  \t" + str(wilk_f(root))

if __name__ == "__main__":
    run_poly_range(printRoots=False, printPolys=False)
    #run_poly(14, printRoots=False, printPolys=False)
    #run_wilkinson(printRoots=True)
