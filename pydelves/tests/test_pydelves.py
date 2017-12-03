from examples import *
from functions import *

import unittest

class pydelves_testcase(unittest.TestCase):
    def setUp(self):
        set_default_parameters()

class trig_testcase(pydelves_testcase):
    def runTest(self):
        status,retroots = trig()
        self.assertFalse(was_warning(status))
        roots = np.asarray(retroots)
        self.assertTrue(two_sets_almost_equal(roots/np.pi,
                        [-5.,-4.,-3.,-2.,-1.,-0.,1.,2.,3.,4.,5.]))

# We require that all tests pass for the default parameters.
# Of course, different parameters may result in not all roots being found.
class poly_testcase(pydelves_testcase):
    def runTest(self):
        for polyOrder in range(2,41):
            print "Testing polyOrder " + str(polyOrder)
            status,roots_delves,roots_numpy=\
                poly_roots(polyOrder)
            self.assertFalse(was_warning(status),"Bad Status")
            self.assertEqual(len(roots_delves),len(roots_numpy),"Bad Root Nums")
            for root_numpy in roots_numpy:
                found_equal = False
                for root_delves in roots_delves:
                    if almost_equal(root_numpy,root_delves,eps=1e-5):
                        found_equal = True
                        break
                self.assertTrue(found_equal,"Bad Roots")

class wilkinson_testcase(pydelves_testcase):
    def runTest(self):
        status,retroots = wilkinson(N=100) # N=100 or exception
        # self.assertFalse(was_warning(status)) # See issue #15
        retroots.sort(key=abs)
        self.assertEqual(len(retroots),20,"Bad Root Nums")
        for i in range(1,21):
            equal = almost_equal(abs(retroots[i-1]),float(i),eps=1e-5)
            self.assertTrue(equal,"Bad Roots")

# We want to test that warnings are generated when all the roots are not found.
# Run with non-optimal parameters.
class poly_bad_parameters_testcase(pydelves_testcase):
    def runTest(self):

        # These modes cause not all roots to be found for some Ns:
        mode = mode_dont_recurse_on_inaccurate_roche
        mode |= mode_dont_recurse_on_not_all_interior_found

        num_bad_status = 0
        for polyOrder in range(2,41):
            print "Testing polyOrder " + str(polyOrder)
            status,roots_delves,roots_numpy=\
                poly_roots(polyOrder,mode=mode)

            good_status = True
            if len(roots_delves) == len(roots_numpy):
                for root_numpy in roots_numpy:
                    found_equal = False
                    for root_delves in roots_delves:
                        if almost_equal(root_numpy,root_delves,eps=1e-5):
                            found_equal = True
                            break
                    if not found_equal:
                        good_status = False
                        break
            else:
                good_status = False

            if not good_status:
                num_bad_status += 1
                self.assertTrue(was_warning(status),"No Bad Status")

        # For the test to be valid require enough bad status over the poly
        # range. If this fails then relax the parameters.
        self.assertTrue(num_bad_status>3,"Not enough bad status.")