from examples import *
from functions import *

import unittest

class pydelves_testcase(unittest.TestCase):
    def setUp(self):
        set_default_parameters()

class boundary_root_changing_region_testcase(pydelves_testcase):
    def runTest(self):
        mode = mode_boundary_smoothing_off
        set_advanced_parameters(dist_eps=10-5)
        reset_delves_test()
        status,roots=boundary_root(mode=mode)
        #There are actually two roots, both at x=0. Since the root is identified
        #in the boundary search it is subtracted and we end up with a zero 
        #roche.
        self.assertTrue(len(roots)==1,"Too many roots found.")
        #Since the routine doesn't deal with root multiplicities it gets I0=2.
        #As only one distinct root is returned it thinks it has missed one.
        self.assertTrue(status&warn_could_not_locate_roche_root,
                        "Expected bad status")
        self.assertTrue(get_test_region_changes()>0)
        self.assertTrue(almost_equal(roots[0],0.,1e-06),"Incorrect root.")


class boundary_root_changing_region_off_testcase(pydelves_testcase):
    def runTest(self):
        mode = mode_boundary_smoothing_off | mode_boundary_change_off
        set_advanced_parameters(dist_eps=10-5)
        status,roots=boundary_root(mode=mode)
        #There are actually two roots, both at x=0. Since the root is identified
        #in the boundary search it is subtracted and we end up with a zero 
        #roche.
        self.assertTrue(len(roots)==0,"Too many roots found.")
        #Since the routine doesn't deal with root multiplicities it gets I0=2.
        #As only one distinct root is returned it thinks it has missed one.
        self.assertTrue(status&warn_bad_roche, "Expected bad status.")

if __name__ == "__main__":
    #Just for debug
    b = boundary_root_changing_region_testcase()
    b.runTest()
    