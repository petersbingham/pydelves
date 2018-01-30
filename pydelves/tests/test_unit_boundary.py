from examples import *

import unittest

class boundary_no_limits(unittest.TestCase):
    def runTest(self):
        lp = local_parameters(0,0,0,0)
        set_changing_region_parameters(1.,1.,1.,1.)
        for rx in [2.,-2.]:
            for ry in [2.,-2.]:
                b = boundary(rx,ry,1.,1.)
                for cnt in range(0,4):
                    b.set(lp,0.,False)
                    self.assertEqual(b.rx,rx)
                    self.assertEqual(b.ry,ry)
                    self.assertEqual(b.rw_i,1.+cnt)
                    self.assertEqual(b.rh_i,1.+cnt)

class boundary_limits(unittest.TestCase):
    def _str(self,rx,ry,lmt,cnt):
        return ",".join(map(lambda x:str(x),[rx,ry,lmt,cnt]))

    def runTest(self):
        lp = local_parameters(0,0,0,0)
        for rx in [2.,-2.]:
            for ry in [2.,-2.]:
                for lmt in range(3):
                    set_changing_region_parameters(1.,1.,1.,1.,
                        rx-1.-lmt,rx+1.+lmt,ry-1.-lmt,ry+1.+lmt)
                    b = boundary(rx,ry,1.,1.)
                    for cnt in range(4):
                        b.set(lp,0.,False)
                        self.assertEqual(b.rx,rx,self._str(rx,ry,lmt,cnt))
                        self.assertEqual(b.ry,ry,self._str(rx,ry,lmt,cnt))
                        if cnt>lmt:          
                            self.assertEqual(b.rw_i,1.+lmt,self._str(rx,ry,lmt,cnt))
                            self.assertEqual(b.rh_i,1.+lmt,self._str(rx,ry,lmt,cnt))
                            self.assertTrue(b.is_stuck(),self._str(rx,ry,lmt,cnt))
                        else:
                            self.assertEqual(b.rw_i,1.+cnt,self._str(rx,ry,lmt,cnt))
                            self.assertEqual(b.rh_i,1.+cnt,self._str(rx,ry,lmt,cnt))
                            self.assertFalse(b.is_stuck(),self._str(rx,ry,lmt,cnt))

class boundary_limits_center_change(unittest.TestCase):
    def runTest(self):
        lp = local_parameters(0,0,0,0)
        for rx in [2.,-2.]:
            for ry in [2.,-2.]:
                for i in range(4):
                    b = boundary(rx,ry,1.,1.)
                    if i==0:
                        set_changing_region_parameters(1.,1.,1.,1.,
                            rx-1.,None,None,None)
                    elif i==1:
                        set_changing_region_parameters(1.,1.,1.,1.,
                            None,rx+1.,None,None)
                    elif i==2:
                        set_changing_region_parameters(1.,1.,1.,1.,
                            None,None,ry-1.,None)
                    elif i==3:
                        set_changing_region_parameters(1.,1.,1.,1.,
                            None,None,None,ry+1.)
                    for cnt in range(4):
                        b.set(lp,0.,False)
                        if i==0:
                            self.assertEqual(b.rx,rx+0.5*cnt)
                        elif i==1:
                            self.assertEqual(b.rx,rx-0.5*cnt)
                        else:
                            self.assertEqual(b.rx,rx)

                        if i==2:
                            self.assertEqual(b.ry,ry+0.5*cnt)
                        elif i==3:
                            self.assertEqual(b.ry,ry-0.5*cnt)
                        else:
                            self.assertEqual(b.ry,ry)

                        if i<2:
                            self.assertEqual(b.rw_i,1.+0.5*cnt)
                            self.assertEqual(b.rh_i,1.+cnt)
                        else:
                            self.assertEqual(b.rw_i,1.+cnt)
                            self.assertEqual(b.rh_i,1.+0.5*cnt)
                        self.assertFalse(b.is_stuck())

if __name__ == "__main__":
    #Just for debug
    b = boundary_limits()
    b.runTest()
    