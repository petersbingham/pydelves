# pydelves
Python Implementation of the Delves Routine.

Finds the roots of a function `f` in the complex plane inside a rectangular region.
The implementation is based on the method in the following paper:

 Delves, L. M., and J. N. Lyness. "A numerical method for locating the zeros of
 an analytic function." Mathematics of computation 21.100 (1967): 543-560.

The main idea is to compute integrals of functions of the form
`z^k f'/f` around a contour, for integer values of k. Here `f'` denotes the
derivative of `f`. The resulting values of the contour integrals are proportional
to `\sum_i z_i^k`, where i is the index of the roots. See https://en.wikipedia.org/wiki/Rouch%C3%A9%27s_theorem.

This project was forked from the module contained in https://github.com/tabakg/potapov_interpolation. Github user tabakg is credited with providing an initial implementation, which is contained within the second commit for purpose of comparison.

## Installation

Clone the repository and install with the following commands:

    git clone https://github.com/petersbingham/pydelves.git
    cd pydelves
    python setup.py install
    
## Dependencies

numpy and scipy.
    
## Usage

The following example is taken from the file examples.py and shows code to calculate the roots of the sine function that lie within the specified region:

    from pydelves import *
    f = lambda z: np.sin(z)
    fp = lambda z: np.cos(z)
    x_cent = y_cent = 0.
    width = height = 5.*np.pi+1e-5

    status,roots = droots(f,fp,x_cent,y_cent,width,height)
    print_status(status)  # or was_warning(status)
 
The technique with the default parameters has worked well for simple systems. However for expensive `f`s and `f'`s the routine can become less reliable (although failure to find roots should be indicated via the returned status) and slow for the default parameters. Four functions are provided for changing the parameters:
  * set_delves_routine_parameters
  * set_muller_parameters
  * set_mode_parameters
  * set_advanced_parameters

Various modes are also provided to control recursion and mechanisms for (or not) validating the roots. The user is referred to the comments inside the main module file, which will usually be found with a path like (on windows) C:\Python27\Lib\site-packages\pydelves\\_\_init__.py

The following is a brief description of how the routine works (in default mode):

1. Roots are located around the boundary. These are subtracted from `f`. This is so that there are no poles (divide by zeros) when performing the contour integral. An N is provided for the number of points around the region; used for both calculation of the boundary roots and for the Roche integral (see https://en.wikipedia.org/wiki/Rouch%C3%A9%27s_theorem).
2. The number of remaining roots in the region is calculated using Roche's theorem (ie the integral is performed).
3. If this number of roots is accurate and less than some set maximum then the coefficients are obtained and the polynomial is solved to give 'rough' positions for the roots. These 'rough' positions are then used as starting points for muller searches.
4. If the muller fails to locate all of the roots, the roche estimate is inaccurate or if the power of the polynomial is too big to solve stably then the region is subdivided and the process applied to these subregions. 
5. The routine continues recursively until all of the roots are found or the set max number of steps is hit.
