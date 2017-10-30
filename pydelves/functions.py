# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:59:30 2015

@author: Gil Tabak
@title: Potapov

Functions used by other files.

"""

import numpy as np
import numpy.linalg as la
import scipy.constants as consts
from fractions import gcd
import time
import math

def make_dict_values_to_lists_of_inputs(values,inputs):
    '''
    Make a dictionary mapping value to lists of corresponding inputs.

    Args:
        values (list of floats):
            Values in a list, corresponding to the inputs.
        inputs (list of floats):
            Inputs in a list.

    Returns:
        D (dict):
            dictionary mapping value to lists of corresponding inputs.
    '''
    D = {}
    for k, v in zip(values,inputs):
        if not math.isnan(k):
            D.setdefault(k, []).append(v)
    return D

def timeit(method):
    '''
    from https://www.andreas-jung.com/contents/a-python-decorator-for-measuring-the-execution-time-of-methods
    '''
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result
    return timed

def gcd_lst(lst):
    l = len(lst)
    if l == 0:
        return None
    elif l == 1:
        return lst[0]
    elif l == 2:
        return gcd(lst[0],lst[1])
    else:
        return gcd(lst[0],gcd_lst(lst[1:]))

def der(f,z,eps = 1e-5):
    '''
    Estimate the derivative of the function f at z

    Args:
        f (function): the function to use.

        z (complex number): point at which to evaluate derivative.

        eps(optional[complex number]): number to perturb z to find derivative.

    Returns:
        Derivative of f. (complex):

    '''
    return (f(z+eps)-f(z-eps))/(2*eps)

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
    try:
        s=sum(f(z0 + c_el) for c_el in c)/float(N)
        return s
    except:
        print "Something went wrong in estimating the limit."
        return

def factorial(n):
    '''Find the factorial of n.

    Args:
        n (integer).

    Returns:
        factorial of n. (int):

    '''
    end = 1
    for k in xrange(1,n+1):
        end *= k
    return end

def pade_approx(n):
    '''Numerator coefficients of symmetric Pade approximation of math:`e^z` of order n.

    Args:
        n (integer).

    Returns:
        Coefficients for Pade approximation numerator. (float):

    '''
    output = [0]*(n+1)
    for k in xrange(0,n+1):
        output[n-k] = float(factorial(2*n-k)) * factorial(n) / \
                      float((factorial(2*n) )* factorial(k) * factorial(n - k) )
    return output

def pade_roots(n):
    '''Extract roots of Pade polynomial.

    Args:
        n (integer).

    Returns:
        Roots of Pade polynomial. (list of complex numbers) :

    '''
    return np.roots(pade_approx(n))

def Q(z,n):
    r'''Numerator of Pade approximation of :math:`e^z`.

    Args:
        n (integer): order of approximation.

        z (complex number): point of evaluation.

    Returns:
        Value of Numerator of Pade approximation. (float):

    '''
    coeffs = pade_approx(n)
    sum = 0
    for i in xrange(0,n+1):
        sum += coeffs[i]*pow(z,n-i)
    return sum

def Pade(n,z):
    r'''Pade pproximation of :math:`e^z`

    Args:
        n (integer): order of approximation

        z (complex number): point of evaluation.

    Returns:
        Value of Pade approximation. (float):

    '''
    return Q(z,n)/Q(-z,n)

def double_up(M1,M2=None):
    r'''

    Takes a given matrix M1 and an optional matrix M2 and generates a
    doubled-up matrix to use for simulations when the doubled-up notation
    is needed. i.e.

    .. math::
        \begin{pmatrix}
            M_1 && M_2
        \end{pmatrix}
        \to
        \begin{pmatrix}
            M_1 && M_2 \\
            M_2^\# && M_1^\#
        \end{pmatrix}

    In the case M2 == None, it becomes replaced by the zero matrix.

    Args:
        M1 (matrix): matrix to double-up

        M2 (matrix): optional second matrix to double-up

    Returns:
        (complex-valued matrix):
            The doubled-up matrix.

    '''
    if M2 == None:
        M2 = np.zeros_like(M1)
    top = np.hstack([M1,M2])
    bottom = np.hstack([np.conj(M2),np.conj(M1)])
    return np.vstack([top,bottom])

def spatial_modes(roots,M1,E,delays=None):
    '''
    Obtain the spetial mode profile at each node up to a constant.
    If the delays are provided, the modes will be normalized using the delays.
    Otherwise, the modes will not be normalized.

    Args:
        roots (list of complex numbers): The eigenvalues of the system.

        M1 (matrix): The connectivity matrix among internal nodes.

        E (matrix-valued function): Time-delay matrix.

        delays (optional[list of floats]): List of delays in the network.

    Returns:
        A list of spatial eigenvectors. (list of complex-valued column matrices):
    '''

    spatial_vecs = []
    for i in xrange(len(roots)):
        evals,evecs = la.eig(M1*E(roots[i]))
        spatial_vecs.append(evecs[:,np.argmin(abs(1.-evals))])
    if delays == None:
        return spatial_vecs
    if type(delays) != list:
        raise Exception('delays must be a list of delays.')

    for mode in spatial_vecs:
        mode /= _norm_of_mode(mode,delays)
    return spatial_vecs

def inner_product_of_two_modes(root1,root2,v1,v2,delays,eps=1e-7,
                                func=lambda z : z.imag):
    '''
    This function integrates two spatial modes against each other
    along the various delays of the system. Each delay follows a
    node in the system.

    The frequency is assumed to be the imaginary part of each root.

    Args:
        root1,root2 (complex number): the two roots.

        v1,v2 (column matrices):
            the amplitude of each mode at the
            various nodes.

        delays (list of floats):
            The duration of each delay following
            each node in the system.

        eps(optional[float]):
            cutoff for two frequencies being equal

        func (optional[funciton]):
            used to transform the roots. Default
            value is set to lambda z: z.imag, meaning we take the frequency
            of each mode.

    Returns:
        The inner product of the two modes. (complex):
            Sanity check: if root1==root2 and v1==v2, returns real value.
    '''
    s = 0j
    for delay,e1,e2 in zip(delays,v1,v2):
        if abs(func(root1-root2)) < eps:
            s+=e1*np.conj(e2)*delay
        else:
            s += (e1*e2.H*1j*(np.exp(-1j*delay*func(root1-root2)) - 1. )
                        /func(root1-root2) )
    return s[0,0]

def _norm_of_mode(mode,delays):
    '''
    Find the norm of the given mode

    Args:
        mode (vector):
            column of complex numbers describing the amplitude of
            each mode at the various nodes.

        delays (list of floats):
            time delays in the network.

    Returns:
        norm (float):
            the norm of the mode.
    '''
    return np.sqrt(inner_product_of_two_modes(0,0,mode,mode,delays))

def make_normalized_inner_product_matrix(roots,modes,delays,eps=1e-12,
                                func=lambda z : z.imag):
    '''
    Given a list of roots and a list of vectors representing the
    electric field at each node of the corresponding nodes, compute
    the normalized matrix representing the inner products among the
    various modes.

    TODO: add weights for different delays to account for geometry.

    Args:
        roots (list of complex numbers):
            The roots of the various eigenmodes.

        modes (list of column matrices):
            the amplitudes of the modes at
            various nodes.

        delays (list of floats):
            The duration of each delay following
            each node in the system.

        eps(optional[float]):
            cutoff for two frequencies being equal.

        func (optional[funciton]):
            used to transform the roots. Default
            value is set to lambda z: z.imag, meaning we take the frequency
            of each mode.

    Returns:
        inner product matrix (complex-valued matrix):
            A matrix of normalized inner products representing the geometric
            overlap of the various given modes in the system.
    '''
    dim = len(roots)
    norms = [0]*dim
    for i,(root,v) in enumerate(zip(roots,modes)):
        norms[i] = inner_product_of_two_modes(root,root,v,v,delays,eps=eps,
                                        func=func)
    inner_prods = np.zeros((dim,dim),dtype='complex_')
    for i in range(dim):
        for j in range(dim):
            inner_prods[i,j] = ((inner_product_of_two_modes(roots[i],roots[j],
                                modes[i],modes[j],delays)) /
                                np.sqrt(norms[i]*norms[j]) )
    return inner_prods

def make_nonlinear_interaction(natural_freqs, modes, delays, delay_indices,
                                start_nonlin,length_nonlin,plus_or_minus_arr,
                                indices_of_refraction = None,
                                eps=1e-12):
    '''
    This function takes several (say M) natural_freqs and their corresponding modes,
    as well as the (N) delay lengths of the network, and determines the term
    we need to add to the Hamiltonian corresponding to the resulting
    nonlinearity. We assume there is a crystal going from start_nonlin to
    and has length length_nonlin. The plus_or_minus_arr is an array of
    length m of 1 or -1 used to determined whether a mode corresponds to a
    creation (1, a^\dag) or annihilation (-1,a) operator. The corresponding
    electric field integrated will be E^\dag for 1 and E for -1.

    The k-vectors are computed from the following formula:
    k = omega / v_p = omega n(omega) / c.

    If the indices of refraction n(omega_i) are given, we use them to compute
    the phase-mismatch delta_k. Otherwise we assume they are all equal to 1.

    Args:
        natural_freqs (list of complex numbers):
            The natural frequencies of the
            various eigenmodes.

        modes (list of column matrices):
            the amplitudes of the modes at
            various nodes.

        delays (list of floats):
            The duration of each delay following
            each node in the system.

        delay_indices (int OR list/tuple of ints):
            the index representing the
            delay line along which the nonlinearity lies. If given a list/tuple
            then the nonlinearity interacts the N different modes.

        start_nonlin (float OR list/tuple of floats):
            the beginning of the
            nonlinearity. If a list/tuple then each nonlinearity begins at a
            different time along its corresponding delay line.

        length_nonlin (float):
            duration of the nonlinearity in terms of length.

        plus_or_minus_arr (array of 1s and -1s):
            Creation/annihilation of
            a photon in each of the given modes.

        indices_of_refraction (float/int or list/tuple of float/int):
            the
            indices of refraction corresponding to the various modes. If float
            or int then all are the same.

        eps(optional[float]):
            cutoff for two frequencies being equal.

    Returns:
        nonlinear interaction (complex):
            strength of nonlinearity.
    '''

    M = len(natural_freqs)
    if len(modes) != M:
        raise Exception('number of modes different than number of natural_freqs.')

    if type(delay_indices) == int:
        delay_indices = [delay_indices] * M
    elif not type(delay_indices) in [list,tuple]:
        raise Exception('delay_indices must be an int or a list/tuple')

    if type(start_nonlin) in [int,float]:
        start_nonlin = [start_nonlin] * M
    elif not type(start_nonlin) in [list,tuple]:
        raise Exception('start_nonlin must be an int/float or a list/tuple')

    if length_nonlin < 0:
        raise Exception('length_nonlin must be greater than 0.')

    for delay_index,start_loc in zip(delay_indices,start_nonlin):
        if start_loc < 0:
            raise Exception('each element of start_nonlin must be greater than 0.')
        # Below is the condition we would need to check when
        # the index of refraction is 1. In the case the index of refraction
        # is different, length_nonlin is multiplied by the refractiive index.
        # However, the duration of the delay lengthens by the same amount so the
        # condition remains unchanged.
        if length_nonlin / consts.c + start_loc > delays[delay_index]:
            raise Exception('length_nonlin + start_loc must be less than the '
                           +'delay of index delay_index for start_loc in '
                           +'start_nonlin and delay_index in delay_indices.')

    if indices_of_refraction is None:
        indices_of_refraction = [1.] * M
    elif type(indices_of_refraction) in [float,int]:
        indices_of_refraction = [float(indices_of_refraction)] * M
    elif not type(indices_of_refraction) in [list,tuple]:
        raise Exception('indices_of_refraction is not a float, integer, list, '
                       +'tuple, or None.')

    def pick_conj(m,sign):
        if sign == 1:
            return m
        elif sign == -1:
            return np.conj(m)
        else:
            raise Exception('bad input value -- must be 1 or -1.')

    values_at_nodes = [m_vec[delay_index,0] for m_vec,delay_index
        in zip(modes,delay_indices)]
    delta_k = ( sum([n*omega*sign for n,omega,sign
        in zip(indices_of_refraction,natural_freqs,plus_or_minus_arr)])
        / consts.speed_of_light )
    const = np.prod([pick_conj(m*np.exp(-1j*delta_k*start_loc),sign)
            for m,sign,start_loc
            in zip(values_at_nodes,plus_or_minus_arr,start_nonlin)])

    if abs(delta_k) < eps: ## delta_k \approx 0
        return const * length_nonlin
    else:
        return 1j*const*(np.exp(-1j*delta_k*length_nonlin) - 1 ) / delta_k
