def get_root_bounds(roots):
    x_lmt = [None,None]
    y_lmt = [None,None]
    for root in roots:
        if x_lmt[0] is None or x_lmt[0]>root.real:
            x_lmt[0] = root.real
        if x_lmt[1] is None or x_lmt[1]<root.real:
            x_lmt[1] = root.real
        if y_lmt[0] is None or y_lmt[0]>root.imag:
            y_lmt[0] = root.imag
        if y_lmt[1] is None or y_lmt[1]<root.imag:
            y_lmt[1] = root.imag
    return x_lmt, y_lmt

def almost_equal(el1,el2,eps=1e-7):
    if abs(el1 - el2) < eps:
        return True
    else: return False  

def two_sets_almost_equal(S1,S2,eps=1e-7):
    '''
    Tests if two iterables have the same elements up to some tolerance eps.

    Args:
        S1,S2 (lists): two lists
        eps (optional[float]): precision for testing each elements

    Returns:
        True if the two sets are equal up to eps, false otherwise
    '''
    if len(S1) != len(S2):
        return False

    ran2 = range(len(S2))
    for i in range(len(S1)):
        found_match = False
        for j in ran2:
            if almost_equal(S1[i],S2[j],eps):
                found_match = True
                ran2.remove(j)
                break
        if not found_match:
            return False
    return True