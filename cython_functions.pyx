from libc.math cimport sqrt 
import numpy as np
import cmath
import angles
cimport cython
cimport numpy as np
import python_functions as pf

cdef double pi = cmath.pi
cdef double eps_col = 1e-15

cdef class Params:
    """    
    Class designed to store information about the triangular billiard
    """    
    cdef public double alpha, beta
    cdef public double a, b,
    cdef public CaseParams d0, d1, d2
    cdef public list ds

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.sides(alpha, beta)
        self.d0 = CaseParams(0, alpha, beta, self.a, self.b)
        self.d1 = CaseParams(1, alpha, beta, self.a, self.b)
        self.d2 = CaseParams(2, alpha, beta, self.a, self.b)
        self.ds = list([self.d0, self.d1, self.d2])

    cdef void sides(self, double alpha, double beta):
        cdef double var = pi - alpha - beta
        self.a = np.sin(alpha) / np.sin(var)
        self.b = np.sin(beta) / np.sin(var)

    cpdef CaseParams tk_d(self, double t):
        cdef double a, b

        if t < 1:
            return self.d0
        elif t < 1 + self.a:
            return self.d1
        else:
            return self.d2

cdef class CaseParams:
    """
    Class desinged for storing information from Table XX 
    """
    cdef public unsigned int _case
    cdef public double tk, tk1, tk2, sinak, sinak1, cosak, cosak1, cotak

    def __init__(self, unsigned int _case, double alpha, double beta,
                 double a, double b):
        if _case == 0:
            self.tk = 1 + a + b
            self.tk1 = 1
            self.tk2 = 1 + a
            self.sinak = np.sin(beta)
            self.sinak1 = np.sin(alpha)
            self.cosak = np.cos(beta)
            self.cosak1 = np.cos(alpha)
            self.cotak = np.cos(beta) / np.sin(beta)

        elif _case == 1:
            self.tk = 1
            self.tk1 = 1 + a
            self.tk2 = 1 + a + b
            self.sinak = np.sin(pi - alpha - beta)
            self.sinak1 = np.sin(beta)
            self.cosak = np.cos(pi - alpha - beta)
            self.cosak1 = np.cos(beta)
            self.cotak = np.cos(pi - alpha - beta) / np.sin(pi - alpha - beta)

        elif _case == 2:
            self.tk = 1 + a
            self.tk1 = 1 + a + b
            self.tk2 = 1 + a + b + 1
            self.sinak = np.sin(alpha)
            self.sinak1 = np.sin(pi - alpha - beta)
            self.cosak = np.cos(alpha)
            self.cosak1 = np.cos(pi - alpha - beta)
            self.cotak = np.cos(alpha) / np.sin(alpha)
        else:
            raise ValueError()

    @cython.cdivision(True)
    cdef double p_t_critical(self, double t):
        """
        Method of the class CaseParams returning the critical momentum
        ie a momentum needed for a trajectory
        starting at position t to finish at the opposite corner

        Input: double t, indicating current position
        Output: double p_t, indicating critical momentum
        """
        cdef double c_t, p_t, tk1, tk2

        tk1 = self.tk1
        tk2 = self.tk2

        c_t = (tk1 - t) * (1. / self.sinak) / abs(tk2 - tk1) - self.cotak
        p_t = (1.0 * c_t) / ((1 + c_t * c_t)**0.5)
        return p_t




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef (double, double) next_t_next_p(double t, double p, Params par):
    """Calculate the position and momentum after the next collision."""
    cdef CaseParams case_par
    cdef double new_p, new_t, p_t, p_expr, cap, a, b

    # access and store the lengths of sides a and b
    a = par.a   
    b = par.b

    # store the circumference of the triangle
    cap = 1 + a + b

    new_p = 0.
    new_t = 0.
    p_expr = sqrt((1. - p * p))

    if t > cap:  # making sure that variable t is in the correct range
        print('error, t out of range')
        t = t % cap

    if abs(p) >= 1:  # making sure that variable p is in the correct range
        print('error, p out of range')
        return -1, -1  # NOTE: change in behaviour!

    # computing critical momentum
    case_par = par.tk_d(t)
    p_t = case_par.p_t_critical(t)


    # check which side particle ends up after collision
    # calculate the new position and momentum
    if p > p_t:
        new_p = p_expr * case_par.sinak - p * case_par.cosak
        new_t = (case_par.tk1 - t) * p_expr / (p_expr * case_par.cosak + p * case_par.sinak) + case_par.tk1

    elif p < p_t:  
        new_p = -p_expr * case_par.sinak1 - p * case_par.cosak1

        if t > 1:
            new_t = case_par.tk - (t - case_par.tk) * p_expr / (p_expr * case_par.cosak1 - p * case_par.sinak1)
        else: 
            new_t = case_par.tk2 + b - (t * p_expr) / (p_expr * case_par.cosak1 - p * case_par.sinak1)

    else:   # in the case we end up in the corner
        print('Game over!')

    # check new position so that it does not overflow
    if new_t > cap:
        new_t = new_t % cap

    return new_t, new_p


cpdef N_collisions(double t, double p, Params par, unsigned int N):
    """Calculate the next N collisions and return them in an array."""
    cdef unsigned int j

    ts_arr = np.zeros(N, dtype=np.double)
    ps_arr = np.zeros(N, dtype=np.double)
    cdef double[:] ts = ts_arr  # memory view of the array
    cdef double[:] ps = ps_arr  # memory view of the array

    ts[0] = t
    ps[0] = p

    for j in range(1, N):
        t, p = next_t_next_p(t, p, par)

        ts[j] = t
        ps[j] = p
    return ts_arr, ps_arr



#######################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cython_pk_sum(double alpha, double beta, int N, int N_init,str txt):
    """
    Function computing finite time ergodic averages of momentum from random uniformly distributed initial conditions

    Input: angles alpha and beta, number of collisions N for each initial condition, 
    number of intial conditions N_init, string for generation of file name

    Output: Saved .npy file of length N_init, where each entry corresponds to an ergodic average
    """
    cdef Params par
    
    cdef unsigned int i
    cdef CaseParams case_par
    cdef double newp, newt, p_t, p_expr, cap, a, b
    
    cdef double[:] ts 
    cdef double[:] ps
    cdef double[:] res = np.zeros(N_init)
    cdef double sm
    
    par = Params(alpha,beta)
    a = par.a
    b = par.b
    cap = 1 + a + b
    
    # computing and scaling uniform random initial conditions
    ts = np.random.random_sample(N_init) * cap
    ps = np.random.random_sample(N_init)*2 -1 
    

    # iterationg over intial conditions
    for init in range(N_init):
        # initialising variables
        sm = ps[init]
        p = ps[init]
        t = ts[init]

        # iterating over number of collisions
        for i in range(N):

            newp = 0.
            newt = 0.
            p_expr = sqrt((1. - p * p))

            if t > cap:  # making sure that variable t is in the correct range
                t = t % cap

            if abs(p) >= 1: # making sure that variable p is in the correct range
                return -1 # NOTE: change in behaviour! 

            case_par = par.tk_d(t)
            p_t = case_par.p_t_critical(t)

            if p > p_t: 
                newp = (p_expr) * case_par.sinak - p * case_par.cosak
                newt = (case_par.tk1 - t) * (p_expr)/((p_expr) * case_par.cosak + p * case_par.sinak) +case_par.tk1

            elif p < p_t:  #p < p_t:
                newp = -(p_expr) * case_par.sinak1 - p * case_par.cosak1

                if t > 1:
                    newt = case_par.tk - (t - case_par.tk)*(p_expr)/((p_expr) * case_par.cosak1 - p * case_par.sinak1)
                else: 
                    newt = case_par.tk2 + b - (t*p_expr)/((p_expr) * case_par.cosak1 - p * case_par.sinak1)

            if newt > cap:
                newt = newt % cap

            t = newt
            p = newp
            sm += p

        # saving ergodic sum for given initial condition
        res[init] = sm/N

    # saving file
    np.save('Pk_sum/pk_sum_'+txt+'.npy',np.array(res, dtype=np.double))

