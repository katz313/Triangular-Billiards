# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:07:05 2020

@author: katerina
"""
import numpy as np
import pyfftw
import pickle

import cython_functions as cf

def get_fft_ifft(_f, hat_f, flags):
    """Return FFTW objects for fft and ifft operations on _f and hat_f."""

    try:
        # Try to plan our transforms with the wisdom we have already
        fft = pyfftw.FFTW(_f, hat_f,
                          direction='FFTW_FORWARD',
                          flags=flags + ('FFTW_WISDOM_ONLY', ))
        ifft = pyfftw.FFTW(hat_f, _f,
                           direction='FFTW_BACKWARD',
                           flags=flags + ('FFTW_WISDOM_ONLY', ),
                           normalise_idft=True)

    except RuntimeError as e:
        # If we don't have enough wisdom, warn the user and proceed.
        print(e)
        start = time.perf_counter()
        fft = pyfftw.FFTW(_f, hat_f,
                          direction='FFTW_FORWARD',
                          flags=flags)
        ifft = pyfftw.FFTW(hat_f, _f,
                           direction='FFTW_BACKWARD',
                           flags=flags,
                           normalise_idft=True)
        print('Generating wisdom took {}s'.format(time.perf_counter() - start))
    return fft, ifft


def fftw_corr(f,set_zero):
    """Calculate the auto-correlation of a vector, f."""
    flags = ('FFTW_MEASURE',)
    if len(f) % 2 == 1:
        raise ValueError('Length of the ffted vector is not in the form 2^',len(f))

    len_f = len(f)
    _f = pyfftw.empty_aligned(len_f, dtype='float32')
    hat_f = pyfftw.empty_aligned(len_f//2+1, dtype='complex64')

    fft_object, ifft_object = get_fft_ifft(_f, hat_f, flags)

    _f[:] = f  # note we mustn't copy until after get_fft_ifft()

    fft_object()

    # modify in-place so that hat_f remains byte-aligned
    hat_f *= np.conjugate(hat_f)
    hat_f /= len_f

    # if discount, set hat_f[0]=0
    if set_zero:
        hat_f[0] = 0

    ifft_object()

    # normalise
    _f /= _f[0]  
    return _f 



def fftw_aux(f, set_zero=True):
    """As fft_aux but uses fftw.interfaces."""
   
    if len(f) % 2 == 1:
        raise ValueError('Length of the ffted vector is not in the form 2^')

    hat_f = pyfftw.interfaces.numpy_fft.fft(f)
    ln_f = len(f)
    hat_f = (hat_f * np.conjugate(hat_f)) / ln_f

    if set_zero:
        hat_f[0] = 0

    hat_f = pyfftw.interfaces.numpy_fft.ifft(hat_f).real 

    # normalizing
    hat_f = hat_f / hat_f[0]

    return hat_f


def fft_aux(f, set_zero=True):
    """Auxilliary function calculating autocorrelation of a given vector f."""
    if len(f) % 2 == 1:
        raise ValueError('Length of the ffted vector is not in the form 2^')

    hat_f = np.fft.fft(f)
    ln_f = len(f)
    hat_f = (hat_f * np.conjugate(hat_f)) / ln_f

    if set_zero:
        hat_f[0] = 0    
    
    hat_f = np.fft.ifft(hat_f).real  
    
    # normalizing
    hat_f = hat_f / hat_f[0]

    return hat_f



def corr_p_sint_cuts(alpha, beta, txt,
                fft_func=fftw_corr, n_cuts=1000,
                N_one_cut=2 ** 17,
                set_zero=False
                ):

    """Computes the autocorrelation function for a given triangle,
    number of cuts, and orbit lenght for momentum and cyclic position.

    Input:
        angles alpha,beta
        string for saving txt
        interface for computing the autocorrelation fft_func
        number of cuts n_cuts
        number of collisions per cut N_one_cut
        discounted/non-discounted as set_zero
    Output: 
        save files of autocorrelation of momentum, 
        saved file of autocorrelation of momentum
    """

    # calculating how much of the result we actually save
    k = int(N_one_cut / (2**3))
    if N_one_cut == 2**30:
        k = int(N_one_cut / 2**4)

    # inicializing the triangle
    par = cf.Params(alpha,beta)
     
    # calculating circumference
    cap = 1 + par.a + par.b

    # inicializing arrays
    corr_ps = np.zeros(k)
    corr_sint = np.zeros(k)
    counter = 0

    # generating random initial condition
    
    T = np.random.rand()
    P = np.random.rand()

    # try to load wisdom
    print('Trying to load wisdom')
    try:
        with open('fft.wisdom', 'rb') as the_file:
            wisdom = pickle.load(the_file)
            pyfftw.import_wisdom(wisdom)
            print('Wisdom imported')
    except FileNotFoundError:
        print('Warning: wisdom could not be imported')

    # iterating over cuts
    for cut in range(n_cuts):
        # compute position and momentum of length N_one_cut+1
        ts, ps = cf.N_collisions(t=T, p=P, par=par,
                                        N=N_one_cut)
        # reasign starting point                                      
        T = ts[-1]
        P = ps[-1]
        counter += 1

        # compute autocorrelation of momentum and add
        temp = fft_func(ps,set_zero)
        corr_ps += temp[:k]

        # compute autocorrelation of cyclic position and add
        temp = fft_func(np.sin(2*np.pi*ts/cap),set_zero)
        corr_sint += temp[:k]

    # average over the number of cuts
    corr_ps = corr_ps / counter
    corr_sint = corr_sint/counter

    # save files
    np.save('Master_corr_cesaro/corr_ps_'+txt+'.npy', corr_ps)
    np.save('Master_corr_cesaro/corr_sint_'+txt+'.npy', corr_sint)

    # save wisdom
    print('saving wisdom')
    with open('fft.wisdom', 'wb') as the_file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, the_file)

#

def corr_p_rand(alpha, beta, txt,
                fft_func=fftw_corr, n_init=1000,
                N_one_cut=2 ** 17, set_zero=False
                ):

    """Computes the autocorrelation function for a given triangle,
    number of initial conditions, and orbit lenght for momentum and cyclic position.

    Input:
        angles alpha,beta
        string for saving txt
        interface for computing the autocorrelation fft_func
        number of initial conditions n_init
        number of collisions per cut N_one_cut
        discounted/non-discounted as set_zero
    Output: 
        save files of autocorrelation of momentum, 
        saved file of autocorrelation of momentum
    """

    # calculating how much of the result we actually save
    k = int(N_one_cut / (2**3))
    if N_one_cut == 2**30:
        k = int(N_one_cut / 2**4)

    # inicializing the triangle
    par = cf.Params(alpha,beta)
     
    # calculating circumference
    cap = 1 + par.a + par.b

    # inicializing arrays
    corr_ps = np.zeros(k)
    corr_sint = np.zeros(k)
    counter = 0

    # generating random initial condition
    TS = np.random.rand(n_init)*cap
    PS = np.random.rand(n_init)*2 - np.ones(n_init)

    # try to load wisdom
    print('Trying to load wisdom')
    try:
        with open('fft.wisdom', 'rb') as the_file:
            wisdom = pickle.load(the_file)
            pyfftw.import_wisdom(wisdom)
            print('Wisdom imported')
    except FileNotFoundError:
        print('Warning: wisdom could not be imported')

    # iterate over initial conditions
    for i in range(n_init):
        # compute vector of collisions of desired lenght
        ts, ps = cf.N_collisions(t=TS[i], p=PS[i], par=par,
                                        N=N_one_cut-1)
        counter += 1
        # compute autocorrelation of momentum and save
        temp = fft_func(ps,set_zero)
        corr_ps += temp[:k]

        # conmpute autocorrelation of cyclic position and save
        temp = fft_func(np.sin(2*np.pi*ts/cap),set_zero)
        corr_sint += temp[:k]

    # average over the number of initial conditions
    corr_ps = corr_ps / counter
    corr_sint = corr_sint/counter

    # save files
    np.save('Master_corr_cesaro/corr_ps_'+txt+'_rand.npy', corr_ps)
    np.save('Master_corr_cesaro/corr_sint_'+txt+'_rand.npy', corr_sint)
    
    # save wisdom
    print('saving wisdom')
    with open('fft.wisdom', 'wb') as the_file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, the_file)
