# Triangular billiards

## Code overview
`cython_functions`
Contains cython code of the following:
* class `Params`: stores basic information about the triangular billiard
* class `CaseParams`: stores information for computing collisions, corresponds to Table 1-A
- `next_t_next_p`: computes coordinates of the next collision
- `N_collisions`: returns initial condition and computed coordinates of the next $N-1$ collisions 
- `cython_pk_sum`: returns ergodic averages of momentum for given number of initial conditions and for a given orbit lenght

`python_functions`
Contains python code of the following:
- `get_fft_ifft`: returns FFT and iFFT objects from the FFTW package
- `fftw_corr`: returns autocorrelation of a vector computed using the FFTW package
- `fftw_aux`: returns autocorrelation of a vector computed using FFTW interface (slower than `fftw_corr`)
- `fft_aux`: returns autocorrelation of a vector computed using NumPy
- `corr_p_sint_cuts`: returns autocorrelation function computed using the cuts method
- `corr_p_sint_rand`: returns autocorrelation function computed using the random method

`angles`
Contains angles used in our computations.

`setup_cython_functions`
File necessary for compiling `cython_functions`

`all_code_jupyter`
Contains all above in as a Jupyter notebook (easier to compile)
