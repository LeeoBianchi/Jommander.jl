# Jommander.jl

## A Parallel Julia Implementation of CMB Gibbs Sampling

Jommander exploits the package [HealpixMPI.jl](https://github.com/LeeoBianchi/HealpixMPI.jl) to implement a parallel and Julia-only Gibbs Sampling algorithm of CMB power spectrum.

Mathematically, the algorithm works by solving, through a conjugate gradient method, the following equation:



## Run
To run an example of CMB power spectrum sampling, starting from an observed simulated map with noise and incomplete sky coverage, use the script `PSchainMPI.jl` as:
````shell
$ mpiexec -n {Ntask} julia PSchainMPI.jl
````
