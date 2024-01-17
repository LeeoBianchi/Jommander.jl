using Test
using MPI
using Random
using Plots
using LaTeXStrings
using DelimitedFiles
include("SamplerMPI.jl")

#= READ AND DOWNGRADE WMAP MASK
mask_map = readMapFromFITS(fits*"wmap_tmask.fits", 1, Float64)
mask_map = udgrade(mask_map, 128, pess = true, threshold = 1e-4)
mask_map = nest2ring(mask_map)
mask_map.pixels = floor.(mask_map.pixels)
saveToFITS(mask_map, fits*"wmap_mask128")
plot(mask_map)
=#
rng = Random.seed!(4567)

const NSIDE = 128
W = 4*π/nside2npix(NSIDE)
const lmax = 3*NSIDE - 1
const FWHM_deg = 1
const Noise_var = 3
beam = Healpix.gaussbeam(deg2rad(FWHM_deg), lmax)
wind = Healpix.pixwin(NSIDE)
#convolution beam - pixel
A_l = beam[1:lmax + 1] .* wind[1:lmax + 1]

C_l = readClFromFITS("Planck_TTTEEE_bestfit_Cl", Float64)
InvC = 1. ./ C_l[1:lmax+1]
InvC[1:2] = [0.,0.]

MPI.Init()
comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0

if crank == root
    data_map = Healpix.readMapFromFITS("observed_map_$(NSIDE)_beam$FWHM_deg", 1, Float64)
    mask_map = Healpix.readMapFromFITS("wmap_mask128", 1, Float64)
    s_alm = Alm(lmax, lmax, zeros(ComplexF64, numberOfAlms(lmax)))
    #Noise matrix diagonal
    N_matr = ones(nside2npix(NSIDE))*Noise_var
    N_matr = ifelse.(mask_map.pixels .== 0.0, Inf, N_matr)
else
    data_map = nothing
    s_alm = nothing
    N_matr = nothing
end

d_dmap = HealpixMPI.DMap{HealpixMPI.RR}(comm)
MPI.Scatter!(data_map, d_dmap)

s_dalm = HealpixMPI.DAlm{HealpixMPI.RR}(comm)
MPI.Scatter!(s_alm, s_dalm)

local_N = MPI.Scatter(N_matr, NSIDE, comm)

aux_alm_leg = Array{ComplexF64,3}(undef, (length(s_dalm.info.mval), numOfRings(d_dmap.info.nside), 1)) # loc_nm * tot_nr
aux_map_leg = Array{ComplexF64,3}(undef, s_dalm.info.mmax+1, length(d_dmap.info.rings), 1)            # tot_nm * loc_nr

NT = 2 #set number of threads

#POWER SPECTRUM CHAIN

SamplePS!(s_dalm, d_dmap, local_N, A_l, InvC, 50, NT, "PSchain$NSIDE.txt")
