include("ConjGrad.jl")

using Random
using Healpix
using MPI
using HealpixMPI

######################################################################
#NOTE:
# - aux_alm_leg: leg matrix on the "alm" side: it has (loc_nm, tot_nr, ncomp) shape
# - aux_out_leg: leg matrix on the "map" side: it has (tot_nm, loc_nr, ncomp) shape

######################################################################
"""
    Gets fluctuations on RHS
"""
function get_ω(
    ω_map::HealpixMPI.DMap{S,Float64}, #aux d_map
    ω₁_lm::HealpixMPI.DAlm{S,ComplexF64},
    aux_alm_leg::StridedArray{ComplexF64,3},
    aux_map_leg::StridedArray{ComplexF64,3},
    A_l::AbstractArray,
    InvC::AbstractVector,
    local_N::AbstractVector;
    nthreads::Integer = 1
) where {S<:HealpixMPI.Strategy}
    @views ω_map.pixels[:,1] = randn(size(ω_map)[1]) ./ sqrt.(local_N)
    Healpix.adjoint_alm2map!(ω_map, ω₁_lm, aux_map_leg, aux_alm_leg; nthreads = nthreads) #Yᵀ
    Healpix.almxfl!(ω₁_lm, A_l) #we convolve it with the beam
    ω₂_lm = deepcopy(ω₁_lm)
    Healpix.synalm!(InvC, ω₂_lm)
    ω₁_lm.alm .+= ω₂_lm.alm
    ω₁_lm
end

#######################################################################

#r.h.s. in harmonic space without ω and noise cov matrix (stored as array (diag))

#with aux alm and map in input, more efficient!!
function get_rhs_noω(
    d_map::HealpixMPI.DMap{S,Float64},
    aux_lm::HealpixMPI.DAlm{S,ComplexF64},
    aux_map::HealpixMPI.DMap{S,Float64},
    aux_alm_leg::StridedArray{ComplexF64,3},
    aux_map_leg::StridedArray{ComplexF64,3},
    A_l::AbstractArray,
    InvC::AbstractVector,
    local_N::AbstractVector; #to distribute with Scatter in the main
    nthreads::Integer = 1
) where {S<:HealpixMPI.Strategy}
    @views aux_map.pixels[:,1] = d_map.pixels[:,1] ./ local_N
    Healpix.adjoint_alm2map!(aux_map, aux_lm, aux_map_leg, aux_alm_leg; nthreads = nthreads) #Y^t = W^-1 Y^-1
    Healpix.almxfl!(aux_lm, A_l) #Aᵀ, this is the rhs with no ω's
    res_lm = aux_lm + get_ω(aux_map, deepcopy(aux_lm), aux_alm_leg, aux_map_leg, A_l, InvC.*0., local_N.*Inf; nthreads = nthreads)
    res_lm
end

#######################################################################

#r.h.s. in harmonic space without ω and noise cov matrix (stored as array (diag))

#with aux alm and map in input, more efficient!!
function get_rhs_ω(
    d_map::HealpixMPI.DMap{S,Float64},
    aux_lm::HealpixMPI.DAlm{S,ComplexF64},
    aux_map::HealpixMPI.DMap{S,Float64},
    aux_alm_leg::StridedArray{ComplexF64,3},
    aux_map_leg::StridedArray{ComplexF64,3},
    A_l::AbstractArray,
    InvC::AbstractVector,
    local_N::AbstractVector; #to distribute with Scatter in the main
    nthreads::Integer = 1
) where {S<:HealpixMPI.Strategy}
    @views aux_map.pixels[:,1] = d_map.pixels[:,1] ./ local_N
    get_ω(aux_map, deepcopy(aux_lm), aux_alm_leg, aux_map_leg, A_l, InvC, local_N; nthreads = nthreads)
end

#######################################################################

#r.h.s. in harmonic space with ω and noise cov matrix (stored as array (diag))

#with aux alm and map in input, more efficient!!
function get_rhs(
    d_map::HealpixMPI.DMap{S,Float64},
    aux_lm::HealpixMPI.DAlm{S,ComplexF64},
    aux_map::HealpixMPI.DMap{S,Float64},
    aux_alm_leg::StridedArray{ComplexF64,3},
    aux_map_leg::StridedArray{ComplexF64,3},
    A_l::AbstractArray,
    InvC::AbstractVector,
    local_N::AbstractVector; #to distribute with Scatter in the main
    nthreads::Integer = 1
) where {S<:HealpixMPI.Strategy}
    @views aux_map.pixels[:,1] = d_map.pixels[:,1] ./ local_N
    Healpix.adjoint_alm2map!(aux_map, aux_lm, aux_map_leg, aux_alm_leg; nthreads = nthreads) #Y^t = W^-1 Y^-1
    Healpix.almxfl!(aux_lm, A_l) #Aᵀ, this is the rhs with no ω's
    aux_lm += get_ω(aux_map, deepcopy(aux_lm), aux_alm_leg, aux_map_leg, A_l, InvC, local_N; nthreads = nthreads)
    aux_lm
end
#######################################################################

#l.h.s. for noise cov matrix, applies the lhs operator to the x_lm passed
#(InvC + Aᵀ Yᵀ InvN Y A) x_lm
#it does NOT change x_lm
#implements aux map and alms passed, more efficient

function get_lhs(
    dalm::HealpixMPI.DAlm{S,ComplexF64},
    aux_lm::HealpixMPI.DAlm{S,ComplexF64},
    aux_map::HealpixMPI.DMap{S,Float64},
    aux_alm_leg::StridedArray{ComplexF64,3},
    aux_map_leg::StridedArray{ComplexF64,3},
    A_l::AbstractArray,
    InvC::AbstractVector,
    local_N::AbstractVector;
    nthreads::Integer = 1
) where {S<:HealpixMPI.Strategy}
    aux_lm.alm = deepcopy(dalm.alm)
    Healpix.almxfl!(aux_lm, A_l) #A
    Healpix.alm2map!(aux_lm, aux_map, aux_alm_leg, aux_map_leg; nthreads = nthreads) #Y
    @views aux_map.pixels[:,1] ./= local_N #InvN
    Healpix.adjoint_alm2map!(aux_map, aux_lm, aux_map_leg, aux_alm_leg; nthreads = nthreads) #Yᵀ
    Healpix.almxfl!(aux_lm, A_l) #Aᵀ
    aux_lm.alm .+= almxfl(dalm, InvC).alm #InvC x_lm
    aux_lm #full lhs
end

###############################################################################

"""
    Forward step in Gibbs sampling: sample map s given C.
    Implements the CG
"""
function s_given_C!(
    alm::HealpixMPI.DAlm{S,ComplexF64},     #initial guess for CG (usually previous step)
    d_map::HealpixMPI.DMap{S,Float64},   #observed map (data)
    aux_lm::HealpixMPI.DAlm{S,ComplexF64},  #auxiliary
    aux_map::HealpixMPI.DMap{S,Float64}, #auxiliary
    aux_alm_leg::StridedArray{ComplexF64,3},
    aux_map_leg::StridedArray{ComplexF64,3},
    local_N::AbstractArray,            #local subset of noise cov matrix diagonal
    A_l::AbstractArray,                 #beam
    InvC::AbstractArray;               #signal power spectrum (sampled from backward step)
    nthreads = 1) where {S<:HealpixMPI.Strategy}
    A(x_lm) = get_lhs(x_lm, aux_lm, aux_map, aux_alm_leg, aux_map_leg, A_l, InvC, local_N; nthreads = nthreads)
    b = get_rhs(d_map, aux_lm, aux_map, aux_alm_leg, aux_map_leg, A_l, InvC, local_N; nthreads = nthreads)
    alm.alm = CG(A, b, alm; i_max = 5000).alm #update alm in c using the previous step as initial guess
end


##############################################################################


"""
    Backward step in Gibbs sampling: sample C given a map s
    1) compute power spectrum σ_ℓ from s
    2) draw 2ℓ - 1 samples ρ_ℓ^i ~ N(0,1) for each ℓ, compute ρ_ℓ = sum_i((ρ_ℓ^i)^2)
    3) compute new PS as: C_l = σ_ℓ/ρ_ℓ
"""

#implements step 2) returning the ρ_ℓ
function get_ρ(lmax)
    rho_l = Vector{Float64}(undef, lmax + 1)
    rho_l[1] = 1. #monopole l=0
    for l in 1:lmax
        rho_l_i = randn(Float64, 2*l - 1)
        rho_l[l+1] = sum(rho_l_i .^ 2)
    end
    rho_l
end

function C_given_s!(
    alm::HealpixMPI.DAlm{S,ComplexF64},
    InvC::AbstractVector,
    ) where {S<:HealpixMPI.Strategy}
    l_s = Vector{Float64}(0:alm.info.lmax)*2 .+1
    σ_l = alm2cl(alm) .* l_s #σ_l = Σ |a_lm|²
    #σ_l[1] = 0. #we remove monopole
    C_l = σ_l ./ get_ρ(alm.info.lmax)
    InvC[3:end] = 1. ./ C_l[3:end]
    InvC[1:2] = [0., 0.] #remove mono&dipole
    C_l #for plotting purposes
end

function convolve!(map::HealpixMap, A_l::Vector{Float64})
    alm = map2alm(map, niter = 5) #Y^-1
    almxfl!(alm, A_l)
    alm2map!(alm, map) #Y
end

function convolve(map::HealpixMap, A_l::Vector{Float64})
    newmap = deepcopy(map)
    convolve!(newmap, A_l)
    newmap
end

function SamplePS!(s_dalm, d_dmap, local_N, A_l, InvC, chain_length, nthreads, filename; root = 0)
    #set up
    aux_dmap = deepcopy(d_dmap)
    aux_dalm = deepcopy(s_dalm)
    aux_alm_leg = Array{ComplexF64,3}(undef, (length(s_dalm.info.mval), numOfRings(d_dmap.info.nside), 1)) # loc_nm * tot_nr
    aux_map_leg = Array{ComplexF64,3}(undef, s_dalm.info.mmax+1, length(d_dmap.info.rings), 1)            # tot_nm * loc_nr
    #initialize
    s_given_C!(s_dalm, d_dmap, aux_dalm, aux_dmap, aux_alm_leg, aux_map_leg, local_N, A_l, InvC; nthreads = nthreads)
    Cl = C_given_s!(s_dalm, InvC)
    for c in 1:chain_length
        s_given_C!(s_dalm, d_dmap, aux_dalm, aux_dmap, aux_alm_leg, aux_map_leg, local_N, A_l, InvC; nthreads = nthreads)
        Cl = C_given_s!(s_dalm, InvC)
        if crank == root
            println("####### N SAMPLE = $c ########")
            open(filename, "a") do io
                writedlm(io, reshape(Cl, 1, :)) #we add the Cl's sampled
            end
        end
        MPI.Barrier(d_dmap.info.comm)
    end
end

#saves both PS and map
function SamplePSMap!(s_dalm, d_dmap, local_N, A_l, InvC, chain_length, nthreads, PSfilename, Mapfilename; root = 0)
    #set up
    aux_dmap = deepcopy(d_dmap)
    aux_dalm = deepcopy(s_dalm)
    aux_alm_leg = Array{ComplexF64,3}(undef, (length(s_dalm.info.mval), numOfRings(d_dmap.info.nside), 1)) # loc_nm * tot_nr
    aux_map_leg = Array{ComplexF64,3}(undef, s_dalm.info.mmax+1, length(d_dmap.info.rings), 1)            # tot_nm * loc_nr
    halm = Alm(s_dalm.info.lmax, s_dalm.info.lmax)
    #initialize
    s_given_C!(s_dalm, d_dmap, aux_dalm, aux_dmap, aux_alm_leg, aux_map_leg, local_N, A_l, InvC; nthreads = nthreads)
    Cl = C_given_s!(s_dalm, InvC)
    for c in 1:chain_length
        s_given_C!(s_dalm, d_dmap, aux_dalm, aux_dmap, aux_alm_leg, aux_map_leg, local_N, A_l, InvC; nthreads = nthreads)
        Cl = C_given_s!(s_dalm, InvC)
        if crank == root
            println("####### N SAMPLE = $c ########")
            open(PSfilename, "a") do io
                writedlm(io, reshape(Cl, 1, :)) #we add the Cl's sampled
            end
            MPI.Gather!(s_dalm, halm)
            open(Mapfilename, "a") do io
                writedlm(io, reshape(alm2map(halm, d_dmap.info.nside), 1, :)) #we add the Cl's sampled
            end
        end
        MPI.Barrier(d_dmap.info.comm)
    end
end
