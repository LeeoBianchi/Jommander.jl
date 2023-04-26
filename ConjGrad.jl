using LinearAlgebra
using Dates
include("../HealpixMPI/src/HealpixMPI.jl")
using .HealpixMPI
import Healpix

function CG(    #Ax = b  ----> x = InvA b
    A::Function,
    b::Healpix.Alm, #rhs
    x::Healpix.Alm; #initial guess
    i_max::Integer = 1000,
    ϵ = 1e-10
)
    i = 1
    r = b - A(x)#r = b - Ax
    d = deepcopy(r) #d = r
    q = deepcopy(d) #just initializing q
    δ_new = r ⋅ r #δ_new = rᵀr
    δ_0 = δ_new
    print("########################## \n")
    print("####### CG SOLVER ######## \n")
    print("########################## \n")
    println("δ_0 = $δ_0")
    while i<i_max && δ_new > ϵ^2 * δ_0
        q = A(d) #q = Ad
        α = δ_new / (d ⋅ q) #α = δ_new/(dᵀq)
        x += d * α #x = x + αd FIXME
        if i % 50 == 0 #if i is divisible by 50
            r = b - A(x) #r = b - Ax
            print("i = $i")
            println("δ = $δ_new")
        else
            r -= q * α #r = r - αq  FIXME
        end
        δ_old = δ_new
        δ_new = r ⋅ r #δ_new = rᵀr
        β = δ_new/δ_old
        d = r + d * β #d = r + βd #FIXME
        i += 1
    end
    print("number of iterations = $i")
    println("converged with δ_final = $δ_new")
    x
end

#HealpixMPI-optimized version
function CG(    #Ax = b  ----> x = InvA b
    A::Function,
    b::HealpixMPI.DAlm{S,N,I}, #rhs
    x::HealpixMPI.DAlm{S,N,I}; #initial guess
    i_max::Integer = 1000,
    ϵ = 1e-10
) where {S<:HealpixMPI.Strategy, N<:Number, I<:Integer}
    i = 1
    r = b - A(x)#r = b - Ax
    d = deepcopy(r) #d = r
    q = deepcopy(d) #just initializing q
    δ_new = r ⋅ r #δ_new = rᵀr
    δ_0 = δ_new
    print("########################## \n")
    print("####### CG SOLVER ######## \n")
    print("########################## \n")
    println("δ_0 = $δ_0")
    while i<i_max && δ_new > ϵ^2 * δ_0
        q.alm = A(d).alm #q = Ad
        α = δ_new / (d ⋅ q) #α = δ_new/(dᵀq)
        @. x.alm += d.alm * α #x = x + αd
        if i % 50 == 0 #if i is divisible by 50
            r = b - A(x) #r = b - Ax
            print("i = $i")
            println("δ = $δ_new")
        else
            @. r.alm -= q.alm * α #r = r - αq
        end
        δ_old = δ_new
        δ_new = r ⋅ r #δ_new = rᵀr
        β = δ_new/δ_old
        @. d.alm = r.alm + d.alm * β #d = r + βd
        i += 1
    end
    print("number of iterations = $i")
    println("converged with δ_final = $δ_new")
    x
end

#=
function CG(    #Ax = b  ----> x = InvA b
    A::Function,
    b::HealpixMPI.DAlm{S,N,I}, #rhs
    x::HealpixMPI.DAlm{S,N,I}; #initial guess
    i_max::Integer = 1000,
    ϵ = 1e-10
)where {S<:HealpixMPI.Strategy, N<:Number, I<:Integer}
    i = 1
    r = b - A(x)#r = b - Ax
    d = deepcopy(r) #d = r
    q = deepcopy(d) #just initializing q
    δ_new = r ⋅ r #δ_new = rᵀr
    δ_0 = δ_new
    print("########################## \n")
    print("####### CG SOLVER ######## \n")
    print("########################## \n")
    println("δ_0 = $δ_0")
    while i<i_max && δ_new > ϵ^2 * δ_0
        q = A(d) #q = Ad
        α = δ_new / (d ⋅ q) #α = δ_new/(dᵀq)
        x += d * α #x = x + αd FIXME
        if i % 50 == 0 #if i is divisible by 50
            r = b - A(x) #r = b - Ax
            print("i = $i")
            println("δ = $δ_new")
        else
            r -= q * α #r = r - αq  FIXME
        end
        δ_old = δ_new
        δ_new = r ⋅ r #δ_new = rᵀr
        β = δ_new/δ_old
        d = r + d * β #d = r + βd #FIXME
        i += 1
    end
    print("number of iterations = $i")
    println("converged with δ_final = $δ_new")
    x
end
=#

#benchmark version, to measure the pure iterations with no overhead
function CG_benchmark(    #Ax = b  ----> x = InvA b
    A::Function,
    b::HealpixMPI.DAlm{S,N,I}, #rhs
    x::HealpixMPI.DAlm{S,N,I},
    r::HealpixMPI.DAlm{S,N,I}, #r = b - A(x)
    d::HealpixMPI.DAlm{S,N,I}, #d = deepcopy(r)
    q::HealpixMPI.DAlm{S,N,I}, #q = deepcopy(d)
    δ_new::Float64; #iδ_new = r ⋅ r
    i_max::Integer = 1000
) where {S<:HealpixMPI.Strategy, N<:Number, I<:Integer}
    i = 1
    δ_0 = δ_new
    print("########################## \n")
    print("####### CG SOLVER ######## \n")
    print("########################## \n")
    println("δ_0 = $δ_0")
    while i<i_max #&& δ_new > ϵ^2 * δ_0
        q.alm = A(d).alm #q = Ad
        α = δ_new / (d ⋅ q) #α = δ_new/(dᵀq)
        @. x.alm += d.alm * α #x = x + αd
        if i % 50 == 0 #if i is divisible by 50
            r = b - A(x) #r = b - Ax
            print("i = $i")
            println("δ = $δ_new")
        else
            @. r.alm -= q.alm * α #r = r - αq
        end
        δ_old = δ_new
        δ_new = r ⋅ r #δ_new = rᵀr
        β = δ_new/δ_old
        @. d.alm = r.alm + d.alm * β #d = r + βd
        i += 1
    end
    print("number of iterations = $i")
    println("converged with δ_final = $δ_new")
    x
end

#FIXME: implement pixel space max difference!
function CG_conv_check(
    A::Function,
    b::B,
    x::X;
    i_max::Integer = 10000
) where {B, X}
    i = 0
    r = b - A(x)#r = b - Ax
    d = deepcopy(r) #d = r
    δ_new = r ⋅ r #δ_new = rᵀr
    δ_0 = δ_new
    print("δ_0 = $δ_0 \n")
    x_s = Vector(undef, Int(i_max//10))
    j = 1
    while i<i_max
        q = A(d) #q = Ad
        α = δ_new / (d ⋅ q) #α = δ_new/(dᵀq)
        x += d * α #x = x + αd
        if i % 10 == 0 #if i is divisible by 10
            r = b - A(x) #r = b - Ax
            print("i = $i \n")
            x_s[j] = x
            j += 1
        else
            r -= α * q #r = r - αq
        end
        δ_old = δ_new
        δ_new = r ⋅ r #δ_new = rᵀr
        β = δ_new/δ_old
        d = r + β * d #d = r + βd
        i += 1
    end
    print("number of iterations = $i \n")
    x_s
end
