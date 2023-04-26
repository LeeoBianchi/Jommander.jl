using Healpix
using Plots
using LaTeXStrings
using DelimitedFiles
using Statistics
const NSIDE = 128
lmax = 3*NSIDE - 1

C_l = readClFromFITS("Planck_TTTEEE_bestfit_Cl", Float64)
ClMatr = readdlm("PSchain$NSIDE.txt", Float64)
nsamp, lmax = size(ClMatr)
DlMatr = zeros(Float64, nsamp, lmax)
Dl_up = zeros(Float64, lmax)
Dl_lo = zeros(Float64, lmax)
Dl_avg = zeros(Float64, lmax)
for s in 1:nsamp
    DlMatr[s,:] = cl2dl(ClMatr[s,:], 0)
end

for li in 1:lmax
    m = mean(DlMatr[:,li])
    s = stdm(DlMatr[:,li], m)
    Dl_avg[li] = m
    Dl_lo[li] = m - 2*s
    Dl_up[li] = m + 2*s
end

plot(Dl_lo[3:lmax],
    fillrange = Dl_up[3:lmax],
    linealpha = 0.0,
    fillalpha = 0.6,
    c="purple",
    ylims=(0,7000),
    label="2σ Confidence Interval",
    xlabel= L"\ell",
    ylabel=L"C_\ell \, \ell(\ell + 1) / 2 \pi",
    title="Sampled CMB TT Power Spectrum")
plot!(Dl_lo[3:lmax], c="purple", label = "")
plot!(Dl_up[3:lmax], c="purple", label = "")
plot!(cl2dl(C_l[1:lmax], 0)[3:lmax], lw = 2, c="darkblue", label = "ΛCDM best-fit")
savefig("SampledPS.pdf")

#Maybe 2D HISTOGRAM!
