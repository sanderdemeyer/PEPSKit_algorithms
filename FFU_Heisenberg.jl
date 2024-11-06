using JLD2
include("utility.jl")
include("FFU_ab.jl")

dτ = 1e-4
D = 4
χenv = 32

lattice_size = 2
max_iterations = 10

Js = (-1, 1, -1)

H, twosite = get_operators_Heisenberg(Js, lattice_size)

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=Arnoldi(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

psi = normalize(InfinitePEPS(2, D; unitcell=(lattice_size,lattice_size)))
vspace_env = ComplexSpace(χenv)

FFU(psi, twosite, dτ, max_iterations, ctm_alg, χenv = χenv)

println("Well Done!")

# file = jldopen("SU_Heisenberg_D_$(D)_chienv_$(χenv).jld2", "w")
# file["psi"] = copy(psi)
# file["lambdas"] = lambdas
# energy = get_energy(deepcopy(psi), H, ctm_alg, vspace_env)
# println("Energy after simple update is $(energy)")
# file["energy"] = energy
# close(file)
