using JLD2
include("utility.jl")
include("simple_update.jl")

dτ = 1e-5
D = 2
χenv = 4

lattice_size = 2
max_iterations = 200

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

(psi, lambdas, energies) = simple_update(psi, twosite, dτ, D, max_iterations, ctm_alg; translate = true, χenv = χenv, printing_freq = 5);

# file = jldopen("SU_Heisenberg_D_$(D)_chienv_$(χenv).jld2", "w")
# file["psi"] = copy(psi)
# file["lambdas"] = lambdas
# energy = get_energy(deepcopy(psi), H, ctm_alg, vspace_env)
# println("Energy after simple update is $(energy)")
# file["energy"] = energy
# close(file)
