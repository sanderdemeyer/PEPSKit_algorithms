using JLD2
include("utility.jl")
include("simple_update.jl")

dτ = 1e-4
D = 2
χenv = 4

lattice_size = 2
max_iterations = 20

t = 1.0
U = 0.0

H, twosite = get_operators_Hubbard(t, lattice_size)

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=Arnoldi(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

I, pspace = ASymSpace()
vspace = Vect[I]((0) => D/2, (1) => D/2)
vspace_env = Vect[I]((0) => χenv/2, (1) => χenv/2)
Pspaces = fill(pspace, lattice_size, lattice_size)
Nspaces = Espaces = fill(vspace, lattice_size, lattice_size)
psi = normalize(InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces))

(psi, lambdas) = simple_update(psi, twosite, dτ, D, max_iterations, ctm_alg; translate = true, χenv = χenv, printing_freq = 5);

file = jldopen("SU_Hubbard_t_$(t)_D_$(D)_chienv_$(χenv).jld2", "w")
file["psi"] = copy(psi)
file["lambdas"] = lambdas
energy = get_energy(deepcopy(psi), H, ctm_alg, vspace_env)
println("Energy after simple update is $(energy)")
file["energy"] = energy
close(file)
