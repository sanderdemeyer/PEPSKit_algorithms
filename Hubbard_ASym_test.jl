using LinearAlgebra
using Random
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit
using MPSKit
using DelimitedFiles

include("Hubbard_tensors.jl")
include("spin_tensors.jl")

function iterate(psi_init, h, opt_alg, env_init, maxiter)
    io = open("check_energies_Hubbard_ASym.csv", "w")
    result = fixedpoint(psi_init, h, opt_alg, env_init)
    println("E = $(result.E), grad = $(result.grad)")
    writedlm(io, [1 result.E norm(result.grad)])
    file = jldopen("Hubbard_ASym_test_t_1_U_0.jld2", "w")
    file["grad"] = copy(result.grad)
    file["E"] = result.E
    file["psi"] = result.peps
    file["grad"] = result.grad
    file["norm_grad"] = norm(result.grad)
    close(file)

    for i = 2:maxiter
        result = fixedpoint(result.peps, h, opt_alg, result.env)
        println("E = $(result.E), grad = $(result.grad)")
        writedlm(io, [i result.E norm(result.grad)])
        file = jldopen("Hubbard_ASym_test_t_1_U_0.jld2", "w")
        file["grad"] = copy(result.grad)
        file["E"] = result.E
        file["psi"] = result.peps
        file["grad"] = result.grad
        file["norm_grad"] = norm(result.grad)
        close(file)
    end
    close(io)
end

t = 1

T = ComplexF64

lattice_size = 1

# P = 1
# Q = 1
# spin = true
# Isym, pspacesym = SymSpace(P, Q, spin)
# c⁺c⁻sym, nsym = Hopping(P, Q, spin)
# onsitesym = U*OSInteraction(P, Q, spin)

Ps = Vect[fℤ₂]((0) => 1, (1) => 1)
I = fℤ₂
Vodd = Vect[I]((1) => 1)

lattice = fill(Ps, lattice_size, lattice_size)
c⁺u = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vodd)
blocks(c⁺u)[I((0))] .= 1

c⁺d = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vodd)
blocks(c⁺d)[I((0))] .= 1 # [0 0; -1 0]

cu = TensorMap(zeros, ComplexF64, Vodd ⊗ Ps ← Ps)
blocks(cu)[I((0))] .= 1

cd = TensorMap(zeros, ComplexF64, Vodd ⊗ Ps ← Ps)
blocks(cd)[I((0))] .= 1 # [0 -1; 0 0]

@planar twosite_up[-1 -2; -3 -4] := c⁺u[-1; -3 1] * cu[1 -2; -4]
@planar twosite_down[-1 -2; -3 -4] := c⁺d[-1; -3 1] * cd[1 -2; -4]
twosite = twosite_up + twosite_down
twosite = -(twosite + twosite')

h = nearest_neighbour_hamiltonian(lattice, twosite)

D = 2
χ = 2

vspace = Vect[I]((0) => D/2, (1) => D/2)
vspace_env = Vect[I]((0) => χ/2, (1) => χ/2)

Pspaces = fill(pspace, lattice_size, lattice_size)
Nspaces = Espaces = fill(vspace, lattice_size, lattice_size)

psi_init = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
env0 = CTMRGEnv(psi_init, vspace_env)

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=Arnoldi(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=1, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)

# env_init = leading_boundary(env0, psi_init, ctm_alg);

maxiter = 100
result = iterate(psi_init, h, opt_alg, env0, maxiter)
# result = iterate(psi_init, h, opt_alg, env_init, maxiter)
