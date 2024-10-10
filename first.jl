using TensorKit, PEPSKit, KrylovKit, OptimKit

T = ComplexF64

Jx=-1
Jy=1
Jz=-1
unitcell = (1, 1)

physical_space = ComplexSpace(2)
lattice = fill(physical_space, 1, 1)
σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
σy = TensorMap(T[0 im; -im 0], physical_space, physical_space)
σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
H = (Jx * σx ⊗ σx) + (Jy * σy ⊗ σy) + (Jz * σz ⊗ σz)
H = repeat(nearest_neighbour_hamiltonian(lattice, H / 4), unitcell...)

# constructing the Hamiltonian:
H = square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1) # sublattice rotation to obtain single-site unit cell




# configuring the parameters
D = 1
chi = 4
ctm_alg = CTMRG(; tol=1e-10, miniter=2, maxiter=4, verbosity=1, trscheme=truncdim(chi))
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=4, gradtol=1e-4, verbosity=2),
    gradient_alg=LinSolver(),
    reuse_env=true,
)

# ground state search
state = InfinitePEPS(2, D)
ctm = leading_boundary(CTMRGEnv(state, ComplexSpace(chi)), state, ctm_alg)
result = fixedpoint(state, H, opt_alg, ctm)

@show result.E # -0.6625...