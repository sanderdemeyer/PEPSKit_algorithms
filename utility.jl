using TensorKit, PEPSKit, KrylovKit, OptimKit

function get_gate(dτ, Js)
    (Jx, Jy, Jz) = Js
    physical_space = ComplexSpace(2)
    lattice = fill(physical_space, 1, 1)
    T = ComplexF64
    σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
    σy = TensorMap(T[0 im; -im 0], physical_space, physical_space)
    σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
    H = (Jx * σx ⊗ σx) + (Jy * σy ⊗ σy) + (Jz * σz ⊗ σz)
    return exp(-dτ*H)
end
