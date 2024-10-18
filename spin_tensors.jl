function S_x(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return if spin == 1 // 2
        TensorMap(T[0 1; 1 0], ℂ^2 ← ℂ^2)
    elseif spin == 1
        TensorMap(T[0 1 0; 1 0 1; 0 1 0], ℂ^3 ← ℂ^3) / sqrt(2)
    else
        throw(ArgumentError("spin $spin not supported"))
    end
end
function S_y(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return if spin == 1 // 2
        TensorMap(T[0 -im; im 0], ℂ^2 ← ℂ^2)
    elseif spin == 1
        TensorMap(T[0 -im 0; im 0 -im; 0 im 0], ℂ^3 ← ℂ^3) / sqrt(2)
    else
        throw(ArgumentError("spin $spin not supported"))
    end
end
function S_z(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return if spin == 1 // 2
        TensorMap(T[1 0; 0 -1], ℂ^2 ← ℂ^2)
    elseif spin == 1
        TensorMap(T[1 0 0; 0 0 0; 0 0 -1], ℂ^3 ← ℂ^3)
    else
        throw(ArgumentError("spin $spin not supported"))
    end
end
function S_xx(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return S_x(Trivial, T; spin) ⊗ S_x(Trivial, T; spin)
end
function S_yy(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return S_y(Trivial, T; spin) ⊗ S_y(Trivial, T; spin)
end
function S_zz(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return S_z(Trivial, T; spin) ⊗ S_z(Trivial, T; spin)
end
