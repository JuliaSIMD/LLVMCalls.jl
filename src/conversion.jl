function convert_func(op, T1, W1, T2, W2 = W1)
  typ1 = LLVM_TYPES[T1]
  typ2 = LLVM_TYPES[T2]
  vtyp1 = vtype(W1, typ1)
  vtyp2 = vtype(W2, typ2)
  instrs = """
      %res = $op $vtyp2 %0 to $vtyp1
      ret $vtyp1 %res
      """
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$W1,$T1}, Tuple{_Vec{$W2,$T2}}, data(v)))
  end
end
# For bitcasting between signed and unsigned integers (LLVM does not draw a distinction, but they're separate in Julia)
function identity_func(W, T1, T2)
  vtyp1 = vtype(W, LLVM_TYPES[T1])
  instrs = """
      ret $vtyp1 %0
      """
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$W,$T1}, Tuple{_Vec{$W,$T2}}, data(v)))
  end
end

### `Base.convert(::Type{<:AbstractSIMDVector}, x)` methods
### These are the critical `Base.convert` methods; scalar and `VecUnroll` are implemented with respect to them.
@generated function Base.convert(::Type{Vec{W,F}}, v::Vec{W,T}) where {W,F<:Union{Float32,Float64},T<:IntegerTypesHW}
  convert_func(T <: Signed ? "sitofp" : "uitofp", F, W, T)
end

@generated function Base.convert(::Type{Vec{W,T}}, v::Vec{W,F}) where {W,F<:Union{Float32,Float64},T<:IntegerTypesHW}
  convert_func(T <: Signed ? "fptosi" : "fptoui", T, W, F)
end
@generated function Base.convert(::Type{Vec{W,T1}}, v::Vec{W,T2}) where {W,T1<:IntegerTypesHW,T2<:IntegerTypesHW}
  sz1 = sizeof(T1)::Int; sz2 = sizeof(T2)::Int
  if sz1 < sz2
    convert_func("trunc", T1, W, T2)
  elseif sz1 == sz2
    identity_func(W, T1, T2)
  else
    convert_func(((T1 <: Signed) && (T2 <: Signed)) ? "sext" : "zext", T1, W, T2)
  end
end
@generated function Base.convert(::Type{Vec{W,Float32}}, v::Vec{W,Float64}) where {W}
  convert_func("fptrunc", Float32, W, Float64, W)
end
@generated function Base.convert(::Type{Vec{W,Float64}}, v::Vec{W,Float32}) where {W}
  convert_func("fpext", Float64, W, Float32, W)
end
@inline Base.convert(::Type{<:AbstractMask{W}}, v::Vec{W,Bool}) where {W} = tomask(v)
@inline Base.convert(::Type{M}, v::Vec{W,Bool}) where {W,U,M<:AbstractMask{W,U}} = tomask(v)
@inline Base.convert(::Type{<:AbstractMask{W,U} where U}, v::Vec{W,Bool}) where {W} = tomask(v)
@inline Base.convert(::Type{<:AbstractMask{L,U} where {L,U}}, v::Vec{W,Bool}) where {W} = tomask(v)
# @inline Base.convert(::Type{Mask}, v::Vec{W,Bool}) where {W} = tomask(v)
# @generated function Base.convert(::Type{<:AbstractMask{W}}, v::Vec{W,Bool}) where {W}
#     instrs = String[]
#     push!(instrs, "%m = trunc <$W x i8> %0 to <$W x i1>")
#     zext_mask!(instrs, 'm', W, '0')
#     push!(instrs, "ret i$(max(8,W)) %res.0")
#     U = mask_type_symbol(W);
#     quote
#         $(Expr(:meta,:inline))
#         Mask{$W}($LLVMCALL($(join(instrs, "\n")), $U, Tuple{_Vec{$W,Bool}}, data(v)))
#     end
# end
@inline Base.convert(::Type{Vec{W,Bit}}, v::Vec{W,Bool}) where {W,Bool} = tomask(v)

@inline Base.convert(::Type{Vec{W,T}}, v::Vec{W,T}) where {W,T<:IntegerTypesHW} = v
@inline Base.convert(::Type{Vec{W,T}}, v::Vec{W,T}) where {W,T} = v
@inline Base.convert(::Type{Vec{W,T}}, s::NativeTypes) where {W,T} = vbroadcast(Val{W}(), T(s))
@inline Base.convert(::Type{Vec{W,T}}, s::IntegerTypesHW) where {W,T<:IntegerTypesHW} = _vbroadcast(StaticInt{W}(), s % T, StaticInt{W}() * static_sizeof(T))
@inline Base.convert(::Type{V}, u::VecUnroll) where {V<:AbstractSIMDVector} = VecUnroll(fmap(Base.convert, V, getfield(u, :data)))
@inline Base.convert(::Type{V}, u::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMDVector} = u


@inline Base.convert(::Type{<:AbstractSIMDVector{W,T}}, i::MM{W,X}) where {W,X,T} = vrangeincr(Val{W}(), T(data(i)), Val{0}(), Val{X}())
@inline Base.convert(::Type{MM{W,X,T}}, i::MM{W,X}) where {W,X,T} = MM{W,X}(T(getfield(i, :i)))

@inline Base.convert(::Type{V}, v::AbstractMask{W}) where {W, T <: NativeTypesExceptBit, V <: AbstractSIMDVector{W,T}} = ifelse(v, one(T), zero(T))
@inline Base.convert(::Type{V}, v::AbstractMask{W}) where {W, V <: AbstractSIMDVector{W,Bit}} = v
@inline Base.convert(::Type{V}, v::Vec{W,Bool}) where {W, T <: Base.HWReal, V <: AbstractSIMDVector{W,T}} = ifelse(v, one(T), zero(T))


### `Base.convert(::Type{<:NativeTypes}, x)` methods. These forward to `Base.convert(::Type{Vec{W,T}}, x)`
@inline Base.convert(::Type{T}, v::AbstractSIMD{W,T}) where {T<:NativeTypes,W} = v
@inline Base.convert(::Type{T}, v::AbstractSIMD{W,S}) where {T<:NativeTypes,S,W} = convert(Vec{W,T}, v)

### `Base.convert(::Type{<:VecUnroll}, x)` methods
@inline Base.convert(::Type{VecUnroll{N,W,T,V}}, s::NativeTypes) where {N,W,T,V} = VecUnroll{N}(convert(V, s))
@inline Base.convert(::Type{VecUnroll{N,W,T,V}}, v::AbstractSIMDVector{W}) where {N,W,T,V} = VecUnroll{N}(convert(V, v))
@inline Base.convert(::Type{VecUnroll{N,W,T,V}}, v::VecUnroll{N}) where {N,W,T,V} = VecUnroll(fmap(convert, V, getfield(v, :data)))
@inline Base.convert(::Type{VecUnroll{N,W,T,V}}, v::VecUnroll{N,W,T,V}) where {N,W,T,V} = v
@generated function Base.convert(::Type{VecUnroll{N,1,T,T}}, s::NativeTypes) where {N,T}
  quote
    $(Expr(:meta,:inline))
    x = convert($T, s)
    VecUnroll((Base.Cartesian.@ntuple $(N+1) n -> x))
  end
end

# @inline Base.convert(::Type{T}, v::T) where {T} = v


@generated function splitvectortotuple(::StaticInt{N}, ::StaticInt{W}, v::AbstractMask{L}) where {N,W,L}
  N*W == L || throw(ArgumentError("Can't split a vector of length $L into $N pieces of length $W."))
  t = Expr(:tuple, :(Mask{$W}(u)))
  s = 0
  for n ∈ 2:N
    push!(t.args, :(Mask{$W}(u >>> $(s += W))))
  end
  # This `Base.convert` will dispatch to one of the following two `Base.convert` methods
  Expr(:block, Expr(:meta,:inline), :(u = data(v)), t)
end
@generated function splitvectortotuple(::StaticInt{N}, ::StaticInt{W}, v::AbstractSIMDVector{L}) where {N,W,L}
  N*W == L || throw(ArgumentError("Can't split a vector of length $L into $N pieces of length $W."))
  t = Expr(:tuple);
  j = 0
  for i ∈ 1:N
    val = Expr(:tuple)
    for w ∈ 1:W
      push!(val.args, j)
      j += 1
    end
    push!(t.args, :(shufflevector(v, Val{$val}())))
  end
  Expr(:block, Expr(:meta,:inline), t)
end
@generated function splitvectortotuple(::StaticInt{N}, ::StaticInt{W}, v::LazyMulAdd{M,O}) where {N,W,M,O}
  # LazyMulAdd{M,O}(splitvectortotuple(StaticInt{N}(), StaticInt{W}(), getfield(v, :data)))
  t = Expr(:tuple)
  for n ∈ 1:N
    push!(t.args, :(LazyMulAdd{$M,$O}(splitdata[$n])))
  end
  Expr(:block, Expr(:meta,:inline), :(splitdata = splitvectortotuple(StaticInt{$N}(), StaticInt{$W}(), getfield(v, :data))), t)
end

@generated function Base.convert(::Type{VecUnroll{N, W, T, V}}, v::AbstractSIMDVector{L}) where {N, W, T, V, L}
  if W == L # _vconvert will dispatch to one of the two above
    Expr(:block, Expr(:meta,:inline), :(convert(VecUnroll{$N,$W,$T,$V}, v)))
  else
    Expr(:block, Expr(:meta,:inline), :(convert(VecUnroll{$N,$W,$T,$V}, VecUnroll(splitvectortotuple(StaticInt{$(N+1)}(), StaticInt{$W}(), v)))))
  end
end

@inline Vec{W,T}(v::Vec{W,S}) where {W,T,S} = Base.convert(Vec{W,T}, v)
@inline Vec{W,T}(v::S) where {W,T,S<:NativeTypes} = Base.convert(Vec{W,T}, v)


@inline vsigned(v::AbstractSIMD{W,T}) where {W,T <: Base.BitInteger} = v % signed(T)
@inline vunsigned(v::AbstractSIMD{W,T}) where {W,T <: Base.BitInteger} = v % unsigned(T)

@generated function _vfloat(v::Vec{W,I}, ::StaticInt{RS}) where {W, I <: Integer, RS}
  ex = if 8W ≤ RS
    :(Base.convert(Vec{$W,Float64}, v))
  else
    :(Base.convert(Vec{$W,Float32}, v))
  end
  Expr(:block, Expr(:meta, :inline), ex)
end
@inline Base.float(v::Vec{W,I}) where {W, I <: Integer} = _vfloat(v, register_size())
@inline Base.float(v::AbstractSIMD{W,T}) where {W,T <: Union{Float32,Float64}} = v
@inline Base.float(vu::VecUnroll) = VecUnroll(fmap(vfloat, getfield(vu, :data)))
@inline Base.float(x::Union{Float32,Float64}) = x
@inline Base.float(x::UInt64) = Base.uitofp(Float64, x)
@inline Base.float(x::Int64) = Base.sitofp(Float64, x)
@inline Base.float(x::Union{UInt8,UInt16,UInt32}) = Base.uitofp(Float32, x)
@inline Base.float(x::Union{Int8,Int16,Int32}) = Base.sitofp(Float32, x)
# @inline Base.float(v::Vec{W,I}) where {W, I <: Union{UInt64, Int64}} = Vec{W,Float64}(v)


@inline vfloat_fast(v::AbstractSIMDVector{W,T}) where {W,T <: Union{Float32,Float64}} = v
@inline vfloat_fast(vu::VecUnroll{W,T}) where {W,T<:Union{Float32,Float64}} = vu
@inline vfloat_fast(vu::VecUnroll) = VecUnroll(fmap(vfloat_fast, getfield(vu, :data)))

@generated function __vfloat_fast(v::Vec{W,I}, ::StaticInt{RS}) where {W, I <: Integer, RS}
  arg = if (2W*sizeof(I) ≤ RS) || sizeof(I) ≤ 4
    :v
  elseif I <: Signed
    :(v % Int32)
  else
    :(v % UInt32)
  end
  ex = if 8W ≤ RS
    :(Vec{$W,Float64}($arg))
  else
    :(Vec{$W,Float32}($arg))
  end
  Expr(:block, Expr(:meta, :inline), ex)
end
@inline _vfloat_fast(v, ::False) = __vfloat_fast(v, register_size())
@inline _vfloat_fast(v, ::True) = float(v)


@inline Base.reinterpret(::Type{Vec{1,T}}, x::S) where {T,S<:NativeTypes} = reinterpret(T,x)
@inline vrem(x::NativeTypes, ::Type{T}) where {T} = x % T
@generated function Base.reinterpret(::Type{T1}, v::Vec{W2,T2}) where {W2, T1 <: NativeTypes, T2}
  W1 = W2 * sizeof(T2) ÷ sizeof(T1)
  Expr(:block, Expr(:meta, :inline), :(reinterpret(Vec{$W1,$T1}, v)))
end
@inline Base.reinterpret(::Type{Vec{1,T1}}, v::Vec{W,T2}) where {W,T1,T2<:Base.BitInteger} = reinterpret(T1, fuseint(v))
@generated function Base.reinterpret(::Type{Vec{W1,T1}}, v::Vec{W2,T2}) where {W1, W2, T1, T2}
  @assert sizeof(T1) * W1 == W2 * sizeof(T2)
  convert_func("bitcast", T1, W1, T2, W2)
end

@inline vunsafe_trunc(::Type{I}, v::Vec{W,T}) where {W,I,T} = convert(Vec{W,I}, v)
@inline vrem(v::AbstractSIMDVector{W,T}, ::Type{I}) where {W,I,T} = convert(Vec{W,I}, v)
@inline vrem(v::AbstractSIMDVector{W,T}, ::Type{V}) where {W,I,T,V<:AbstractSIMD{W,I}} = convert(V, v)
@inline vrem(r::IntegerTypesHW, ::Type{V}) where {W, I, V <: AbstractSIMD{W,I}} = convert(V, r % I)
