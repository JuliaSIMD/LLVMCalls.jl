module LLVMCalls

using Static: Zero, One, True, False, StaticInt, StaticBool, lt, gt
import IfElse: ifelse

using Base: HWReal

export Vec, VecUnroll, MM, Mask

const LLVMCALL = GlobalRef(Base, :llvmcall)
const FloatingTypes = Union{Float32, Float64} # Float16

# const SignedHW = Union{Int8,Int16,Int32,Int64,Int128}
# const UnsignedHW = Union{UInt8,UInt16,UInt32,UInt64,UInt128}
const SignedHW = Union{Int8,Int16,Int32,Int64}
const UnsignedHW = Union{UInt8,UInt16,UInt32,UInt64}
const IntegerTypesHW = Union{SignedHW,UnsignedHW}
const IntegerTypes = Union{StaticInt,IntegerTypesHW}

struct Bit; data::Bool; end # Dummy for Ptr
const Boolean = Union{Bit,Bool}
const NativeTypesExceptBit = Union{Bool,HWReal}
const NativeTypes = Union{NativeTypesExceptBit, Bit}

const _Vec{W,T<:Number} = NTuple{W,Core.VecElement{T}}

abstract type AbstractSIMD{W,T <: Union{<:StaticInt,NativeTypes}} <: Real end
abstract type AbstractSIMDVector{W,T} <: AbstractSIMD{W,T} end
struct VecUnroll{N,W,T,V<:Union{NativeTypes,AbstractSIMD{W,T}}} <: AbstractSIMD{W,T}
  data::Tuple{V,Vararg{V,N}}
  @inline (VecUnroll(data::Tuple{V,Vararg{V,N}})::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMD{W,T}} = new{N,W,T,V}(data)
  @inline (VecUnroll(data::Tuple{T,Vararg{T,N}})::VecUnroll{N,T,T}) where {N,T<:NativeTypes} = new{N,1,T,T}(data)
end
# @inline VecUnroll(data::Tuple) = VecUnroll(promote(data))
const VecOrScalar = Union{AbstractSIMDVector,NativeTypes}
const NativeTypesV = Union{AbstractSIMD,NativeTypes,StaticInt}
const IntegerTypesV = Union{AbstractSIMD{<:Any,<:IntegerTypes},IntegerTypesHW}
struct Vec{W,T} <: AbstractSIMDVector{W,T}
  data::NTuple{W,Core.VecElement{T}}
  @inline Vec{W,T}(x::NTuple{W,Core.VecElement{T}}) where {W,T<:NativeTypes} = new{W,T}(x)
  @generated function Vec(x::Tuple{Core.VecElement{T},Vararg{Core.VecElement{T},_W}}) where {_W,T<:NativeTypes}
    W = _W + 1
    vtyp = Expr(:curly, :Vec, W, T)
    Expr(:block, Expr(:meta,:inline), Expr(:(::), Expr(:call, vtyp, :x), vtyp))
  end
end

@generated mask_type(::StaticInt{W}) where {W} = Symbol(:UInt, clamp(nextpow2(W), 8, 128))
@inline data(vu::VecUnroll) = getfield(vu, :data)
abstract type AbstractMask{W,U<:Union{UnsignedHW,UInt128}} <: AbstractSIMDVector{W,Bit} end
struct Mask{W,U} <: AbstractMask{W,U}
  u::U
  @inline function Mask{W,U}(u::Unsigned) where {W,U} # ignores U...
    U2 = mask_type(StaticInt{W}())
    new{W,U2}(u % U2)
  end
end
struct EVLMask{W,U} <: AbstractMask{W,U}
  u::U
  evl::UInt32
  @inline function EVLMask{W,U}(u::Unsigned, evl) where {W,U} # ignores U...
    U2 = mask_type(StaticInt{W}())
    new{W,U2}(u % U2, evl % UInt32)
  end
end
const AnyMask{W} = Union{AbstractMask{W},VecUnroll{<:Any,W,Bit,<:AbstractMask{W}}}
@inline Mask{W}(u::U) where {W,U<:Unsigned} = Mask{W,U}(u)
@inline EVLMask{W}(u::U, i) where {W,U<:Unsigned} = EVLMask{W,U}(u, i)
@inline Mask{1}(b::Bool) = b
@inline EVLMask{1}(b::Bool, i) = b
@inline Mask(m::EVLMask{W,U}) where {W,U} = Mask{W,U}(getfield(m,:u))
# Const prop is good enough; added an @inferred test to make sure.
# Removed because confusion can cause more harm than good.

@inline Base.broadcastable(v::AbstractSIMDVector) = Ref(v)

Vec{W,T}(x::Vararg{NativeTypes,W}) where {W,T<:NativeTypes} = Vec(ntuple(w -> Core.VecElement{T}(x[w]), Val{W}()))
Vec{1,T}(x::Union{Float32,Float64}) where {T<:NativeTypes} = T(x)
Vec{1,T}(x::Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Bool}) where {T<:NativeTypes} = T(x)

@inline Base.length(::AbstractSIMDVector{W}) where W = W
@inline Base.size(::AbstractSIMDVector{W}) where W = (W,)
@inline Base.eltype(::AbstractSIMD{W,T}) where {W,T} = T
@inline Base.conj(v::AbstractSIMDVector) = v # so that things like dot products work.
@inline Base.adjoint(v::AbstractSIMDVector) = v # so that things like dot products work.
@inline Base.transpose(v::AbstractSIMDVector) = v # so that things like dot products work.

# Not using getindex/setindex as names to emphasize that these are generally treated as single objects, not collections.
@generated function extractelement(v::Vec{W,T}, i::I) where {W,I <: IntegerTypesHW,T}
  typ = LLVM_TYPES[T]
  instrs = """
          %res = extractelement <$W x $typ> %0, i$(8sizeof(I)) %1
          ret $typ %res
      """
  call = :($LLVMCALL($instrs, $T, Tuple{_Vec{$W,$T},$I}, data(v), i))
  Expr(:block, Expr(:meta, :inline), call)
end
@generated function insertelement(v::Vec{W,T}, x::T, i::I) where {W,I <: IntegerTypesHW,T}
  typ = LLVM_TYPES[T]
  instrs = """
          %res = insertelement <$W x $typ> %0, $typ %1, i$(8sizeof(I)) %2
          ret <$W x $typ> %res
      """
  call = :(Vec($LLVMCALL($instrs, _Vec{$W,$T}, Tuple{_Vec{$W,$T},$T,$I}, data(v), x, i)))
  Expr(:block, Expr(:meta, :inline), call)
end
@inline (v::AbstractSIMDVector)(i::IntegerTypesHW) = extractelement(v, i - one(i))
@inline (v::AbstractSIMDVector)(i::Integer) = extractelement(v, Int(i) - 1)
Base.@propagate_inbounds (vu::VecUnroll)(i::Integer, j::Integer) = getfield(vu, :data)[j](i)

@inline Base.Tuple(v::Vec{W}) where {W} = ntuple(v, Val{W}())

# Use with care in function signatures; try to avoid the `T` to stay clean on Test.detect_unbound_args

@inline Base.copy(v::AbstractSIMDVector) = v
@inline data(v) = v
@inline data(v::Vec) = getfield(v, :data)

function Base.show(io::IO, v::AbstractSIMDVector{W,T}) where {W,T}
  name = typeof(v)
  print(io, "$(name)<")
  for w ∈ 1:W
    print(io, repr(extractelement(v, w-1)))
    w < W && print(io, ", ")
  end
  print(io, ">")
end
Base.bitstring(m::AbstractMask{W}) where {W} = bitstring(data(m))[end-W+1:end]
function Base.show(io::IO, m::AbstractMask{W}) where {W}
  bits = data(m)
  if m isa EVLMask
    print(io, "EVLMask{$W,Bit}<")
  else
    print(io, "Mask{$W,Bit}<")
  end
  for w ∈ 0:W-1
    print(io, bits & 1)
    bits >>= 1
    w < W-1 && print(io, ", ")
  end
  print(io, ">")
end
function Base.show(io::IO, vu::VecUnroll{N,W,T,V}) where {N,W,T,V}
  println(io, "$(N+1) x $V")
  d = data(vu)
  for n in 1:N+1
    show(io, d[n]);
    n > N || println(io)
  end
end

"""
  The name `MM` type refers to _MM registers such as `XMM`, `YMM`, and `ZMM`.
  `MMX` from the original MMX SIMD instruction set is a [meaningless initialism](https://en.wikipedia.org/wiki/MMX_(instruction_set)#Naming).

  The `MM{W,X}` type is used to represent SIMD indexes of width `W` with stride `X`.
  """
struct MM{W,X,I<:Union{HWReal,StaticInt}} <: AbstractSIMDVector{W,I}
  i::I
  @inline MM{W,X}(i::T) where {W,X,T<:Union{HWReal,StaticInt}} = new{W,X::Int,T}(i)
end
@inline MM(i::MM{W,X}) where {W,X} = MM{W,X}(getfield(i, :i))
@inline MM{W}(i::Union{HWReal,StaticInt}) where {W} = MM{W,1}(i)
@inline MM{W}(i::Union{HWReal,StaticInt}, ::StaticInt{X}) where {W,X} = MM{W,X}(i)
@inline data(i::MM) = getfield(i, :i)

@inline extractelement(i::MM{W,X,I}, j) where {W,X,I} = getfield(i, :i) + (X % I) * (j % I)

function Base.getproperty(::AbstractSIMD, ::Symbol)
  throw(ErrorException("""
  `Base.getproperty` not defined on AbstractSIMD.
  If you wish to work with the data as a tuple, it is recommended to use `Tuple(v)`. Once you have an ordinary tuple, you can access
  individual elements normally. Alternatively, you can index using parenthesis, e.g. `v(1)` indexes the first element.
  Parenthesis are used instead of `getindex`/square brackets because `AbstractSIMD` objects represent a single number, and
  for `x::Number`, `x[1] === x`.

  If you wish to perform a reduction on the collection, the naming convention is prepending the base function with a `v`. These functions
  are not overloaded, because for `x::Number`, `sum(x) === x`. Functions include `vsum`, `vprod`, `vmaximum`, `vminimum`, `vany`, and `vall`.

  If you wish to define a new operation applied to the entire vector, do not define it in terms of operations on the individual eleemnts.
  This will often lead to bad code generation -- bad in terms of both performance, and often silently producing incorrect results!
  Instead, implement them in terms of existing functions defined on `::AbstractSIMD`. Please feel free to file an issue if you would like
  clarification, and especially if you think the function may be useful for others and should be included in `VectorizationBase.jl`.
  """))
end

nextpow2(W::T) where {T<:Base.BitInteger} = one(T) << (T(8sizeof(T)) - leading_zeros(W - one(T)))

@generated function Vec(y::T, x::Vararg{T,_W}) where {T<:NativeTypes, _W}
  W = 1 + _W
  Wfull = nextpow2(W)
  ty = LLVM_TYPES[T]
  init = W == Wfull ? "undef" : "zeroinitializer"
  instrs = ["%v0 = insertelement <$Wfull x $ty> $init, $ty %0, i32 0"]
  Tup = Expr(:curly, :Tuple, T)
  for w ∈ 1:_W
    push!(instrs, "%v$w = insertelement <$Wfull x $ty> %v$(w-1), $ty %$w, i32 $w")
    push!(Tup.args, T)
  end
  push!(instrs, "ret <$Wfull x $ty> %v$_W")
  llvmc = :($LLVMCALL($(join(instrs,"\n")), _Vec{$Wfull,$T}, $Tup))
  push!(llvmc.args, :y)
  for w ∈ 1:_W
    push!(llvmc.args, Expr(:ref, :x, w))
  end
  quote
    $(Expr(:meta,:inline))
    Vec($llvmc)
  end
end
@inline reduce_to_onevec(f::F, vu::VecUnroll) where {F} = ArrayInterface.reduce_tup(f, data(vu))

include("static.jl")
include("llvm_types.jl")
include("lazymul.jl")
include("vector_width.jl")
include("ranges.jl")
include("binary_ops.jl")
include("conversion.jl")
include("masks.jl")
include("intrin_funcs.jl")
include("memory_addr.jl")
include("unary_ops.jl")
include("vbroadcast.jl")
include("vector_ops.jl")
include("nonbroadcastingops.jl")
include("integer_fma.jl")
include("fmap.jl")

end
