
function binary_op(op, W, @nospecialize(T))
  ty = LLVM_TYPES[T]
  if isone(W)
    V = T
  else
    ty = "<$W x $ty>"
    V = NTuple{W,VecElement{T}}
  end
  instrs = "%res = $op $ty %0, %1\nret $ty %res"
  call = :($LLVMCALL($instrs, $V, Tuple{$V,$V}, data(v1), data(v2)))
  W > 1 && (call = Expr(:call, :Vec, call))
  Expr(:block, Expr(:meta, :inline), call)
end

# Integer
for (op,f) ∈ [("add",:+),("sub",:-),("mul",:*),("shl",:<<)]
  ff = Symbol('v', op)
  fnsw = Symbol(ff,"_nsw")
  fnuw = Symbol(ff,"_nuw")
  fnw = Symbol(ff,"_nw")
  ff_fast = if op == "shl"
    @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($op, W, T)
  else
    ff_fast = :(Base.FastMath.$(Symbol(op * "_fast")))
    @eval begin
      @generated $ff_fast(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($op, W, T)
      @inline Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = $ff_fast(v1, v2)
    end
  end
  @eval begin
    @generated $fnsw(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($(op * " nsw"), W, T)
    @generated $fnuw(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($(op * " nuw"), W, T)
    @generated $fnw(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($(op * " nsw nuw"), W, T)
    @generated $fnsw(v1::T, v2::T) where {T<:IntegerTypesHW} = binary_op($(op * " nsw"), 1, T)
    @generated $fnuw(v1::T, v2::T) where {T<:IntegerTypesHW} = binary_op($(op * " nuw"), 1, T)
    @generated $fnw(v1::T, v2::T) where {T<:IntegerTypesHW} = binary_op($(op * " nsw nuw"), 1, T)
  end
end
for (op,f) ∈ [("div",:÷),("rem",:%)]
  @eval begin
    @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, W, T)
  end
end
# for (op,f) ∈ [("div",:÷),("rem",:%)]
#   @eval begin
#     @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, W, T)
#     # @generated Base.$f(v1::T, v2::T) where {T<:IntegerTypesHW} = binary_op((T <: Signed ? 's' : 'u') * $op, 1, T)
#   end
# end
# @inline vcld(x, y) = vadd(vdiv(vsub(x, one(x)), y), one(x))
@inline div_fast(x::T, y::T) where {T <: SignedHW} = Base.sdiv_int(x,y)
@inline div_fast(x::T, y::T) where {T <: UnsignedHW} = Base.udiv_int(x,y)
@inline div_fast(x::AbstractSIMD, y::AbstractSIMD) = x ÷ y
@inline vcld(x, y) = (div_fast((x - one(x)), y) + one(x))
@inline function divrem_fast(x, y)
  d = div_fast(x, y)
  r = x - (d * y)
  d, r
end
for (op,sub) ∈ [
  ("ashr",:SignedHW),
  ("lshr",:UnsignedHW),
  ("lshr",:IntegerTypesHW),
  ("and",:IntegerTypesHW),
  ("or",:IntegerTypesHW),
  ("xor",:IntegerTypesHW)
  ]
  ff = sub === :UnsignedHW ? :vashr : Symbol('v', op)
  @eval begin
    @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:$sub}  = binary_op($op, W, T)
    @generated $ff(v1::T, v2::T) where {T<:$sub}  = binary_op($op, 1, T)
  end
end

for (op,f,s) ∈ [("ashr",:>>,0x01),("lshr",:>>,0x02),("lshr",:>>>,0x03),("and",:&,0x03),("or",:|,0x03),("xor",:⊻,0x03)]
  fdef = Expr(:where, :(Base.$f(v1::Vec{W,T}, v2::Vec{W,T})), :W)
  if iszero(s & 0x01)
    @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:UnsignedHW} = binary_op($op, W1, T, W2)
  elseif iszero(s & 0x02)
    @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:SignedHW} = binary_op($op, W1, T, W2)
  else
    @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($op, W1, T, W2)
  end
  if op === "ashr" # replace first iteration with <<
    f = :(<<)
  end
  @eval begin
    @inline Base.$f(v::Vec, i::Integer) = ((x, y) = promote(v, i); $f(x, y))
    @inline Base.$f(i::Integer, v::Vec) = ((x, y) = promote(i, v); $f(x, y))
  end
end

for (op,f,ff) ∈ [("fadd",:(+),:add_fast),("fsub",:(-),:sub_fast),("fmul",:(*),:mul_fast),("fdiv",:(/),:div_fast),("frem",:(%),:rem_fast)]

  fop_fast = f === :(/) ? "fdiv fast" : op * ' ' * fast_flags(true)
  fop_contract = op * ' ' * fast_flags(false)
  @eval begin
    @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = binary_op($fop_contract, W, T)
    @generated Base.FastMath.$ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = binary_op($fop_fast, W, T)
  end
end
# @inline vsub(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.sub_float(a,b)
# @inline vadd(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.add_float(a,b)
# @inline vmul(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.mul_float(a,b)
# @inline vsub_fast(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.sub_float_fast(a,b)
# @inline vadd_fast(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.add_float_fast(a,b)
# @inline vmul_fast(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.mul_float_fast(a,b)

@inline Base.div(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:FloatingTypes} = floor(signed(Base.uinttype(T)), Base.FastMath.div_fast(v1, v2))
# @inline Base.div_fast(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:FloatingTypes} = vfdiv_fast(vsub_fast(v1, vrem_fast(v1, v2)), v2)
# @inline vrem_fast(a,b) = a % b
# @inline vdiv_fast(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:IntegerTypesHW} = trunc(T, vfloat_fast(v1) / vfloat_fast(v2))
# @inline function vdiv_fast(v1, v2)
#   v3, v4 = promote_div(v1, v2)
#   vdiv_fast(v3, v4)
# end

@inline vfdiv(a::AbstractSIMDVector{W}, b::AbstractSIMDVector{W}) where {W} = vfdiv(vfloat(a), vfloat(b))
@inline vfdiv_fast(a::AbstractSIMDVector{W}, b::AbstractSIMDVector{W}) where {W} = vfdiv_fast(vfloat_fast(a), vfloat_fast(b))
@inline vfdiv(a, b) = a / b
@inline vfdiv_fast(a, b) = Base.FastMath.div_fast(a, b)

# for f ∈ [:vadd,:vsub,:vmul]
#   for s ∈ [Symbol(""),:_fast,:_nsw,:_nuw,:_nw]
#     fs = Symbol(f,s)
#     @eval begin
#       @inline function $fs(a::Union{FloatingTypes,IntegerTypesHW,AbstractSIMD}, b::Union{FloatingTypes,IntegerTypesHW,AbstractSIMD})
#         c, d = promote(a, b)
#         $fs(c, d)
#       end
#     end
#   end
# end
# @inline vsub(a::T, b::T) where {T<:Base.BitInteger} = Base.sub_int(a, b)
# for (vf,bf) ∈ [
#   (:vadd,:add_int),(:vsub,:sub_int),(:vmul,:mul_int),
#   (:vadd_fast,:add_int),(:vsub_fast,:sub_int),(:vmul_fast,:mul_int),
#   (:vadd_nsw,:add_int),(:vsub_nsw,:sub_int),(:vmul_nsw,:mul_int),
#   (:vadd_nuw,:add_int),(:vsub_nuw,:sub_int),(:vmul_nuw,:mul_int),
#   (:vadd_nw,:add_int),(:vsub_nw,:sub_int),(:vmul_nw,:mul_int),
#   ]
#   @eval begin
#     @inline $vf(a::Int128, b::Int128) = Base.$bf(a, b)
#     @inline $vf(a::UInt128, b::UInt128) = Base.$bf(a, b)
#   end
# end
# @inline vrem(a::Float32, b::Float32) = Base.rem_float_fast(a, b)
# @inline vrem(a::Float64, b::Float64) = Base.rem_float_fast(a, b)

