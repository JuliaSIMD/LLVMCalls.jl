
# nextpow2(W) = vshl(one(W), vsub_fast(8sizeof(W), leading_zeros(vsub_fast(W, one(W)))))



# @inline _pick_vector(::StaticInt{W}, ::Type{T}) where {W,T} = Vec{W,T}
# @inline pick_vector(::Type{T}) where {T} = _pick_vector(pick_vector_width(T), T)
# @inline function pick_vector(::Val{N}, ::Type{T}) where {N, T}
#     _pick_vector(smin(nextpow2(StaticInt{N}()), pick_vector_width(T)), T)
# end
# pick_vector(N::Int, ::Type{T}) where {T} = pick_vector(Val(N), T)

@inline MM(::Union{Val{W},StaticInt{W}}) where {W} = MM{W}(0)
@inline MM(::Union{Val{W},StaticInt{W}}, i) where {W} = MM{W}(i)
@inline MM(::Union{Val{W},StaticInt{W}}, i::AbstractSIMDVector{W}) where {W} = i
@inline MM(::StaticInt{W}, i, ::StaticInt{X}) where {W,X} = MM{W,X}(i)
@inline gep(ptr::Ptr, i::MM) = gep(ptr, data(i))

@inline Base.one(::Type{MM{W,X,I}}) where {W,X,I} = one(I)
@inline staticm1(i::MM{W,X,I}) where {W,X,I} = MM{W,X}(FastMath.sub_fast(data(i), one(I)))
@inline staticp1(i::MM{W,X,I}) where {W,X,I} = MM{W,X}(vadd_nsw(data(i), one(I)))
@inline FastMath.add_fast(i::MM{W,X}, j::IntegerTypesHW) where {W,X} = MM{W,X}(FastMath.add_fast(data(i), j))
@inline FastMath.add_fast(i::IntegerTypesHW, j::MM{W,X}) where {W,X} = MM{W,X}(FastMath.add_fast(i, data(j)))
@inline FastMath.sub_fast(i::MM{W,X}, j::IntegerTypesHW) where {W,X} = MM{W,X}(FastMath.sub_fast(data(i), j))

@inline vadd_nsw(i::MM{W,X}, j::IntegerTypesHW) where {W,X} = MM{W,X}(vadd_nsw(data(i), j))
@inline vadd_nsw(i::IntegerTypesHW, j::MM{W,X}) where {W,X} = MM{W,X}(vadd_nsw(i, data(j)))
@inline vadd_nsw(i::MM{W,X}, ::StaticInt{j}) where {W,X,j} = MM{W,X}(vadd_nsw(data(i), StaticInt{j}()))
@inline vadd_nsw(i::MM{W,X}, ::Zero) where {W,X} = i
@inline vadd_nsw(::StaticInt{i}, j::MM{W,X}) where {W,X,i} = MM{W,X}(vadd_nsw(StaticInt{i}(), data(j)))
@inline vadd_nsw(::Zero, j::MM{W,X}) where {W,X} = j
@inline vsub_nsw(i::MM{W,X}, j::IntegerTypesHW) where {W,X} = MM{W,X}(vsub_nsw(data(i), j))
@inline vsub_nsw(i::MM{W,X}, ::StaticInt{j}) where {W,X,j} = MM{W,X}(vsub_nsw(data(i), StaticInt{j}()))
@inline vsub_nsw(i::MM{W,X}, ::Zero) where {W,X} = i
@inline FastMath.sub_fast(i::MM) = i * StaticInt{-1}()
@inline vsub_nsw(i::MM) = i * StaticInt{-1}()
@inline Base.:(-)(i::MM) = i * StaticInt{-1}()
@inline Base.:(+)(i::MM, j::IntegerTypesHW) = FastMath.add_fast(i, j)
@inline Base.:(+)(j::IntegerTypesHW, i::MM) = FastMath.add_fast(j, i)
@inline Base.:(-)(i::MM, j::IntegerTypesHW) = FastMath.sub_fast(i, j)
@inline Base.:(-)(j::IntegerTypesHW, i::MM) = FastMath.sub_fast(j, i)


@inline vmul_nsw(::StaticInt{M}, i::MM{W,X}) where {M,W,X} = MM{W}(vmul_nsw(data(i), StaticInt{M}()), StaticInt{X}() * StaticInt{M}())
@inline vmul_nsw(i::MM{W,X}, ::StaticInt{M}) where {M,W,X} = MM{W}(vmul_nsw(data(i), StaticInt{M}()), StaticInt{X}() * StaticInt{M}())

@inline vmul_fast(::StaticInt{M}, i::MM{W,X}) where {M,W,X} = MM{W}(vmul_fast(data(i), StaticInt{M}()), StaticInt{X}() * StaticInt{M}())
@inline vmul_fast(i::MM{W,X}, ::StaticInt{M}) where {M,W,X} = MM{W}(vmul_fast(data(i), StaticInt{M}()), StaticInt{X}() * StaticInt{M}())
@inline vmul(a, ::StaticInt{N}) where {N} = vmul_fast(a, StaticInt{N}())
@inline vmul(::StaticInt{N}, a) where {N} = vmul_fast(StaticInt{N}(), a)
@inline vmul(::StaticInt{N}, ::StaticInt{M}) where {N,M} = StaticInt{N}() * StaticInt{M}()

@inline vrem(i::MM{W,X,I}, ::Type{I}) where {W,X,I<:IntegerTypesHW} = i
@inline vrem(i::MM{W,X}, ::Type{I}) where {W,X,I<:IntegerTypesHW} = MM{W,X}(data(i) % I)
@inline veq(::AbstractIrrational, ::MM{W,<:Integer}) where {W} = zero(Mask{W})
@inline veq(x::AbstractIrrational, i::MM{W}) where {W} = x == Vec(i)
@inline veq(::MM{W,<:Integer}, ::AbstractIrrational) where {W} = zero(Mask{W})
@inline veq(i::MM{W}, x::AbstractIrrational) where {W} = Vec(i) == x

@inline vsub_nsw(i::NativeTypes, j::MM{W,X}) where {W,X} = MM(StaticInt{W}(), vsub_nsw(i, data(j)), -StaticInt{X}())
@inline FastMath.sub_fast(i::NativeTypes, j::MM{W,X}) where {W,X} = MM(StaticInt{W}(), FastMath.sub_fast(i, data(j)), -StaticInt{X}())
@inline vsub_nsw(i::Union{FloatingTypes,IntegerTypesHW}, j::MM{W,X}) where {W,X} = MM(StaticInt{W}(), vsub_nsw(i, data(j)), -StaticInt{X}())
@inline FastMath.sub_fast(i::Union{FloatingTypes,IntegerTypesHW}, j::MM{W,X}) where {W,X} = MM(StaticInt{W}(), FastMath.sub_fast(i, data(j)), -StaticInt{X}())

@inline function Base.in(m::MM{W,X,<:Integer}, r::AbstractUnitRange) where {W,X}
    vm = Vec(m)
    (vm ≥ first(r)) & (vm ≤ last(r))
end

# @inline function pick_vector_width_shift(args::Vararg{Any,K}) where {K}
#     W = pick_vector_width(args...)
#     W, intlog2(W)
# end
    

