@inline static(::Val{N}) where {N} = StaticInt{N}()
@inline static(::Nothing) = nothing
@generated function static_sizeof(::Type{T}) where {T}
    st = Base.allocatedinline(T) ? sizeof(T) : sizeof(Int)
    Expr(:block, Expr(:meta, :inline), Expr(:call, Expr(:curly, :StaticInt, st)))
end

# These have versions that may allow for more optimizations, so we override base methods with a single `StaticInt` argument.
for (f,ff) ∈ [
  (:(Base.:+),:vadd_fast), (:(Base.:-),:vsub_fast), (:(Base.:*),:vmul_fast),
  (:(Base.:+),:vadd_nsw), (:(Base.:-),:vsub_nsw), (:(Base.:*),:vmul_nsw),
  (:(Base.:+),:vadd_nuw), (:(Base.:-),:vsub_nuw), (:(Base.:*),:vmul_nuw),
  (:(Base.:+),:vadd_nw), (:(Base.:-),:vsub_nw), (:(Base.:*),:vmul_nw),
  (:(Base.:<<),:vshl), (:(Base.:÷),:vdiv), (:(Base.:%), :vrem), (:(Base.:>>>),:vashr)
]
  @eval begin
    # @inline $f(::StaticInt{M}, ::StaticInt{N}) where {M, N} = StaticInt{$f(M, N)}()
    # If `M` and `N` are known at compile time, there's no need to add nsw/nuw flags.
    @inline $ff(::StaticInt{M}, ::StaticInt{N}) where {M, N} = $f(StaticInt{M}(),StaticInt{N}())
    # @inline $f(::StaticInt{M}, x) where {M} = $ff(M, x)
    # @inline $f(x, ::StaticInt{M}) where {M} = $ff(x, M)
    @inline $ff(::StaticInt{M}, x::T) where {M,T<:IntegerTypesHW} = $ff(M%T, x)
    @inline $ff(x::T, ::StaticInt{M}) where {M,T<:IntegerTypesHW} = $ff(x, M%T)
    @inline $ff(::StaticInt{M}, x) where {M} = $ff(M, x)
    @inline $ff(x, ::StaticInt{M}) where {M} = $ff(x, M)
  end
end
for f ∈ [:vadd_fast, :vsub_fast, :vmul_fast]
    @eval @inline $f(::StaticInt{M}, n::Number) where {M} = $f(M, n)
    @eval @inline $f(m::Number, ::StaticInt{N}) where {N} = $f(m, N)
end
for f ∈ [:vsub, :vsub_fast, :vsub_nsw, :vsub_nuw, :vsub_nw]
  @eval begin
    @inline $f(::Zero, m::Number) = -m
    @inline $f(::Zero, m::IntegerTypesHW) = -m
    @inline $f(m::Number, ::Zero) =  m
    @inline $f(m::IntegerTypesHW, ::Zero) =  m
    @inline $f(::Zero, ::Zero) = Zero()
    @inline $f(::Zero, ::StaticInt{N}) where {N} = -StaticInt{N}()
    @inline $f(::StaticInt{N}, ::Zero) where {N} = StaticInt{N}()
  end
end
for f ∈ [:vadd, :vadd_fast, :vadd_nsw, :vadd_nuw, :vadd_nw]
  @eval begin
    @inline $f(::StaticInt{N}, ::Zero) where {N} = StaticInt{N}()
    @inline $f(::Zero, ::StaticInt{N}) where {N} = StaticInt{N}()
    @inline $f(::Zero, ::Zero) = Zero()
    @inline $f(a::Number, ::Zero) = a
    @inline $f(a::IntegerTypesHW, ::Zero) = a
    @inline $f(::Zero, a::Number) = a
    @inline $f(::Zero, a::IntegerTypesHW) = a
  end
end

@inline vmul_fast(::StaticInt{N}, ::Zero) where {N} = Zero()
@inline vmul_fast(::Zero, ::StaticInt{N}) where {N} = Zero()
@inline vmul_fast(::Zero, ::Zero) = Zero()
@inline vmul_fast(::StaticInt{N}, ::One) where {N} = StaticInt{N}()
@inline vmul_fast(::One, ::StaticInt{N}) where {N} = StaticInt{N}()
@inline vmul_fast(::One, ::One) = One()
@inline vmul_fast(a::Number, ::One) = a
@inline vmul_fast(a::IntegerTypesHW, ::One) = a
@inline vmul_fast(::One, a::Number) = a
@inline vmul_fast(::One, a::IntegerTypesHW) = a
@inline vmul_fast(::Zero, ::One) = Zero()
@inline vmul_fast(::One, ::Zero) = Zero()
@inline vmul_fast(i::MM{W,X}, ::StaticInt{1}) where {W,X} = i
@inline vmul_fast(::StaticInt{1}, i::MM{W,X}) where {W,X} = i

for T ∈ [:VecUnroll, :AbstractMask, :MM]
    @eval begin
        @inline Base.:(+)(x::$T, ::Zero) = x
        @inline Base.:(+)(::Zero, x::$T) = x
        @inline Base.:(-)(x::$T, ::Zero) = x
        @inline Base.:(*)(x::$T, ::One) = x
        @inline Base.:(*)(::One, x::$T) = x
        @inline Base.:(*)(::$T, ::Zero) = Zero()
        @inline Base.:(*)(::Zero, ::$T) = Zero()
    end
end
@inline Base.:(+)(m::AbstractMask{W}, ::StaticInt{N}) where {N,W} = m + vbroadcast(Val{W}(), N)
@inline Base.:(+)(::StaticInt{N}, m::AbstractMask{W}) where {N,W} = vbroadcast(Val{W}(), N) + m
# @inline Base.:(*)(::StaticInt{N}, m::Mask{W}) where {N,W} = vbroadcast(Val{W}(), N) * m
@inline vadd_fast(x::VecUnroll, ::Zero) = x
@inline vadd_fast(::Zero, x::VecUnroll) = x
@inline vsub_fast(x::VecUnroll, ::Zero) = x
@inline vmul_fast(x::VecUnroll, ::One) = x
@inline vmul_fast(::One, x::VecUnroll) = x
@inline vmul_fast(::VecUnroll, ::Zero) = Zero()
@inline vmul_fast(::Zero, ::VecUnroll) = Zero()

for V ∈ [:AbstractSIMD, :MM]
    @eval begin
        @inline Base.FastMath.mul_fast(::Zero, x::$V) = Zero()
        @inline Base.FastMath.mul_fast(::One, x::$V) = x
        @inline Base.FastMath.mul_fast(x::$V, ::Zero) = Zero()
        @inline Base.FastMath.mul_fast(x::$V, ::One) = x

        @inline Base.FastMath.add_fast(::Zero, x::$V) = x
        @inline Base.FastMath.add_fast(x::$V, ::Zero) = x

        @inline Base.FastMath.sub_fast(::Zero, x::$V) = -x
        @inline Base.FastMath.sub_fast(x::$V, ::Zero) =  x
    end
end


@inline vload(::StaticInt{N}, args...) where {N} = StaticInt{N}()
@inline stridedpointer(::StaticInt{N}) where {N} = StaticInt{N}()
@inline zero_offsets(::StaticInt{N}) where {N} = StaticInt{N}()

