@inline static(::Val{N}) where {N} = StaticInt{N}()
@inline static(::Nothing) = nothing
@generated function static_sizeof(::Type{T}) where {T}
    st = Base.allocatedinline(T) ? sizeof(T) : sizeof(Int)
    Expr(:block, Expr(:meta, :inline), Expr(:call, Expr(:curly, :StaticInt, st)))
end

# These have versions that may allow for more optimizations, so we override base methods with a single `StaticInt` argument.
for (f,ff) ∈ [
  # (:(Base.:+),:vadd_fast), (:(Base.:-),:vsub_fast), (:(Base.:*),:vmul_fast),
  (:(Base.:+),:vadd_nsw), (:(Base.:-),:vsub_nsw), (:(Base.:*),:vmul_nsw),
  (:(Base.:+),:vadd_nuw), (:(Base.:-),:vsub_nuw), (:(Base.:*),:vmul_nuw),
  (:(Base.:+),:vadd_nw), (:(Base.:-),:vsub_nw), (:(Base.:*),:vmul_nw),
  # (:(Base.:<<),:vshl), (:(Base.:÷),:(Base.div)), (:(Base.:%), :(Base.rem)), (:(Base.:>>>),:(Base.:>>>))
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
for f ∈ [:vsub_nsw, :vsub_nuw, :vsub_nw]
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
for f ∈ [:vadd_nsw, :vadd_nuw, :vadd_nw]
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

@inline Base.:(+)(m::AbstractMask{W}, ::StaticInt{N}) where {N,W} = m + vbroadcast(Val{W}(), N)
@inline Base.:(+)(::StaticInt{N}, m::AbstractMask{W}) where {N,W} = vbroadcast(Val{W}(), N) + m

for V ∈ [:AbstractSIMD, :MM, :VecUnroll, :AbstractMask]
  for (mod,m,a,s) ∈ ((:FastMath, :mul_fast, :add_fast, :sub_fast),(:Base,:(*),:(+),:(-)))
    @eval begin
      @inline $mod.$m(::Zero, x::$V) = Zero()
      @inline $mod.$m(::One, x::$V) = x
      @inline $mod.$m(x::$V, ::Zero) = Zero()
      @inline $mod.$m(x::$V, ::One) = x

      @inline $mod.$a(::Zero, x::$V) = x
      @inline $mod.$a(x::$V, ::Zero) = x

      @inline $mod.$s(::Zero, x::$V) = -x
      @inline $mod.$s(x::$V, ::Zero) =  x
    end
  end
end


@inline vload(::StaticInt{N}, args...) where {N} = StaticInt{N}()
@inline stridedpointer(::StaticInt{N}) where {N} = StaticInt{N}()
@inline zero_offsets(::StaticInt{N}) where {N} = StaticInt{N}()

