
# unary # 2^2 - 2 = 2 definitions
@inline fmap(f::F, x::Tuple{X}) where {F,X} = (f(first(x)),)
@inline fmap(f::F, x::NTuple) where {F} = (f(first(x)), fmap(f, Base.tail(x))...)

# binary # 2^3 - 2 = 6 definitions
@inline fmap(f::F, x::Tuple{X}, y::Tuple{Y}) where {F,X,Y} = (f(first(x), first(y)),)
@inline fmap(f::F, x::Tuple{X}, y) where {F,X} = (f(first(x), y),)
@inline fmap(f::F, x, y::Tuple{Y}) where {F,Y} = (f(x, first(y)),)
@inline fmap(f::F, x::Tuple{Vararg{Any,N}}, y::Tuple{Vararg{Any,N}}) where {F,N} = (f(first(x), first(y)), fmap(f, Base.tail(x), Base.tail(y))...)
@inline fmap(f::F, x::Tuple, y) where {F} = (f(first(x), y), fmap(f, Base.tail(x), y)...)
@inline fmap(f::F, x, y::Tuple) where {F} = (f(x, first(y)), fmap(f, x, Base.tail(y))...)


fmap(f::F, x::Tuple{X}, y::Tuple) where {F,X} = throw("Dimension mismatch.")
fmap(f::F, x::Tuple, y::Tuple{Y}) where {F,Y} = throw("Dimension mismatch.")
fmap(f::F, x::Tuple, y::Tuple) where {F} = throw("Dimension mismatch.")

@generated function fmap(f::F, x::Vararg{Any,N}) where {F,N}
    q = Expr(:block, Expr(:meta, :inline))
    t = Expr(:tuple)
    U = 1
    call = Expr(:call, :f)
    syms = Vector{Symbol}(undef, N)
    istup = Vector{Bool}(undef, N)
    gf = GlobalRef(Core, :getfield)
    for n ∈ 1:N
        syms[n] = xₙ = Symbol(:x_, n)
        push!(q.args, Expr(:(=), xₙ, Expr(:call, gf, :x, n, false)))
        istup[n] = ist = (x[n] <: Tuple)
        if ist
            U = length(x[n].parameters)
            push!(call.args, Expr(:call, gf, xₙ, 1, false))
        else
            push!(call.args, xₙ)
        end
    end
    push!(t.args, call)
    for u ∈ 2:U
        call = Expr(:call, :f)
        for n ∈ 1:N
            xₙ = syms[n]
            if istup[n]
                push!(call.args, Expr(:call, gf, xₙ, u, false))
            else
                push!(call.args, xₙ)
            end
        end
        push!(t.args, call)
    end
    push!(q.args, t); q
end

for op ∈ [:(-), :abs, , :floor, :ceil, :trunc, :round, :sqrt, :!, :~, :leading_zeros, :trailing_zeros, :inv]
    @eval @inline Base.$op(v1::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, getfield(v1, :data)))
end
for op ∈ [:abs_fast, :abs2_fast, :sub_fast, :sqrt_fast, :inv_fast]
    @eval @inline FastMath.$op(v1::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, getfield(v1, :data)))
end
# only for `Float32` and `Float64`
@inline inv_approx(v::VecUnroll{N,W,T}) where {N,W,T<:Union{Float32,Float64}} = VecUnroll(fmap(inv_approx, getfield(v, :data)))

@inline Base.reinterpret(::Type{T}, v::VecUnroll) where {T<:Number} = VecUnroll(fmap(vreinterpret, T, getfield(v, :data)))
for op ∈ [:(Base.+),:(Base.-),:(Base.*),:(Base.&),:(Base.|),:(Base.⊻),:(Base.<),:(Base.≤),:(Base.>),:(Base.≥),:(Base.==),:(Base.≠),
          :(FastMath.add_fast),:(FastMath.sub_fast),:(FastMath.mul_fast)]
  @eval begin
    @inline $op(v1::VecUnroll, v2::VecUnroll) = VecUnroll(fmap($op, getfield(v1, :data), getfield(v2, :data)))
    @inline $op(v1::VecUnroll, v2::VecOrScalar) = VecUnroll(fmap($op, getfield(v1, :data), v2))
    @inline $op(v1::VecOrScalar, v2::VecUnroll) = VecUnroll(fmap($op, v1, getfield(v2, :data)))
    @inline $op(v1::VecUnroll{N,W,T,V}, ::StaticInt{M}) where {N,W,T,V,M} = VecUnroll(fmap($op, getfield(v1, :data), vbroadcast(Val{W}(), T(M))))
    @inline $op(::StaticInt{M}, v1::VecUnroll{N,W,T,V}) where {N,W,T,V,M} = VecUnroll(fmap($op, vbroadcast(Val{W}(), T(M)), getfield(v1, :data)))

    @inline $op(v1::VecUnroll{N,W,T,V}, ::StaticInt{1}) where {N,W,T,V} = VecUnroll(fmap($op, getfield(v1, :data), One()))
    @inline $op(::StaticInt{1}, v1::VecUnroll{N,W,T,V}) where {N,W,T,V} = VecUnroll(fmap($op, One(), getfield(v1, :data)))

    @inline $op(v1::VecUnroll{N,W,T,V}, ::StaticInt{0}) where {N,W,T,V} = VecUnroll(fmap($op, getfield(v1, :data), Zero()))
    @inline $op(::StaticInt{0}, v1::VecUnroll{N,W,T,V}) where {N,W,T,V} = VecUnroll(fmap($op, Zero(), getfield(v1, :data)))
  end
end
for op ∈ [:(FastMath.div_fast), :(FastMath.min_fast), :(FastMath.max_fast), :(Base./),:(Base.÷), :(Base.max), :(Base.min), :(Base.copysign),
          :(Base.<<), :(Base.>>), :(Base.>>>), :(Base.%), :(FastMath.rem_fast)]
  @eval begin
    @inline $op(v1::VecUnroll, v2::VecUnroll) = VecUnroll(fmap($op, getfield(v1, :data), getfield(v2, :data)))
    @inline $op(v1::VecOrScalar, v2::VecUnroll) = VecUnroll(fmap($op, v1, getfield(v2, :data)))
    @inline $op(v1::VecUnroll, v2::VecOrScalar) = VecUnroll(fmap($op, getfield(v1, :data), v2))
  end
end
for op ∈ [:%, :&, :|, :⊻, :<<, :>>, :>>>, :<, :≤,:>,:≥,:==,:≠]
  @eval begin
    @inline Base.$op(vu::VecUnroll, i::MM) = $op(vu, Vec(i))
    @inline Base.$op(i::MM, vu::VecUnroll) = $op(Vec(i), vu)
  end
end
for op ∈ [:<<, :>>, :>>>]
    @eval @inline Base.$op(m::AbstractMask, vu::VecUnroll) = $op(Vec(m), vu)
end
for op ∈ [:rotate_left,:rotate_right,:funnel_shift_left,:funnel_shift_right]
  @eval begin
    @inline $op(v1::VecUnroll{N,W,T,V}, v2::R) where {N,W,T,V,R<:IntegerTypes} = VecUnroll(fmap($op, getfield(v1, :data), promote_type(V, R)(v2)))
    @inline $op(v1::R, v2::VecUnroll{N,W,T,V}) where {N,W,T,V,R<:IntegerTypes} = VecUnroll(fmap($op, promote_type(V, R)(v1), getfield(v2, :data)))
    @inline $op(v1::VecUnroll, v2::VecUnroll) = VecUnroll(fmap($op, getfield(v1, :data), getfield(v2, :data)))
  end
end

for op ∈ [:fma, :fma_fast, :vfmadd, :vfnmadd, :vfmsub, :vfnmsub, :vfmadd_fast, :vfnmadd_fast, :vfmsub_fast, :vfnmsub_fast,
          :vfmadd231, :vfnmadd231, :vfmsub231, :vfnmsub231, :ifmahi, :ifmalo]
  @eval begin
    # @generated function $op(v1::VecUnroll{N,W,T1,V1}, v2::VecUnroll{N,W,T2,V2}, v3::VecUnroll{N,W,T3,V3}) where {N,W,T1,T2,T3}
    #   if T1 <: NativeTypes
    #   VecUnroll(fmap($op, getfield(v1, :data), getfield(v2, :data), getfield(v3, :data)))
    #   Expr(:block, Expr(:meta,:inline), ex)
    # end
    @inline function $op(v1::VecUnroll{N,W,<:NativeTypesExceptBit}, v2::VecUnroll{N,W,<:NativeTypesExceptBit}, v3::VecUnroll{N,W}) where {N,W}
      VecUnroll(fmap($op, getfield(v1, :data), getfield(v2, :data), getfield(v3, :data)))
    end
    @inline function $op(v1::VecUnroll{N,W,<:NativeTypesExceptBit}, v2::VecUnroll{N,W,<:NativeTypesExceptBit}, v3::VecOrScalar{W}) where {N,W}
      VecUnroll(fmap($op, getfield(v1, :data), getfield(v2, :data), v3))
    end
    @inline function $op(v1::VecUnroll{N,W,<:NativeTypesExceptBit}, v2::VecOrScalar{W,<:NativeTypesExceptBit}, v3::VecUnroll{N,W}) where {N,W}
      VecUnroll(fmap($op, getfield(v1, :data), v2, getfield(v3, :data)))
    end
    @inline function $op(v1::VecOrScalar{W,<:NativeTypesExceptBit}, v2::VecUnroll{N,W,<:NativeTypesExceptBit}, v3::VecUnroll{N,W}) where {N,W}
      VecUnroll(fmap($op, v1, getfield(v2, :data), getfield(v3, :data)))
    end
    @inline function $op(v1::VecOrScalar{W,<:NativeTypesExceptBit}, v2::VecOrScalar{W,<:NativeTypesExceptBit}, v3::VecUnroll{N,W}) where {N,W}
      VecUnroll(fmap($op, v1, v2, getfield(v3, :data)))
    end
    @inline function $op(v1::VecOrScalar{W,<:NativeTypesExceptBit}, v2::VecUnroll{N,W,<:NativeTypesExceptBit}, v3::VecOrScalar{W}) where {N,W}
      VecUnroll(fmap($op, v1, getfield(v2, :data), v3))
    end
    @inline function $op(v1::VecUnroll{N,W,<:NativeTypesExceptBit}, v2::VecOrScalar{W,<:NativeTypesExceptBit}, v3::VecOrScalar{W}) where {N,W}
      VecUnroll(fmap($op, getfield(v1, :data), v2, v3))
    end
  end
end
@inline function ifelse(v1::VecUnroll{N,W,<:Boolean}, v2::T, v3::T) where {N,W,T<:NativeTypes}
  VecUnroll(fmap(ifelse, getfield(v1, :data), Vec{W,T}(v2), Vec{W,T}(v3)))
end
@inline function ifelse(v1::VecUnroll{N,W,<:Boolean}, v2::T, v3::T) where {N,W,T<:Real}
  VecUnroll(fmap(ifelse, getfield(v1, :data), v2, v3))
end
@inline function ifelse(v1::Vec{W,Bool}, v2::VecUnroll{N,W,T}, v3::Union{NativeTypes,AbstractSIMDVector,StaticInt}) where {N,W,T}
  VecUnroll(fmap(ifelse, Vec{W,T}(v1), getfield(v2, :data), Vec{W,T}(v3)))
end
@inline function ifelse(v1::Vec{W,Bool}, v2::Union{NativeTypes,AbstractSIMDVector,StaticInt}, v3::VecUnroll{N,W,T}) where {N,W,T}
  VecUnroll(fmap(ifelse, Vec{W,T}(v1), Vec{W,T}(v2), getfield(v3, :data)))
end
@inline function ifelse(v1::VecUnroll{N,WB,<:Boolean}, v2::VecUnroll{N,W,T}, v3::Union{NativeTypes,AbstractSIMDVector,StaticInt}) where {N,W,WB,T}
  VecUnroll(fmap(ifelse, getfield(v1, :data), getfield(v2, :data), Vec{W,T}(v3)))
end
@inline function ifelse(v1::VecUnroll{N,WB,<:Boolean}, v2::Union{NativeTypes,AbstractSIMDVector,StaticInt}, v3::VecUnroll{N,W,T}) where {N,W,WB,T}
  VecUnroll(fmap(ifelse, getfield(v1, :data), Vec{W,T}(v2), getfield(v3, :data)))
end
@inline function ifelse(v1::Vec{W,Bool}, v2::VecUnroll{N,W,T}, v3::VecUnroll{N,W,T}) where {N,W,T}
  VecUnroll(fmap(ifelse, Vec{W,T}(v1), getfield(v2, :data), getfield(v3, :data)))
end
@inline function ifelse(v1::VecUnroll{N,WB,<:Boolean}, v2::VecUnroll{N,W,T}, v3::VecUnroll{N,W,T}) where {N,W,WB,T}
  VecUnroll(fmap(ifelse, getfield(v1, :data), getfield(v2, :data), getfield(v3, :data)))
end
@inline function ifelse(v1::VecUnroll{N,WB,<:Boolean}, v2::VecUnroll{N,W}, v3::VecUnroll{N,W}) where {N,W,WB}
  v4, v5 = promote(v2, v3)
  VecUnroll(fmap(ifelse, getfield(v1, :data), getfield(v4, :data), getfield(v5, :data)))
end


@inline Base.:(==)(v::VecUnroll{N,W,T}, x::AbstractIrrational) where {N,W,T} = v == vbroadcast(Val{W}(), T(x))
@inline Base.:(==)(x::AbstractIrrational, v::VecUnroll{N,W,T}) where {N,W,T} = vbroadcast(Val{W}(), T(x)) == v

@inline Base.unsafe_trunc(::Type{T}, v::VecUnroll) where {T<:Real} = VecUnroll(fmap(vunsafe_trunc, T, getfield(v, :data)))
@inline Base.:(%)(v::VecUnroll, ::Type{T}) where {T<:Real} = VecUnroll(fmap(vrem, getfield(v, :data), T))
@inline Base.:(%)(v::VecUnroll{N,W1}, ::Type{VecUnroll{N,W2,T,V}}) where {N,W1,W2,T,V} = VecUnroll(fmap(vrem, getfield(v, :data), V))

@inline (::Type{VecUnroll{N,W,T,V}})(vu::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = vu
@inline function (::Type{VecUnroll{N,W,T,VT}})(vu::VecUnroll{N,W,S,VS})  where {N,W,T,VT<:AbstractSIMDVector{W,T},S,VS<:AbstractSIMDVector{W,S}}
  VecUnroll(fmap(convert, Vec{W,T}, getfield(vu, :data)))
end


function collapse_expr(N, op, final)
  N += 1
  t = Expr(:tuple); s = Vector{Symbol}(undef, N)
  for n ∈ 1:N
    s_n = s[n] = Symbol(:v_, n)
    push!(t.args, s_n)
  end
  q = quote
    $(Expr(:meta,:inline))
    $t = data(vu)
  end
  _final = if final == 1
    1
  else
    2final
  end
  while N > _final
    for n ∈ 1:N >>> 1
      push!(q.args, Expr(:(=), s[n], Expr(:call, op, s[n], s[n + (N >>> 1)])))
    end
    isodd(N) && push!(q.args, Expr(:(=), s[1], Expr(:call, op, s[1], s[N])))
    N >>>= 1
  end
  if final != 1
    for n ∈ final+1:N
      push!(q.args, Expr(:(=), s[n-final], Expr(:call, op, s[n-final], s[n])))
    end
    t = Expr(:tuple)
    for n ∈ 1:final
      push!(t.args, s[n])
    end
    push!(q.args, :(VecUnroll($t)))
  end
  q
end
@generated collapse_add(vu::VecUnroll{N}) where {N} = collapse_expr(N, :+, 1)
@generated collapse_mul(vu::VecUnroll{N}) where {N} = collapse_expr(N, :*, 1)
@generated collapse_max(vu::VecUnroll{N}) where {N} = collapse_expr(N, :max, 1)
@generated collapse_min(vu::VecUnroll{N}) where {N} = collapse_expr(N, :min, 1)
@generated collapse_and(vu::VecUnroll{N}) where {N} = collapse_expr(N, :&, 1)
@generated collapse_or(vu::VecUnroll{N}) where {N} = collapse_expr(N, :|, 1)

@generated contract_add(vu::VecUnroll{N}, ::StaticInt{C}) where {N,C} = collapse_expr(N, :+, C)
@generated contract_mul(vu::VecUnroll{N}, ::StaticInt{C}) where {N,C} = collapse_expr(N, :*, C)
@generated contract_max(vu::VecUnroll{N}, ::StaticInt{C}) where {N,C} = collapse_expr(N, :max, C)
@generated contract_min(vu::VecUnroll{N}, ::StaticInt{C}) where {N,C} = collapse_expr(N, :min, C)
@generated contract_and(vu::VecUnroll{N}, ::StaticInt{C}) where {N,C} = collapse_expr(N, :&, C)
@generated contract_or(vu::VecUnroll{N}, ::StaticInt{C}) where {N,C} = collapse_expr(N, :|, C)
@inline vsum(vu::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = VecUnroll(fmap(vsum, data(vu)))::VecUnroll{N,1,T,T}
@inline vsum(s::VecUnroll, vu::VecUnroll) = VecUnroll(fmap(vsum, data(s), data(vu)))
@inline vprod(vu::VecUnroll) = VecUnroll(fmap(vprod, data(vu)))
@inline vprod(s::VecUnroll, vu::VecUnroll) = VecUnroll(fmap(vprod, data(s), data(vu)))
@inline vmaximum(vu::VecUnroll) = VecUnroll(fmap(vmaximum, data(vu)))
@inline vminimum(vu::VecUnroll) = VecUnroll(fmap(vminimum, data(vu)))
@inline vall(vu::VecUnroll) = VecUnroll(fmap(vall, data(vu)))
@inline vany(vu::VecUnroll) = VecUnroll(fmap(vany, data(vu)))

@inline collapse_add(x) = x
@inline collapse_mul(x) = x
@inline collapse_max(x) = x
@inline collapse_min(x) = x
@inline collapse_and(x) = x
@inline collapse_or(x) = x
# @inline vsum(vu::VecUnroll) = vsum(collapse_add(vu))
# @inline vsum(s, vu::VecUnroll) = vsum(s, collapse_add(vu))
# @inline vprod(vu::VecUnroll) = vprod(collapse_mul(vu))
# @inline vprod(s, vu::VecUnroll) = vprod(s, collapse_mul(vu))
# @inline vmaximum(vu::VecUnroll) = vmaximum(collapse_max(vu))
# @inline vminimum(vu::VecUnroll) = vminimum(collapse_min(vu))
# @inline vall(vu::VecUnroll) = vall(collapse_and(vu))
# @inline vany(vu::VecUnroll) = vany(collapse_or(vu))

