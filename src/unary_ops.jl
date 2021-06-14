
function sub_quote(W, @nospecialize(T), fast::Bool)
  vtyp = vtype(W, T)
  instrs = "%res = fneg $(fast_flags(fast)) $vtyp %0\nret $vtyp %res"
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$W,$T}, Tuple{_Vec{$W,$T}}, data(v)))
  end
end

@generated Base.:(-)(v::Vec{W,T}) where {W, T <: Union{Float32,Float64}} = sub_quote(W, T, false)
@generated Base.FastMath.sub_fast(v::Vec{W,T}) where {W, T <: Union{Float32,Float64}} = sub_quote(W, T, true)

@inline Base.:(-)(v::Vec{<:Any,<:NativeTypes}) = FastMath.sub_fast(zero(v), v)
@inline Base.FastMath.sub_fast(v::Vec{<:Any,<:NativeTypes}) = FastMath.sub_fast(zero(v), v)

@inline Base.inv(v::AbstractSIMD{W,<:FloatingTypes}) where {W} = FastMath.div_fast(one(v), v)
@inline Base.inv(v::AbstractSIMD{W,<:IntegerTypesHW}) where {W} = inv(float(v))
@inline FastMath.inv_fast(v::AbstractSIMD{<:Any,<:IntegerTypesHW}) = FastMath.inv_fast(float(v))
@inline FastMath.inv_fast(v::AbstractSIMD{<:Any,<:FloatingTypes}) = FastMath.div_fast(one(v), v)

@inline Base.abs(v::AbstractSIMD{W,<:Unsigned}) where {W} = v
@inline Base.abs(v::AbstractSIMD{W,<:Signed}) where {W} = ifelse(v > 0, v, -v)

@inline Base.round(v::AbstractSIMD{W,<:Integer}) where {W} = v
@inline Base.round(v::AbstractSIMD{W,<:Integer}, ::RoundingMode) where {W} = v

