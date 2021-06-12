
# This is experimental, as few arches support it, and I can't think of many uses other than floating point RNGs.

@inline __ifmalo(v1, v2, v3) = ((((v1 % UInt64)) * ((v2 % UInt64))) & 0x000fffffffffffff) + (v3 % UInt64)
@inline _ifmalo(v1, v2, v3) = __ifmalo(v1, v2, v3)
function ifmahi_quote(W)
  mask = W > 1 ? llvmconst(W, "i64 4503599627370495") : "4503599627370495"
  shift = W > 1 ? llvmconst(W, "i128 52") : "52"
  t64 = W > 1 ? "<$W x i64>" : "i64"
  t128 = W > 1 ? "<$W x i128>" : "i128"
  instrs = """
          %a52 = and $t64 %0, $mask
          %b52 = and $t64 %1, $mask
          %a128 = zext $t64 %a52 to $t128
          %b128 = zext $t64 %b52 to $t128
          %c128 = mul $t128 %a128, %b128
          %csr = lshr $t128 %c128, $shift
          %c64 = trunc $t128 %csr to $t64
          %res = add $t64 %c64, %2
          ret $t64 %res
      """
  jt = W > 1 ? :(_Vec{$W,UInt64}) : :UInt64
  call = :($LLVMCALL($instrs, $jt, Tuple{$jt,$jt,$jt}, data(v1), data(v2), data(v3)))
  W > 1 && (call = Expr(:call, :Vec, call))
  Expr(:block, Expr(:meta,:inline), call)
end
@generated _ifmahi(v1::UInt64, v2::UInt64, v3::UInt64) = ifmahi_quote(1)
@generated __ifmahi(v1::Vec{W,UInt64}, v2::Vec{W,UInt64}, v3::Vec{W,UInt64}) where {W} = ifmahi_quote(W)

function ifmaquote(W::Int, lo::Bool)
  op = lo ? "@llvm.x86.avx512.vpmadd52l.uq.$(64W)" : "@llvm.x86.avx512.vpmadd52h.uq.$(64W)"
  decl = "declare <$W x i64> $op(<$W x i64>, <$W x i64>, <$W x i64>)"
  instrs = "%res = call <$W x i64> $op(<$W x i64> %0, <$W x i64> %1, <$W x i64> %2)\n ret <$W x i64> %res"
  llvmcall_expr(decl, instrs, :(_Vec{$W,UInt64}), :(Tuple{_Vec{$W,UInt64},_Vec{$W,UInt64},_Vec{$W,UInt64}}), "<$W x i64>", ["<$W x i64>","<$W x i64>","<$W x i64>"], [:(data(v3)), :(data(v1)), :(data(v2))])
end
@generated _ifmalo(v1::Vec{W,UInt64},v2::Vec{W,UInt64},v3::Vec{W,UInt64}, ::True) where {W} = ifmaquote(W, true)
@generated _ifmahi(v1::Vec{W,UInt64},v2::Vec{W,UInt64},v3::Vec{W,UInt64}, ::True) where {W} = ifmaquote(W, false)

@inline _ifmalo(v1::Vec{W,UInt64},v2::Vec{W,UInt64},v3::Vec{W,UInt64}, ::False) where {W} = __ifmalo(v1, v2, v3)
@inline _ifmahi(v1::Vec{W,UInt64},v2::Vec{W,UInt64},v3::Vec{W,UInt64}, ::False) where {W} = __ifmahi(v1, v2, v3)


