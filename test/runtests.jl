using LLVMCalls
using Test

using LLVMCalls: Vec, VecUnroll

include("testsetup.jl")

@testset "LLVMCalls.jl" begin

  if false
  
  println("Binary Functions")
  @time @testset "Binary Functions" begin
    # TODO: finish getting these tests to pass
    # for I1 ∈ (Int32,Int64,UInt32,UInt64), I2 ∈ (Int32,Int64,UInt32,UInt64)
    for (vf,bf,testfloat) ∈ [(LLVMCalls.vadd,+,true),(LLVMCalls.vadd_fast,Base.FastMath.add_fast,true),(LLVMCalls.vadd_nsw,+,false),#(LLVMCalls.vadd_nuw,+,false),(LLVMCalls.vadd_nw,+,false),
                             (LLVMCalls.vsub,-,true),(LLVMCalls.vsub_fast,Base.FastMath.sub_fast,true),(LLVMCalls.vsub_nsw,-,false),#(LLVMCalls.vsub_nuw,-,false),(LLVMCalls.vsub_nw,-,false),
                             (LLVMCalls.vmul,*,true),(LLVMCalls.vmul_fast,Base.FastMath.mul_fast,true),(LLVMCalls.vmul_nsw,*,false),#(LLVMCalls.vmul_nuw,*,false),(LLVMCalls.vmul_nw,*,false),
                             (LLVMCalls.vrem,%,true),(LLVMCalls.vrem_fast,%,true)]
      for i ∈ -10:10, j ∈ -6:6
        ((j == 0) && (bf === %)) && continue
        @test vf(i%Int8,j%Int8) == bf(i%Int8,j%Int8)
        @test vf(i%UInt8,j%UInt8) == bf(i%UInt8,j%UInt8)
        @test vf(i%Int16,j%Int16) == bf(i%Int16,j%Int16)
        @test vf(i%UInt16,j%UInt16) == bf(i%UInt16,j%UInt16)
        @test vf(i%Int32,j%Int32) == bf(i%Int32,j%Int32)
        @test vf(i%UInt32,j%UInt32) == bf(i%UInt32,j%UInt32)
        @test vf(i%Int64,j%Int64) == bf(i%Int64,j%Int64)
        @test vf(i%UInt64,j%UInt64) == bf(i%UInt64,j%UInt64)
        @test vf(i%Int128,j%Int128) == bf(i%Int128,j%Int128)
        @test vf(i%UInt128,j%UInt128) == bf(i%UInt128,j%UInt128)
      end
      if testfloat
        for i ∈ -1.5:0.4:1.8, j ∈ -3:0.1:3.0
          # `===` for `NaN` to pass
          @test vf(i,j) === bf(i,j)
          @test vf(Float32(i),Float32(j)) === bf(Float32(i),Float32(j))
        end
      end
    end
    let WI = 2
      for I ∈ (Int32,Int64,UInt32,UInt64)
        # TODO: No longer skip these either.
        vi1 = LLVMCalls.VecUnroll((
          Vec(ntuple(_ -> (rand(I)), Val(WI))...),
          Vec(ntuple(_ -> (rand(I)), Val(WI))...),
          Vec(ntuple(_ -> (rand(I)), Val(WI))...),
          Vec(ntuple(_ -> (rand(I)), Val(WI))...)
        ))
        # srange = one(I):(Bool(LLVMCalls.has_feature(Val(:x86_64_avx512dq))) ? I(8sizeof(I)-1) : I(31))
        srange = one(I):(I(31))
        vi2 = LLVMCalls.VecUnroll((
          Vec(ntuple(_ -> rand(srange), Val(WI))...),
          Vec(ntuple(_ -> rand(srange), Val(WI))...),
          Vec(ntuple(_ -> rand(srange), Val(WI))...),
          Vec(ntuple(_ -> rand(srange), Val(WI))...)
        ))
        i = rand(srange); j = rand(I);
        m1 = LLVMCalls.VecUnroll((MM{WI}(I(7)), MM{WI}(I(1)), MM{WI}(I(13)), MM{WI}(I(32%last(srange)))));
        m2 = LLVMCalls.VecUnroll((MM{WI,2}(I(3)), MM{WI,2}(I(8)), MM{WI,2}(I(39%last(srange))), MM{WI,2}(I(17))));
        @test typeof(m1 + I(11)) === typeof(m1)
        @test typeof(m2 + I(11)) === typeof(m2)
        xi1 = tovector(vi1); xi2 = tovector(vi2);
        xi3 =  mapreduce(tovector, vcat, LLVMCalls.data(m1));
        xi4 =  mapreduce(tovector, vcat, LLVMCalls.data(m2));
        # I4 = sizeof(I) < sizeof(I) ? I : (sizeof(I) > sizeof(I) ? I : I)
        for f ∈ [
          +, -, *, ÷, /, %, <<, >>, >>>, ⊻, &, |, fld, mod,
          LLVMCalls.rotate_left, LLVMCalls.rotate_right, copysign, maxi, mini, maxi_fast, mini_fast
          ]
          # for f ∈ [+, -, *, div, ÷, /, rem, %, <<, >>, >>>, ⊻, &, |, fld, mod, LLVMCalls.rotate_left, LLVMCalls.rotate_right, copysign, max, min]
          # @show f, I, I
          # if (!Bool(LLVMCalls.has_feature(Val(:x86_64_avx512dq)))) && (f === /) && sizeof(I) === sizeof(I) === 8
          #     continue
          # end
          # check_within_limits(tovector(@inferred(f(vi1, vi2))),  trunc_int.(f.(size_trunc_int.(xi1, I), size_trunc_int.(xi2, I)), I));
          check_within_limits(tovector(@inferred(f(j, vi2))), trunc_int.(f.(size_trunc_int.(j, I), size_trunc_int.(xi2, I)), I));
          check_within_limits(tovector(@inferred(f(vi1, i))), trunc_int.(f.(size_trunc_int.(xi1, I), size_trunc_int.(i, I)), I));
          check_within_limits(tovector(@inferred(f(m1, i))), trunc_int.(f.(size_trunc_int.(xi3, I), size_trunc_int.(i, I)), I));
          check_within_limits(tovector(@inferred(f(m1, vi2))), trunc_int.(f.(size_trunc_int.(xi3, I), size_trunc_int.(xi2, I)), I));
          check_within_limits(tovector(@inferred(f(m1, m2))), trunc_int.(f.(size_trunc_int.(xi3, I), size_trunc_int.(xi4, I)), I));
          check_within_limits(tovector(@inferred(f(m1, m1))), trunc_int.(f.(size_trunc_int.(xi3, I), size_trunc_int.(xi3, I)), I));
          check_within_limits(tovector(@inferred(f(m2, i))), trunc_int.(f.(size_trunc_int.(xi4, I), size_trunc_int.(i, I)), I));
          check_within_limits(tovector(@inferred(f(m2, vi2))), trunc_int.(f.(size_trunc_int.(xi4, I), size_trunc_int.(xi2, I)), I));
          check_within_limits(tovector(@inferred(f(m2, m2))), trunc_int.(f.(size_trunc_int.(xi4, I), size_trunc_int.(xi4, I)), I));
          check_within_limits(tovector(@inferred(f(m2, m1))), trunc_int.(f.(size_trunc_int.(xi4, I), size_trunc_int.(xi3, I)), I));
          if !((f === LLVMCalls.rotate_left) || (f === LLVMCalls.rotate_right))
            check_within_limits(tovector(@inferred(f(j, m1))), trunc_int.(f.(j, xi3), I));
            # @show 12
            # check_within_limits(tovector(@inferred(f(j, m2))), trunc_int.(f.(size_trunc_int.(j, I), size_trunc_int.(xi4, I)), I));
          end
        end
        @test tovector(@inferred(vi1 ^ i)) ≈ Float64.(xi1) .^ i
        @test @inferred(LLVMCalls.vall(@inferred(1 - MM{WI}(1)) == (1 - Vec(ntuple(identity, Val(WI))...)) ))
      end
      vf1 = LLVMCalls.VecUnroll((
        Vec(ntuple(_ -> (randn()), Val(WI))...),
        Vec(ntuple(_ -> (randn()), Val(WI))...)
      ))
      vf2 = Vec(ntuple(_ -> Core.VecElement(randn()), Val(WI)))
      xf1 = tovector(vf1); xf2 = tovector(vf2); xf22 = vcat(xf2,xf2)
      a = randn();
      for f ∈ [+, -, *, /, %, max, min, copysign, rem, Base.FastMath.max_fast, Base.FastMath.min_fast, Base.FastMath.div_fast, Base.FastMath.rem_fast, Base.FastMath.hypot_fast]
        # @show f
        @test tovector(@inferred(f(vf1, vf2))) ≈ f.(xf1, xf22)
        @test tovector(@inferred(f(a, vf1))) ≈ f.(a, xf1)
        @test tovector(@inferred(f(a, vf2))) ≈ f.(a, xf2)
        @test tovector(@inferred(f(vf1, a))) ≈ f.(xf1, a)
        @test tovector(@inferred(f(vf2, a))) ≈ f.(xf2, a)
      end

      vi2 = LLVMCalls.VecUnroll((
        Vec(ntuple(_ -> (rand(1:M-1)), Val(WI))...),
        Vec(ntuple(_ -> (rand(1:M-1)), Val(WI))...),
        Vec(ntuple(_ -> (rand(1:M-1)), Val(WI))...),
        Vec(ntuple(_ -> (rand(1:M-1)), Val(WI))...)
      ))
      vones, vi2f, vtwos = promote(1.0, vi2, 2f0); # promotes a binary function, right? Even when used with three args?
      @test vones === LLVMCalls.VecUnroll((vbroadcast(Val(WI), 1.0),vbroadcast(Val(WI), 1.0),vbroadcast(Val(WI), 1.0),vbroadcast(Val(WI), 1.0)));
      @test vtwos === LLVMCalls.VecUnroll((vbroadcast(Val(WI), 2.0),vbroadcast(Val(WI), 2.0),vbroadcast(Val(WI), 2.0),vbroadcast(Val(WI), 2.0)));
      @test LLVMCalls.vall(LLVMCalls.collapse_and(vi2f == vi2))
      W32 = StaticInt(WI)*StaticInt(2)
      vf2 = LLVMCalls.VecUnroll((
        Vec(ntuple(_ -> (randn(Float32)), W32)...),
        Vec(ntuple(_ -> (randn(Float32)), W32)...)
      ))
      vones32, v2f32, vtwos32 = promote(1.0, vf2, 2f0); # promotes a binary function, right? Even when used with three args?
      if LLVMCalls.simd_integer_register_size() == LLVMCalls.register_size()
        @test vones32 === LLVMCalls.VecUnroll((vbroadcast(W32, 1f0),vbroadcast(W32, 1f0)))
        @test vtwos32 === LLVMCalls.VecUnroll((vbroadcast(W32, 2f0),vbroadcast(W32, 2f0)))
        @test vf2 === v2f32
      else
        @test vones32 === LLVMCalls.VecUnroll((vbroadcast(W32, 1.0),vbroadcast(W32, 1.0)))
        @test vtwos32 === LLVMCalls.VecUnroll((vbroadcast(W32, 2.0),vbroadcast(W32, 2.0)))
        @test convert(Float64, vf2) === v2f32
      end
      i = rand(1:31)
      m1 = LLVMCalls.VecUnroll((MM{WI}(7), MM{WI}(1), MM{WI}(13), MM{WI}(18)))
      @test tovector(clamp(m1, 2:i)) == clamp.(tovector(m1), 2, i)
      @test tovector(mod(m1, 1:i)) == mod1.(tovector(m1), i)

      @test LLVMCalls.vdivrem.(1:30, 1:30') == divrem.(1:30, 1:30')
      @test LLVMCalls.vcld.(1:30, 1:30') == cld.(1:30, 1:30')
      @test LLVMCalls.vrem.(1:30, 1:30') == rem.(1:30, 1:30')

      @test gcd(Vec(42,64,0,-37), Vec(18,96,-38,0)) === Vec(6,32,38,37)
      @test lcm(Vec(24,16,42,0),Vec(18,12,18,17)) === Vec(72, 48, 126, 0)
    end
  end
  println("Ternary Functions")
  @time @testset "Ternary Functions" begin
    for T ∈ (Float32, Float64)
      v1, v2, v3, m = let W = LLVMCalls.StaticInt(16 ÷ sizeof(T))# @inferred(LLVMCalls.pick_vector_width(T))
        v1 = LLVMCalls.VecUnroll((
          Vec(ntuple(_ -> randn(T), W)...),
          Vec(ntuple(_ -> randn(T), W)...)
        ))
        v2 = LLVMCalls.VecUnroll((
          Vec(ntuple(_ -> randn(T), W)...),
          Vec(ntuple(_ -> randn(T), W)...)
        ))
        v3 = LLVMCalls.VecUnroll((
          Vec(ntuple(_ -> randn(T), W)...),
          Vec(ntuple(_ -> randn(T), W)...)
        ))
        # _W = Int(@inferred(LLVMCalls.pick_vector_width(T)))
        _W = Int(W)
        m = LLVMCalls.VecUnroll((Mask{_W}(rand(UInt16)),Mask{_W}(rand(UInt16))))
        v1, v2, v3, m
      end
      x1 = tovector(v1); x2 = tovector(v2); x3 = tovector(v3);
      a = randn(T); b = randn(T)
      a64 = Float64(a); b64 = Float64(b); # test promotion
      mv = tovector(m)
      for f ∈ [
        muladd, fma, clamp, LLVMCalls.vmuladd_fast, LLVMCalls.vfma_fast,
        LLVMCalls.vfmadd, LLVMCalls.vfnmadd, LLVMCalls.vfmsub, LLVMCalls.vfnmsub,
        LLVMCalls.vfmadd_fast, LLVMCalls.vfnmadd_fast, LLVMCalls.vfmsub_fast, LLVMCalls.vfnmsub_fast,
        LLVMCalls.vfmadd231, LLVMCalls.vfnmadd231, LLVMCalls.vfmsub231, LLVMCalls.vfnmsub231
        ]
        @test tovector(@inferred(f(v1, v2, v3))) ≈ map(f, x1, x2, x3)
        @test tovector(@inferred(f(v1, v2, a64))) ≈ f.(x1, x2, a)
        @test tovector(@inferred(f(v1, a64, v3))) ≈ f.(x1, a, x3)
        @test tovector(@inferred(f(a64, v2, v3))) ≈ f.(a, x2, x3)
        @test tovector(@inferred(f(v1, a64, b64))) ≈ f.(x1, a, b)
        @test tovector(@inferred(f(a64, v2, b64))) ≈ f.(a, x2, b)
        @test tovector(@inferred(f(a64, b64, v3))) ≈ f.(a, b, x3)

        @test tovector(@inferred(LLVMCalls.ifelse(f, m, v1, v2, v3))) ≈ ifelse.(mv, f.(x1, x2, x3), x3)
        @test tovector(@inferred(LLVMCalls.ifelse(f, m, v1, v2, a64))) ≈ ifelse.(mv, f.(x1, x2, a), a)
        @test tovector(@inferred(LLVMCalls.ifelse(f, m, v1, a64, v3))) ≈ ifelse.(mv, f.(x1, a, x3), x3)
        @test tovector(@inferred(LLVMCalls.ifelse(f, m, a64, v2, v3))) ≈ ifelse.(mv, f.(a, x2, x3), x3)
        @test tovector(@inferred(LLVMCalls.ifelse(f, m, v1, a64, b64))) ≈ ifelse.(mv, f.(x1, a, b), b)
        @test tovector(@inferred(LLVMCalls.ifelse(f, m, a64, v2, b64))) ≈ ifelse.(mv, f.(a, x2, b), b)
        @test tovector(@inferred(LLVMCalls.ifelse(f, m, a64, b64, v3))) ≈ ifelse.(mv, f.(a, b, x3), x3)
      end
    end
    # let WI = Int(LLVMCalls.pick_vector_width(Int64))
    let WI = 2
      vi64 = LLVMCalls.VecUnroll((
        Vec(ntuple(_ -> rand(Int64), Val(WI))...),
        Vec(ntuple(_ -> rand(Int64), Val(WI))...),
        Vec(ntuple(_ -> rand(Int64), Val(WI))...)
      ))
      vi32 = LLVMCalls.VecUnroll((
        Vec(ntuple(_ -> rand(Int32), Val(WI))...),
        Vec(ntuple(_ -> rand(Int32), Val(WI))...),
        Vec(ntuple(_ -> rand(Int32), Val(WI))...)
      ))
      xi64 = tovector(vi64); xi32 = tovector(vi32);
      @test tovector(@inferred(LLVMCalls.ifelse(vi64 > vi32, vi64, vi32))) == ifelse.(xi64 .> xi32, xi64, xi32)
      @test tovector(@inferred(LLVMCalls.ifelse(vi64 < vi32, vi64, vi32))) == ifelse.(xi64 .< xi32, xi64, xi32)
      @test tovector(@inferred(LLVMCalls.ifelse(true, vi64, vi32))) == ifelse.(true, xi64, xi32)
      @test tovector(@inferred(LLVMCalls.ifelse(false, vi64, vi32))) == ifelse.(false, xi64, xi32)
      vu64_1 = LLVMCalls.VecUnroll((
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...)
      ))
      vu64_2 = LLVMCalls.VecUnroll((
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...)
      ))
      vu64_3 = LLVMCalls.VecUnroll((
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...)
      ))
      xu1 = tovector(vu64_1); xu2 = tovector(vu64_2); xu3 = tovector(vu64_3);
      for f ∈ [clamp, muladd, LLVMCalls.vfmadd, LLVMCalls.vfnmadd, LLVMCalls.vfmsub, LLVMCalls.vfnmsub]#, LLVMCalls.ifmalo, LLVMCalls.ifmahi
        @test tovector(@inferred(f(vu64_1,vu64_2,vu64_3))) == f.(xu1, xu2, xu3)
      end
    end
  end
    
  end

end
