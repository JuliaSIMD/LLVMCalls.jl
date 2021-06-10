using LLVMCalls
using Documenter

DocMeta.setdocmeta!(LLVMCalls, :DocTestSetup, :(using LLVMCalls); recursive=true)

makedocs(;
    modules=[LLVMCalls],
    authors="Chris Elrod <elrodc@gmail.com> and contributors",
    repo="https://github.com/JuliaSIMD/LLVMCalls.jl/blob/{commit}{path}#{line}",
    sitename="LLVMCalls.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaSIMD.github.io/LLVMCalls.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSIMD/LLVMCalls.jl",
)
