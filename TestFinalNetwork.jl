include("SimpleNetwork.jl")

TestSimpleNetwork("finalNetwork.jld", 128)
println("L1 of the Network With Only Training On L1 Loss: ")
println(TestSimpleNetwork("testNetworkMAEOnly.jld", 1000, false))
println("L1 of the Final Network With L2 Loss Refinement: ")
println(TestSimpleNetwork("finalNetwork.jld", 1000, false))

