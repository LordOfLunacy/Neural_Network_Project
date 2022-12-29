include("SimpleNetwork.jl")

filename = "SimpleNetwork.jld"
featureCount = 8
batchSize = 64
epochs = 1
stepsize = 0.001
clipping = 1e10

train(filename, featureCount, batchSize, epochs, Float32(stepsize), Float32(clipping))