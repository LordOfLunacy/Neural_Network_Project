include("DataAndGenerate.jl")
include("Convolution.jl")
include("AdamUpdate.jl")
include("Activations.jl")

import Pkg
Pkg.add("JLD")
using JLD

mutable struct SimpleNet
    conv1::Convolution
    conv2::Convolution
    conv3::Convolution
    #ignore conv4, this was accidentally left in, and I can't remove it now without breaking compatibility with already 
    #trained networks
    conv4::Convolution

    function SimpleNet(featureChannels::Integer = 8)
        f = featureChannels
        conv1 = Convolution(3, 3, 3, f)
        conv2 = Convolution(3, 3, f, f)
        conv3 = Convolution(3, 3, f, 3)
        conv4 = Convolution(3, 3, f, 3)

        new(conv1, conv2, conv3, conv4)
    end
end

mutable struct GradData
    g1::Convolution
    g2::Convolution
    g3::Convolution

    a1::AdamData
    a2::AdamData
    a3::AdamData

    function GradData(associatedNetwork::SimpleNet, clipping::Float32 = Float32(1e10))
        f = size(associatedNetwork.conv1.weights, 4)
        g1 = Convolution(3, 3, 3, f)
        g2 = Convolution(3, 3, f, f)
        g3 = Convolution(3, 3, f, 3)

        a1 = AdamData(g1, Float32(0.9), Float32(0.999), Float32(1e-8), clipping)
        a2 = AdamData(g2, Float32(0.9), Float32(0.999), Float32(1e-8), clipping)
        a3 = AdamData(g3, Float32(0.9), Float32(0.999), Float32(1e-8), clipping)

        new(g1, g2, g3, a1, a2, a3)
    end
end

mutable struct SimpleNetTrainData
    #output data for the network
    o1::Array{Float32, 4}
    o2::Array{Float32, 4}
    o3::Array{Float32, 4}

    #derivative data for the layers
    d1::Array{Float32, 4}
    d2::Array{Float32, 4}

    #error data for the layers
    e1::Array{Float32, 4}
    e2::Array{Float32, 4}
    e3::Array{Float32, 4}

    s::GradData

    function SimpleNetTrainData(net::SimpleNet, n::Integer, h::Integer, w::Integer, clipping::Float32 = Float32(1e10))
        f = size(net.conv1.weights, 4)
        o1 = zeros(Float32, (n, f, h, w))
        o2 = zeros(Float32, (n, f, h, w))
        o3 = zeros(Float32, (n, 3, h, w))

        d1 = zeros(Float32, (n, f, h, w))
        d2 = zeros(Float32, (n, f, h, w))

        e1 = zeros(Float32, (n, f, h, w))
        e2 = zeros(Float32, (n, f, h, w))
        e3 = zeros(Float32, (n, 3, h, w))

        s = GradData(net, clipping)

        new(o1, o2, o3, d1, d2, e1, e2, e3, s)
    end
end

function SimpleNetInference!(batch, net::SimpleNet, info::SimpleNetTrainData)
    Convolve!(batch, info.o1, net.conv1)
    dReLU!(info.d1, info.o1)
    ReLU!(info.o1)

    Convolve!(info.o1, info.o2, net.conv2)
    dReLU!(info.d2, info.o2)
    ReLU!(info.o2)

    Convolve!(info.o2, info.o3, net.conv3)

    #making output bounded between 0 and 1 helps network train easier while not impacting back propagation
    #by binding output between 0 and 1, no loss is caused by things out of image range
    info.o3 = min.(max.(info.o3, 0.0), 1.0)

    return info
end

function SimpleNetBackprop!(batch, net::SimpleNet, info::SimpleNetTrainData)
    info.e3 = Float32.((info.o3 - batch))
    BackConvolve!(info.e3, info.e2, info.d2, net.conv3)
    BackConvolve!(info.e2, info.e1, info.d1, net.conv2)

    return info
end

function SimpleNetUpdate!(batch, net::SimpleNet, info::SimpleNetTrainData, stepsize = 0.001)
    WeightStep!(batch, info.e1, info.s.g1)
    WeightStep!(info.o1, info.e2, info.s.g2)
    WeightStep!(info.o2, info.e3, info.s.g3)

    AdamUpdate!(info.s.g1, info.s.a1)
    AdamUpdate!(info.s.g2, info.s.a2)
    AdamUpdate!(info.s.g3, info.s.a3)

    WeightUpdate!(net.conv1, info.s.a1, stepsize)
    WeightUpdate!(net.conv2, info.s.a2, stepsize)
    WeightUpdate!(net.conv3, info.s.a3, stepsize)

    return net, info
end


function train(filename::String = "Trained/testNetwork.jld", featureCount::Integer = 32, batchSize::Integer = 64, epochs::Integer = 1, stepsize::Float32 = Float32(0.001), clipping::Float32=Float32(1e10))
    net = SimpleNet(featureCount)
    c = batchSize
    info = SimpleNetTrainData(net, c, 32, 32, clipping)
    noisy = []
    truth = []
    
    #not exactly epochs as new training data is generated rather than iterating over the same training data, however,
    #for lack of a better word
    @time for j = 1:epochs
        noisy, truth = generate(50000 + c)
        @time for i = 1:c:50000
            info = SimpleNetInference!(noisy[i:i+c-1, :, :, :], net, info)
            info = SimpleNetBackprop!(truth[i:i+c-1, :, :, :], net, info)
            net, info = SimpleNetUpdate!(noisy[i:i+c-1, :, :, :], net, info, stepsize)
            println(sum(abs.(info.o3 - truth[i:i+c-1,:,:,:]))/(c*32*32*3))
        end
    end
    save(filename, "net", net)
    println("Test Set MAE: ")
    println(TestSimpleNetwork(filename, 1000, false))

    TestSimpleNetwork(filename, 128)
end

function resumeTrain(filename::String = "Trained/testNetwork.jld", batchSize::Integer = 64, epochs::Integer = 1, stepsize::Float32 = Float32(0.0001), clipping::Float32=Float32(1e10))
    net = load(filename, "net")
    c = batchSize
    info = SimpleNetTrainData(net, c, 32, 32, clipping)
    noisy = []
    truth = []
    
    #not exactly epochs as new training data is generated rather than iterating over the same training data, however,
    #for lack of a better word
    @time for j = 1:epochs
        noisy, truth = generate(50000 + c)
        @time for i = 1:c:50000
            info = SimpleNetInference!(noisy[i:i+c-1, :, :, :], net, info)
            info = SimpleNetBackprop!(truth[i:i+c-1, :, :, :], net, info)
            net, info = SimpleNetUpdate!(noisy[i:i+c-1, :, :, :], net, info, stepsize)
            println(sum(abs.(info.o3 - truth[i:i+c-1,:,:,:]))/(c*32*32*3))
        end
    end
    save(filename, "net", net)

    println(TestSimpleNetwork(filename, 1000, false))

    TestSimpleNetwork(filename, 128)
end

function TestSimpleNetwork(filename::String, c::Integer, saveImages::Bool = true)
    net = load(filename, "net")
    info = SimpleNetTrainData(net, c, 32, 32)

    noisy, truth = generateTest(c)
    info = SimpleNetInference!(noisy, net, info)
    output = info.o3
    output = max.(0, output)
    output = min.(1, output)

    if saveImages
        images = zeros(Float32, (size(output, 1), size(output, 2), size(output, 3), size(output, 4) * 3))
        images[:, :, :, 1:size(output, 4)] = noisy
        images[:, :, :, size(output, 4)+1:(size(output, 4) * 2)] = output
        images[:, :, :, (size(output, 4) * 2)+1:size(output, 4)*3] = truth
        for i = axes(noisy, 1)
            save("Trained/Comparison$i.png", colorview(RGB, images[i, :, :, :, :]))
        end
    end
    return (sum(abs.(output - truth)) / (c * 32 * 32 * 3))
end


#train()
#resumeTrain()
#println(TestSimpleNetwork("testNetworkMAEOnly.jld", 1000, false))
#println(TestSimpleNetwork("Trained/testNetwork.jld", 1000, false))
