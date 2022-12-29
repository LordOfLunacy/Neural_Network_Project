include("DataAndGenerate.jl")
include("Convolution.jl")
include("AdamUpdate.jl")
include("PixelShuffle.jl")
include("Activations.jl")

#All the ugly structs for the network are found here

mutable struct UNetwork
    #First number signifies what stage of the network
    #1 being the high resolution input stage, 3 being the lowest res stage, and 5 being the high resolution output stage
    conv1_1::Convolution
    conv1_2::Convolution
    conv2_1::Convolution
    conv2_2::Convolution
    conv4_3::Convolution
    conv5_1::Convolution
    conv5_2::Convolution
    conv5_3::Convolution

    function UNetwork(featureChannels::Integer = 4)
        f = featureChannels
        conv1_1 = Convolution(3, 3, 3, f)
        conv1_2 = Convolution(3, 3, f, f)
        conv2_1 = Convolution(3, 3, f, f * 2)
        conv2_2 = Convolution(3, 3, f * 2, f * 2)
        conv4_3 = Convolution(3, 3, f*2, f*4)
        conv5_1 = Convolution(3, 3, f*2, f)
        conv5_2 = Convolution(3, 3, f, f)
        conv5_3 = Convolution(1, 1, f, 3)

        new(conv1_1, conv1_2, conv2_1, conv2_2, conv4_3, conv5_1, conv5_2, conv5_3)
    end
end

mutable struct GradData
    g1_1::Convolution
    g1_2::Convolution
    g2_1::Convolution
    g2_2::Convolution
    g4_3::Convolution
    g5_1::Convolution
    g5_2::Convolution
    g5_3::Convolution

    a1_1::AdamData
    a1_2::AdamData
    a2_1::AdamData
    a2_2::AdamData
    a4_3::AdamData
    a5_1::AdamData
    a5_2::AdamData
    a5_3::AdamData

    function GradData(featureChannels::Integer = 4)
        f = featureChannels
        g1_1 = Convolution(3, 3, 3, f)
        g1_2 = Convolution(3, 3, f, f)
        g2_1 = Convolution(3, 3, f, f * 2)
        g2_2 = Convolution(3, 3, f * 2, f * 2)
        g4_3 = Convolution(3, 3, f*2, f*4)
        g5_1 = Convolution(3, 3, f*2, f)
        g5_2 = Convolution(3, 3, f, f)
        g5_3 = Convolution(1, 1, f, 3)

        a1_1 = AdamData(g1_1)
        a1_2 = AdamData(g1_2)
        a2_1 = AdamData(g2_1)
        a2_2 = AdamData(g2_2)
        a4_3 = AdamData(g4_3)
        a5_1 = AdamData(g5_1)
        a5_2 = AdamData(g5_2)
        a5_3 = AdamData(g5_3)

        new(g1_1, g1_2, g2_1, g2_2, g4_3, g5_1, g5_2, g5_3, a1_1, a1_2, a2_1, a2_2, a4_3, a5_1, a5_2, a5_3)
    end
end

mutable struct UNetTrainData
    #Outputs
    o1_1::Array{Float32, 4}
    #skip connections use a different set of activations (no max pooling)
    o1_2a::Array{Float32, 4}
    o1_2b::Array{Float32, 4}
    o2_1::Array{Float32, 4}
    o2_2a::Array{Float32, 4}
    o2_2b::Array{Float32, 4}
    o4_3::Array{Float32, 4}
    o4_3d2s::Array{Float32, 4}
    o5_1::Array{Float32, 4}
    o5_2::Array{Float32, 4}
    o5_3::Array{Float32, 4}

    #Activation Derivatives
    d1_1::Array{Float32, 4}
    #skip connections use a different set of derivatives
    d1_2a::Array{Float32, 4}
    d1_2b::Array{Float32, 4}
    d2_1::Array{Float32, 4}
    d2_2a::Array{Float32, 4}
    d2_2b::Array{Float32, 4}
    d4_3::Array{Float32, 4}
    d4_3d2s::Array{Float32, 4}
    d5_1::Array{Float32, 4}
    d5_2::Array{Float32, 4}

    #Loss
    e1_1::Array{Float32, 4}
    e1_2::Array{Float32, 4}
    e2_1::Array{Float32, 4}
    e2_2::Array{Float32, 4}
    e4_3::Array{Float32, 4}
    e4Temp::Array{Float32, 4}
    e5_1::Array{Float32, 4}
    e5_2::Array{Float32, 4}
    e5_3::Array{Float32, 4}

    #contains both the initial and Adam gradients
    s::GradData

    function UNetTrainData(net::UNetwork, n::Integer, h::Integer, w::Integer)
        @assert h % 2 == 0
        @assert w % 2 == 0
        f = size(net.conv1_1.weights, 4)

        o1_1 = zeros(Float32, (n, f, h, w))
        o1_2a = zeros(Float32, (n, f, h, w))
        o1_2b = zeros(Float32, (n, f, h ÷ 2, w ÷ 2))
        o2_1 = zeros(Float32, (n, f*2, h÷2, w÷2))
        o2_2a = zeros(Float32, (n, f*2, h÷2, w÷2))
        o2_2b = zeros(Float32, (n, f*2, h÷4, w÷4))
        o4_3 = zeros(Float32, (n, f*4, h÷2, w÷2))
        o4_3d2s = zeros(Float32, (n, f, h, w))
        o5_1 = zeros(Float32, (n, f, h, w))
        o5_2 = zeros(Float32, (n, f, h, w))
        o5_3 = zeros(Float32, (n, 3, h, w))

        d1_1 = zeros(Float32, (n, f, h, w))
        d1_2a = zeros(Float32, (n, f, h, w))
        d1_2b = zeros(Float32, (n, f, h, w))
        d2_1 = zeros(Float32, (n, f*2, h÷2, w÷2))
        d2_2a = zeros(Float32, (n, f*2, h÷2, w÷2))
        d2_2b = zeros(Float32, (n, f*2, h÷2, w÷2))
        d4_3 = zeros(Float32, (n, f*4, h÷2, w÷2))
        d4_3d2s = zeros(Float32, (n, f, h, w))
        d5_1 = zeros(Float32, (n, f, h, w))
        d5_2 = zeros(Float32, (n, f, h, w))

        e1_1 = zeros(Float32, (n, f, h, w))
        e1_2 = zeros(Float32, (n, f, h, w))
        e2_1 = zeros(Float32, (n, f*2, h÷2, w÷2))
        e2_2 = zeros(Float32, (n, f*2, h÷2, w÷2))
        e4_3 = zeros(Float32, (n, f*4, h÷2, w÷2))
        e4Temp = zeros(Float32, (n, f*2, h, w))
        e5_1 = zeros(Float32, (n, f, h, w))
        e5_2 = zeros(Float32, (n, f, h, w))
        e5_3 = zeros(Float32, (n, 3, h, w))

        s = GradData(f)
        new(o1_1, o1_2a, o1_2b, o2_1, o2_2a, o2_2b, o4_3, o4_3d2s, o5_1, o5_2, o5_3, d1_1, d1_2a, d1_2b, d2_1, d2_2a, d2_2b, d4_3, d4_3d2s, d5_1, d5_2, e1_1, e1_2, e2_1, e2_2, e4_3, e4Temp, e5_1, e5_2, e5_3, s)
    end

end

function UNetInference!(batch, net::UNetwork, info::UNetTrainData)
    info.o1_1 = Convolve!(batch, info.o1_1, net.conv1_1)
    info.d1_1 = dReLU!(info.d1_1, info.o1_1)
    info.o1_1 = ReLU!(info.o1_1)

    info.o1_2a = Convolve!(info.o1_1, info.o1_2a, net.conv1_2)
    info.d1_2a = dReLU!(info.d1_2a, info.o1_2a)
    info.d1_2b = dMaxPool!(info.d1_2b, info.o1_2a)
    #info.d1_2b .*= info.d1_2a
    info.o1_2b = MaxPool!(info.o1_2b, info.o1_2a)
    info.o1_2a = ReLU!(info.o1_2a)

    info.o2_1 = Convolve!(info.o1_2b, info.o2_1, net.conv2_1)
    info.d2_1 = dReLU!(info.d2_1, info.o2_1)
    info.o2_1 = ReLU!(info.o2_1)

    info.o2_2a = Convolve!(info.o2_1, info.o2_2a, net.conv2_2)
    info.d2_2a = dReLU!(info.d2_2a, info.o2_2a)
    info.o2_2a = ReLU!(info.o2_2a)
    

    info.o4_3 = Convolve!(info.o2_2a, info.o4_3, net.conv4_3)
    info.d4_3 = dReLU!(info.d4_3, info.o4_3)
    info.o4_3 = ReLU!(info.o4_3)

    info.o4_3d2s = DepthToSpace!(info.o4_3, info.o4_3d2s)
    info.d4_3d2s = DepthToSpace!(info.d4_3, info.d4_3d2s)

    info.o5_1 = Convolve!(cat(info.o1_2a, info.o4_3d2s, dims=2), info.o5_1, net.conv5_1)
    info.d5_1 = dReLU!(info.d5_1, info.o5_1)
    info.o5_1 = ReLU!(info.o5_1)

    info.o5_2 = Convolve!(info.o5_1, info.o5_2, net.conv5_2)
    info.d5_2 = dReLU!(info.d5_2, info.o5_2)
    info.o5_2 = ReLU!(info.o5_2)

    info.o5_3 = Convolve!(info.o5_2, info.o5_3, net.conv5_3)

    info.o5_3 = max.(min.(1, info.o5_3), 0)

    return info
end

function UNetBackprop!(batch, net::UNetwork, info::UNetTrainData)
    info.e5_3 = Float32.((info.o5_3 - batch))
    info.e5_2 = BackConvolve!(info.e5_3, info.e5_2, info.d5_2, net.conv5_3)
    info.e5_1 = BackConvolve!(info.e5_2, info.e5_1, info.d5_1, net.conv5_2)

    #contains both e1_2 data and e4_3 data
    info.e4Temp = BackConvolve!(info.e5_1, info.e4Temp, cat(info.d1_2a, info.d4_3d2s, dims=2), net.conv5_1)

    info.e4_3 = SpaceToDepth!(info.e4Temp[:, size(info.e4Temp, 2)÷2+1:size(info.e4Temp, 2), :, :], info.e4_3)
    info.e2_2 = BackConvolve!(info.e4_3, info.e2_2, info.d2_2a, net.conv4_3)
    info.e2_1 = BackConvolve!(info.e2_2, info.e2_1, info.d2_1, net.conv2_2)

    info.e1_2 = MaxPoolingBackConvolve!(info.e2_1, info.e1_2, info.d1_2b, net.conv2_1)
    #add in error from skip connection
    info.e1_2 = info.e4Temp[:, 1:size(info.e4Temp, 2)÷2, :, :]
    info.e1_1 = BackConvolve!(info.e1_2, info.e1_1, info.d1_1, net.conv1_2)

    return info
end

function UNetUpdate!(batch, net::UNetwork, info::UNetTrainData, stepsize = 0.001)
    info.s.g1_1 = WeightStep!(batch, info.e1_1, info.s.g1_1)
    info.s.g1_2 = WeightStep!(info.o1_1, info.e1_2, info.s.g1_2)
    info.s.g2_1 = WeightStep!(info.o1_2b, info.e2_1, info.s.g2_1)
    info.s.g2_2 = WeightStep!(info.o2_1, info.e2_2, info.s.g2_2)
    info.s.g4_3 = WeightStep!(info.o2_2a, info.e4_3, info.s.g4_3)
    info.s.g5_1 = WeightStep!(cat(info.o1_2a, info.o4_3d2s, dims=2), info.e5_1, info.s.g5_1)
    info.s.g5_2 = WeightStep!(info.o5_1, info.e5_2, info.s.g5_2)
    info.s.g5_3 = WeightStep!(info.o5_2, info.e5_3, info.s.g5_3)

    info.s.a1_1 = AdamUpdate!(info.s.g1_1, info.s.a1_1)
    info.s.a1_2 = AdamUpdate!(info.s.g1_2, info.s.a1_2)
    info.s.a2_1 = AdamUpdate!(info.s.g2_1, info.s.a2_1)
    info.s.a2_2 = AdamUpdate!(info.s.g2_2, info.s.a2_2)
    info.s.a4_3 = AdamUpdate!(info.s.g4_3, info.s.a4_3)
    info.s.a5_1 = AdamUpdate!(info.s.g5_1, info.s.a5_1)
    info.s.a5_2 = AdamUpdate!(info.s.g5_2, info.s.a5_2)
    info.s.a5_3 = AdamUpdate!(info.s.g5_3, info.s.a5_3)

    net.conv1_1 = WeightUpdate!(net.conv1_1, info.s.a1_1, stepsize)
    net.conv1_2 = WeightUpdate!(net.conv1_2, info.s.a1_2, stepsize)
    net.conv2_1 = WeightUpdate!(net.conv2_1, info.s.a2_1, stepsize)
    net.conv2_2 = WeightUpdate!(net.conv2_2, info.s.a2_2, stepsize)
    net.conv4_3 = WeightUpdate!(net.conv4_3, info.s.a4_3, stepsize)
    net.conv5_1 = WeightUpdate!(net.conv5_1, info.s.a5_1, stepsize)
    net.conv5_2 = WeightUpdate!(net.conv5_2, info.s.a5_2, stepsize)
    net.conv5_3 = WeightUpdate!(net.conv5_3, info.s.a5_3, stepsize)

    return net, info
end

function train()
    net = UNetwork(8)
    c = 16
    info = UNetTrainData(net, c, 32, 32)
    noisy = []
    truth = []
    stepsize = 0.1
    
    for j = 1:1
        noisy, truth = generate(50000 + c)
        #if j <= 2
           #noisy = truth
        #end
        for i = 1:c:50000
            info = UNetInference!(noisy[i:i+c-1, :, :, :], net, info)
            info = UNetBackprop!(truth[i:i+c-1, :, :, :], net, info)
            net, info = UNetUpdate!(noisy[i:i+c-1, :, :, :], net, info, stepsize)
            println(sum(abs.(info.o5_3 - truth[i:i+c-1,:,:,:]))/(c*32*32*3))
        end
    end
    noisy, truth = generate(c)
    info = UNetInference!(noisy, net, info)
    output = info.o5_3
    output = max.(0, output)
    output = min.(1, output)

    for i = axes(noisy, 1)
        save("Original$i.png", colorview(RGB, noisy[i, :, :, :]))
        save("Output$i.png", colorview(RGB, output[i, :, :, :]))
        save("Truth$i.png", colorview(RGB, truth[i, :, :, :, :]))
    end
end


train()