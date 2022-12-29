mutable struct AdamData
    mw::Array{Float32, 4}
    vw::Array{Float32, 4}
    mb::Vector{Float32}
    vb::Vector{Float32}
    gradw::Array{Float32, 4}
    gradb::Vector{Float32}
    b1::Float32
    b2::Float32
    eps::Float32
    clipping::Float32

    function AdamData(conv::Convolution, b1::Float32 = Float32(0.9), b2::Float32 = Float32(0.999), eps::Float32 = Float32(1e-8), clipping::Float32 = Float32(1e10))
        h = size(conv.weights, 1)
        w = size(conv.weights, 2)
        i = size(conv.weights, 3)
        o = size(conv.weights, 4)

        mw = zeros(Float32, (h,w,i,o))
        mb = zeros(Float32, o)
        vw = zeros(Float32, (h,w,i,o))
        vb = zeros(Float32, o)
        gradw = zeros(Float32, (h,w,i,o))
        gradb = zeros(Float32, o)

        new(mw, vw, mb, vb, gradw, gradb, b1, b2, eps, clipping)
    end
end

function gradClip(grad, c)
    mag = sqrt(sum(grad .* grad))
    return min(1, c / (mag)) .* grad
end

function AdamUpdate!(inputGrad::Convolution, adam::AdamData)

    inputGrad.weights = gradClip(inputGrad.weights, adam.clipping)
    inputGrad.bias = gradClip(inputGrad.bias, adam.clipping)

    adam.mw .*= adam.b1
    adam.mw += (1 - adam.b1) .* inputGrad.weights
    adam.vw .*= adam.b2
    adam.vw += (1 - adam.b2) .* inputGrad.weights .* inputGrad.weights

    adam.mb .*= adam.b1
    adam.mb += (1 - adam.b1) .* inputGrad.bias
    adam.vb .*= adam.b2
    adam.vb += (1 - adam.b2) .* inputGrad.bias .* inputGrad.bias

    adam.gradw = adam.mw ./ (sqrt.(adam.vw) .+ adam.eps)
    adam.gradb = adam.mb ./ (sqrt.(adam.vb) .+ adam.eps)

    return adam
end

function WeightUpdate!(conv::Convolution, adam::AdamData, stepsize = 0.001)
    conv.weights -= stepsize .* adam.gradw
    conv.bias -= stepsize .* adam.gradb
    return conv
end