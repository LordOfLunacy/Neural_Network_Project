
mutable struct Convolution
    using Random
    weights::Array{Float32, 4}
    weights2::Array{Float32, 4}
    bias::Vector{Float32}
    function Convolution(height::Integer, width::Integer, inChannels::Integer, outChannels::Integer)
        #use He initialization for the weights
        inputCount = Float32(width * height * inChannels)
        weights = randn(Float32, (height, width, inChannels, outChannels)) * sqrt(1.0 / (inputCount))
        weights2 = permutedims(weights, [4, 3, 1, 2])
        bias = randn(Float32, outChannels) * sqrt(2.0 / (inputCount))
        new(Float32.(weights), Float32.(weights2), Float32.(bias))
    end

    function Convolution(weights::Array{Float32, 4}, bias::Vector{Float32})
        @assert size(weights, 4) == size(bias, 1)
        weights2 = permutedims(weights, [4, 3, 1, 2])
        new(Float32.(weights), Float32.(weights2), Float32.(bias))
    end
end

#require both the input and the output to be passed in to avoid memory allocations
#input and output are to be in NCHW format
#enforce use of Float32 on this step as its not important to be high precision, but will greatly boost performance
function Convolve!(input::Array{Float32, 4}, output::Array{Float32, 4}, conv::Convolution)
    @assert size(input, 2) == size(conv.weights, 3)
    @assert size(output, 2) == size(conv.weights, 4)
    @assert size(input, 3) == size(output, 3)
    @assert size(input, 4) == size(output, 4)
    @assert size(input, 1) == size(output, 1)
    
    
    
    yPad = Int64(floor(size(conv.weights, 1)/2))
    xPad = Int64(floor(size(conv.weights, 2)/2))
 
    conv.weights2 = permutedims!(conv.weights2, conv.weights, [4, 3, 1, 2])

    output .= 0
    lowHD = 0
    lowWD = 0
    highHD = 0
    highWD = 0
    a = CartesianIndices(output)
    b = CartesianIndices(output)
    for a in CartesianIndices(output)
        n = a[1]
        o = a[2]
        y = a[3]
        x = a[4]
        lowH = max(a[3] - yPad, 1)
        highH = min(size(conv.weights, 1) + a[3] - yPad - 1, size(output, 3))
        lowW = max(a[4] - xPad, 1)
        highW = min(size(conv.weights, 2) + a[4] - xPad - 1, size(output, 4))

        height = highH - lowH
        width = highW - lowW
        lowHD = 1 + -min(0, a[3] - yPad - 1)
        highHD = lowHD + height
        lowWD = 1 + -min(0, a[4] - xPad - 1)
        highWD = lowWD + width

        #println("$lowH,$highH, $lowW, $highW")
        #println("$lowHD,$highHD, $lowWD, $highWD")
        for b in CartesianIndices((1:size(conv.weights2, 2), 1:(height+1), 1:(width+1)))
            @inbounds @fastmath output[a] += input[a[1], b[1], lowH + b[2] - 1, lowW + b[3] - 1] * conv.weights2[a[2], b[1], lowHD + b[2] - 1, lowWD + b[3] - 1]
        end
    end

    #4312
    #permutedims!(conv.weights, conv.weights, [3, 4, 2, 1])
    for i = axes(output, 2)
        output[:, i, :, :] .+= conv.bias[i]
    end

    return output
end



function BackConvolve!(input::Array{Float32, 4}, output::Array{Float32, 4}, derivatives::Array{Float32, 4}, conv::Convolution)
    @assert size(input, 2) == size(conv.weights, 4)
    @assert size(output, 2) == size(conv.weights, 3)
    @assert size(input, 3) == size(output, 3)
    @assert size(input, 4) == size(output, 4)
    @assert size(input, 1) == size(output, 1)
    @assert size(output, 1) == size(derivatives, 1)
    @assert size(output, 2) == size(derivatives, 2)
    @assert size(output, 3) == size(derivatives, 3)
    @assert size(output, 4) == size(derivatives, 4)
    
    output .= 0
    
    yPad = Int64(floor(size(conv.weights, 1)/2)) + 1
    xPad = Int64(floor(size(conv.weights, 2)/2)) + 1

    conv.weights2 = permutedims!(conv.weights2, conv.weights, [4, 3, 1, 2])
    conv.weights2 = reverse!(conv.weights2, dims=(3, 4))

    output .= 0
    lowHD = 0
    lowWD = 0
    highHD = 0
    highWD = 0
    a = CartesianIndices(output)
    b = CartesianIndices(output)


    for a = CartesianIndices(output)
        n = a[1]
        i = a[2]
        y = a[3]
        x = a[4]
        lowH = max(y - yPad, 1)
        highH = min(size(conv.weights, 1) + y - yPad - 1, size(output, 3))
        lowW = max(x - xPad, 1)
        highW = min(size(conv.weights, 2) + x - xPad - 1, size(output, 4))

        height = highH - lowH
        width = highW - lowW
        lowHD = 1 + -min(0, y - yPad - 1)
        highHD = lowHD + height
        lowWD = 1 + -min(0, x - xPad - 1)
        highWD = lowWD + width

        #println("$lowH,$highH, $lowW, $highW")
        #println("$lowHD,$highHD, $lowWD, $highWD")

        for b in CartesianIndices((1:size(conv.weights2, 1), 1:(height+1), 1:(width+1)))
            @inbounds @fastmath output[a] += input[n, b[1], lowH + b[2] - 1, lowW + b[3] - 1] * conv.weights2[b[1], i,lowHD + b[2] - 1, lowWD + b[3] - 1]
        end
    end

    return output .* derivatives
end

function MaxPoolingBackConvolve!(input::Array{Float32, 4}, output::Array{Float32, 4}, derivatives::Array{Float32, 4}, conv::Convolution)
    @assert size(input, 2) == size(conv.weights, 4)
    @assert size(output, 2) == size(conv.weights, 3)
    @assert size(input, 3) * 2 == size(output, 3)
    @assert size(input, 4) * 2 == size(output, 4)
    @assert size(input, 1) == size(output, 1)
    @assert size(output, 1) == size(derivatives, 1)
    @assert size(output, 2) == size(derivatives, 2)
    @assert size(output, 3) == size(derivatives, 3)
    @assert size(output, 4) == size(derivatives, 4)
    
    output .= 0
    
    yPad = Int64(floor(size(conv.weights, 1)/2)) + 1
    xPad = Int64(floor(size(conv.weights, 2)/2)) + 1

    conv.weights2 = permutedims!(conv.weights2, conv.weights, [4, 3, 1, 2])
    conv.weights2 = reverse!(conv.weights2, dims=(3, 4))

    output .= 0
    lowHD = 0
    lowWD = 0
    highHD = 0
    highWD = 0
    a = CartesianIndices(output)
    b = CartesianIndices(output)

    for a = CartesianIndices((1:size(output,1), 1:size(output,2), 1:2:size(output,3), 1:2:size(output,4)))
        n = a[1]
        i = a[2]
        y = (a[3]-1)÷2 + 1
        x = (a[4]-1)÷2 + 1
        lowH = max(y - yPad, 1)
        highH = min(size(conv.weights, 1) + y - yPad - 1, size(output, 3))
        lowW = max(x - xPad, 1)
        highW = min(size(conv.weights, 2) + x - xPad - 1, size(output, 4))

        height = highH - lowH
        width = highW - lowW
        lowHD = 1 + -min(0, y - yPad - 1)
        highHD = lowHD + height
        lowWD = 1 + -min(0, x - xPad - 1)
        highWD = lowWD + width

        #println("$lowH,$highH, $lowW, $highW")
        #println("$lowHD,$highHD, $lowWD, $highWD")

        for b in CartesianIndices((1:size(conv.weights2, 1), 1:(height+1), 1:(width+1)))
            @inbounds @fastmath output[a] += input[n, b[1], lowH + b[2] - 1, lowW + b[3] - 1] * conv.weights2[b[1], i,lowHD + b[2] - 1, lowWD + b[3] - 1]
        end
        output[a[1], a[2], a[3] + 1, a[4]] = output[a]
        output[a[1], a[2], a[3] + 1, a[4] + 1] = output[a]
        output[a[1], a[2], a[3], a[4] + 1] = output[a]
    end

    return output .* derivatives
end

function WeightStep!(input::Array{Float32, 4}, deltas::Array{Float32, 4}, conv::Convolution)
    @assert size(input, 3) == size(deltas, 3)
    @assert size(input, 4) == size(deltas, 4)
    @assert size(input, 1) == size(deltas, 1)
    

    yPad = Int64(floor(size(conv.weights, 1)/2))
    xPad = Int64(floor(size(conv.weights, 2)/2))

    conv.weights = zeros(size(conv.weights))
    lowHD = 0
    lowWD = 0
    highHD = 0
    highWD = 0
    a = CartesianIndices(conv.weights)
    b = CartesianIndices(conv.weights)
    for a = CartesianIndices(conv.weights)
        h = a[1]
        w = a[2]
        i = a[3]
        o = a[4]
        lowH = max(h - yPad, 1)
        highH = min(size(input, 3) - size(conv.weights, 1) + h + yPad, size(input, 3))
        lowW = max(w - xPad, 1)
        highW = min(size(input, 4) - size(conv.weights, 2) + w + xPad, size(input, 4))
        lowHD = max(-h + size(conv.weights, 1), 1)
        highHD = min(size(input, 3) - h + yPad + 1, size(input, 3))
        lowWD = max(-w + size(conv.weights, 2), 1)
        highWD = min(size(input, 4) - w + xPad + 1, size(input, 4))

        offsetW = lowWD - lowW
        offsetH = lowHD - lowH
        #println("$lowH,$highH, $lowW, $highW")
        #println("$lowHD,$highHD, $lowWD, $highWD")
        for b in CartesianIndices((1:size(deltas, 1), lowHD:highHD, lowWD:highWD))
            @inbounds @fastmath conv.weights[a] += input[b[1], i, b[2] - offsetH, b[3] - offsetW] * deltas[b[1], o,b[2], b[3]]
        end
                    
        #@inbounds conv.weights[a] = sum(@view(input[:, i, lowH:highH, lowW:highW]) .* @view(deltas[:, o, lowHD:highHD, lowWD:highWD]))
    end

    for i = axes(conv.bias, 1)
        conv.bias[i] = sum(deltas[:, i, :, :])
    end

    denominator = size(deltas, 1) * size(deltas, 3) * size(deltas, 4)
    conv.weights ./= Float32.(denominator)
    conv.bias ./= Float32.(denominator)

    return conv

end

