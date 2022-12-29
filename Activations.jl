function ReLU!(x)
    
    for i = eachindex(x)
        x[i] = max(x[i], 0.0)
    end
    return x
end

function dReLU!(y, x)
    @assert size(x) == size(y)
    for i = eachindex(x)
        y[i] = (x[i] > 0) ? 1 : 0.0
    end
    return y
end

function MaxPool!(y, x)
    @assert size(y, 3) == size(x, 3) / 2
    @assert size(y, 4) == size(x, 4) / 2
    @assert size(y, 1) == size(x, 1)
    @assert size(y, 2) == size(x, 2)
    for i = CartesianIndices(y)
        cornerY = i[3] * 2 - 1
        cornerX = i[4] * 2 - 1
        y[i] = max(max(x[i[1], i[2], cornerY, cornerX], x[i[1], i[2], cornerY + 1, cornerX]), max(x[i[1], i[2], cornerY, cornerX + 1], x[i[1], i[2], cornerY + 1, cornerX + 1]))
    end
    return y
end

"""function dMaxPool!(y, x)
    @assert size(y) == size(x)
    for i = CartesianIndices((1:size(x, 1), 1:size(x, 2), 1:size(x, 3)÷2, 1:size(x, 4)÷2))
        cornerY = i[3] * 2 - 1
        cornerX = i[4] * 2 - 1
        maximum = max(max(x[i[1], i[2], cornerY, cornerX], x[i[1], i[2], cornerY + 1, cornerX]), max(x[i[1], i[2], cornerY, cornerX + 1], x[i[1], i[2], cornerY + 1, cornerX + 1]))
        y[i[1], i[2], cornerY, cornerX] = (isapprox(maximum, x[i[1], i[2], cornerY, cornerX])) ? 1 : 0
        y[i[1], i[2], cornerY + 1, cornerX] = (isapprox(maximum, x[i[1], i[2], cornerY + 1, cornerX])) ? 1 : 0
        y[i[1], i[2], cornerY, cornerX + 1] = (isapprox(maximum, x[i[1], i[2], cornerY, cornerX + 1])) ? 1 : 0
        y[i[1], i[2], cornerY + 1, cornerX + 1] = (isapprox(maximum, x[i[1], i[2], cornerY + 1, cornerX + 1])) ? 1 : 0
        sum = (y[i[1], i[2], cornerY, cornerX] + y[i[1], i[2], cornerY + 1, cornerX] + y[i[1], i[2], cornerY, cornerX + 1] + y[i[1], i[2], cornerY + 1, cornerX + 1])
        y[i[1], i[2], cornerY, cornerX] /= sum
        y[i[1], i[2], cornerY + 1, cornerX] /= sum
        y[i[1], i[2], cornerY, cornerX + 1] /= sum
        y[i[1], i[2], cornerY + 1, cornerX + 1] /= sum
    end
    return y
end"""
function dMaxPool!(y, x)
    @assert size(y) == size(x)
    for i = CartesianIndices((1:size(x, 1), 1:size(x, 2), 1:size(x, 3)÷2, 1:size(x, 4)÷2))
        cornerY = i[3] * 2 - 1
        cornerX = i[4] * 2 - 1
        maximum = reduce(max, x[i[1], i[2], cornerY:cornerY+1, cornerX:cornerX+1])
        y[i[1], i[2], cornerY:cornerY+1, cornerX:cornerX+1] .= x[i[1], i[2], cornerY:cornerY+1, cornerX:cornerX+1] .== maximum
    end
    y ./= sum(y)
    return y
end