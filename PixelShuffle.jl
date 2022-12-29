include("DataAndGenerate.jl")

function SpaceToDepth!(input::Array{Float32, 4}, output::Array{Float32, 4})
    @assert size(input, 1) == size(output, 1)
    @assert size(input, 2) * 4 == size(output, 2)
    @assert size(input, 3) == size(output, 3) * 2
    @assert size(input, 4) == size(output, 4) * 2

    for i = CartesianIndices(input)
        channel = (i[2] - 1) * 4 + (i[3] - 1) % 2 * 2 + (i[4] - 1) % 2 + 1
        output[i[1], channel, ((i[3] - 1) ÷ 2) + 1, ((i[4] - 1) ÷ 2) + 1] = input[i]
    end
    return output
end

function DepthToSpace!(input::Array{Float32, 4}, output::Array{Float32, 4})
    @assert size(input, 1) == size(output, 1)
    @assert size(input, 2) == size(output, 2) * 4
    @assert size(input, 3) * 2 == size(output, 3)
    @assert size(input, 4) * 2 == size(output, 4)
    for i = CartesianIndices(output)
        channel = (i[2] - 1) * 4 + (i[3] - 1) % 2 * 2 + (i[4] - 1) % 2 + 1
        output[i] = input[i[1], channel, ((i[3] - 1) ÷ 2) + 1, ((i[4] - 1) ÷ 2) + 1]
    end
    return output
end

function testShuffle()
    original = data[1:2, :, :, :]

    temp = zeros(Float32, (2, 12, 16, 16))
    output = zeros(Float32, size(original))

    temp = SpaceToDepth!(original, temp)
    output = DepthToSpace!(temp, output)

    for i = axes(original, 1)
        save("Original$i.png", colorview(RGB, original[i, :, :, :]))
        save("Output$i.png", colorview(RGB, output[i, :, :, :]))
    end
end
