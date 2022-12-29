import Pkg;
Pkg.add("Noise")
Pkg.add("ImageCore")
Pkg.add("Images")

using Random, Noise, ImageCore, Images;

function readCIFAR100Images(filename, imageCount)
    io = open(filename, "r")
    #NCHW format for the image tensors
    images = zeros(UInt8, (imageCount, 3, 32, 32))

    for n = axes(images, 1)
        imageBytes = zeros(UInt8, 3074)
        
        readbytes!(io, imageBytes, 3074)
        
        red = imageBytes[3:1026]
        green = imageBytes[1027:2050]
        blue = imageBytes[2051:3074]

        for i = axes(images, 3)
            low = (i-1) * 32 + 1
            high = i * 32
            images[n, 1, i, :] = red[low:high]
            images[n, 2, i, :] = green[low:high]
            images[n, 3, i, :] = blue[low:high]
        end

    end

    return Float32.(images) ./ 255
end

data = readCIFAR100Images("cifar-100-binary/train.bin", 50000)
test = readCIFAR100Images("cifar-100-binary/test.bin", 10000)
rng = Xoshiro()

@enum NoiseType AddGauss=0 MultGauss=1 SAndP=2 Poisson=3 TypeCount=4

function generate(sampleCount)
    truth = zeros(Float32, (sampleCount, size(data, 2), size(data, 3), size(data, 4)))
    sample = zeros(Float32, (sampleCount, size(data, 2), size(data, 3), size(data, 4)))
    truthIndices = rand(1:size(data, 1), sampleCount)
    Threads.@threads for i = axes(truthIndices, 1)
        truth[i, :, :, :] = data[truthIndices[i], :, :, :]
        sample[i, :, :, :] = addNoise(data[truthIndices[i], :, :, :], NoiseType(i % Int64(TypeCount)) )
    end
    #sample = truth
    return (sample, truth)
end

function generateTest(sampleCount)
    truth = zeros(Float32, (sampleCount, size(test, 2), size(test, 3), size(test, 4)))
    sample = zeros(Float32, (sampleCount, size(test, 2), size(test, 3), size(test, 4)))
    truthIndices = rand(1:size(test, 1), sampleCount)
    Threads.@threads for i = axes(truthIndices, 1)
        truth[i, :, :, :] = test[truthIndices[i], :, :, :]
        sample[i, :, :, :] = addNoise(test[truthIndices[i], :, :, :], NoiseType(i % Int64(TypeCount)) )
    end
    #sample = truth
    return (sample, truth)
end

function addNoise(sample, type)
    output = []
    stdDev = 0.15
    img = colorview(RGB, sample)
    if type == AddGauss
        output = add_gauss(img, clip=true, abs(randn() * stdDev), 0.0)
    elseif type == MultGauss
        output = mult_gauss(img, clip=true, abs(randn() * stdDev), 1.0)
    elseif type == SAndP
        output = salt_pepper(img, salt_prob=0.5, salt=1.0, pepper=0.0, abs(randn() * stdDev) * 0.25)
    elseif type == Poisson
        output = poisson(img, 10^rand(1.0:3.0), clip=true)
    end
    return channelview(output)
end

function mosaic()
    sample, truth = generate(16)
    mosaic = zeros(Float32, (size(truth, 2), size(truth, 3) * 16, size(truth, 4)* 2))
    for i = axes(truth, 1)
        low = (i-1) * size(sample, 4) + 1
        high = i * size(sample, 4)
        mosaic[:, low:high, 1:32] = sample[i, :, :, :]
        mosaic[:, low:high, 33:64] = truth[i, :, :, :]
    end
    #save("test.png", colorview(RGB, mosaic))
    return mosaic
end

#creates a mosaic of 128 examples of the noise generation vs. the original
function squareMosaic()
    x = mosaic()
    squareMosaic = zeros(Float32, (size(x, 1), size(x, 2), size(x, 3) * 8))
    squareMosaic[:,:,1:64] = x
    for i = 1:7
        squareMosaic[:,:,64*i + 1:64*(i+1)] = mosaic()
    end
    save("NoisyVsTruth.png", colorview(RGB, squareMosaic))
end
squareMosaic()
    

